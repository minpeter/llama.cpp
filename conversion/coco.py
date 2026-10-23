from __future__ import annotations

import json

from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf, jinja_str_or_json, logger
from .pplx_decider import PplxDeciderModel

# option label codes in the order of the reference engine (oh-my-jev semif)
_ALPHABET = [chr(ord("A") + i) for i in range(26)]
_LABELS = (_ALPHABET + [a + b for a in _ALPHABET for b in _ALPHABET])[:255]


def _is_coco_checkpoint(dir_model: Path) -> bool:
    if not all((dir_model / name).is_file() for name in ("decision_config.json", "readout.safetensors", "config.json")):
        return False
    with open(dir_model / "decision_config.json", encoding="utf-8") as f:
        return json.load(f).get("structure") == "noncausal_readout"


@ModelBase.register_hparams_loader(_is_coco_checkpoint)
def _load_coco_hparams(dir_model: Path) -> dict[str, Any]:
    logger.info("gguf: detected CoCo-Decision checkpoint")
    hparams = ModelBase.load_hparams(dir_model, False, guess=False)
    hparams["architectures"] = ["CocoDecisionModel"]
    with open(dir_model / "decision_config.json", encoding="utf-8") as f:
        hparams["decision"] = json.load(f)
    return hparams


@ModelBase.register("CocoDecisionModel")
@ModelBase.example("corners-ai/CoCo-Decision-4B")
class CocoDecisionModel(PplxDeciderModel):
    """CoCo-Decision (corners-ai): Qwen3.5 text backbone, full-attention layers bidirectional, a linear readout over
    option-letter tokens (leading space) applied to the last hidden state. Served like vllm-d3 (non-causal, last
    pooling), with the oh-my-jev "semif" prompt that the public Decision Index submission used."""

    model_arch = gguf.MODEL_ARCH.QWEN35

    _DECISION_TYPE = gguf.DecisionType.VLLM_D3

    def _label_codes(self) -> list[str]:
        # labels whose " <code>" is a single token with a readout row, in the order of the reference engine
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.dir_model)
        table = set(self.hparams["decision"]["readout_token_ids"])
        codes = []
        for code in _LABELS:
            enc = tok.encode(" " + code, add_special_tokens=False)
            if len(enc) == 1 and enc[0] in table:
                codes.append(code)
        return codes

    def _systemone_template(self) -> str:
        description = jinja_str_or_json("o.description")
        # choice: description or the key; noul: yes/no with the criteria text or a fallback; score: the level text
        option = (
            "{% if type == 'noul' %}{{ 'yes' if o.key == 'true' else 'no' }}: "
            "{% if o.description is string and o.description %}{{ o.description }}"
            "{% elif o.description is none or o.description == '' %}"
            "{{ 'The statement is true.' if o.key == 'true' else 'The statement is false.' }}"
            "{% else %}{{ o.description | tojson }}{% endif %}"
            "{% else %}{{ o.key }}: {% if o.description is none %}{{ o.key }}{% else %}" + description + "{% endif %}"
            "{% endif %}"
        )
        return (
            "<|im_start|>user\n"
            "{{ 'State:\\n' }}" + jinja_str_or_json("state") + "{{ '\\n\\nQuestion: ' }}" + jinja_str_or_json("instructions")
            + "{{ '\\nOptions:' }}"
            "{% for o in options %}{{ '\\n' }}{{ o.label }}. " + option + "{% endfor %}"
            "{{ '\\nAnswer with the letter only.<|im_end|>\\n<|im_start|>assistant\\n<think>\\nAnswer:' }}"
        )

    def set_gguf_parameters(self):
        # skip PplxDeciderModel (needs decision.temperature), call its parent
        super(PplxDeciderModel, self).set_gguf_parameters()
        self.gguf_writer.add_decision_type(self._DECISION_TYPE)
        for name in ("choice", "score", "noul"):
            self.gguf_writer.add_decision_temperature(name, 1.0)
        # the full attention layers see the whole prompt
        self.gguf_writer.add_causal_attention(False)
        # the server reads the option labels " A" ... (leading space) in this order, instead of its built-in codes
        codes = self._label_codes()
        self.gguf_writer.add_string(f"{self.gguf_writer.arch}.decision.label_codes", " ".join(codes))
        self.gguf_writer.add_string(f"{self.gguf_writer.arch}.decision.label_prefix", " ")
        self.gguf_writer.add_bool(f"{self.gguf_writer.arch}.decision.noul_true_first", True)
        logger.info(f"gguf: {len(codes)} option labels")

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        yield from super(PplxDeciderModel, self).generate_extra_tensors()
        from safetensors.torch import load_file

        readout = load_file(self.dir_model / "readout.safetensors")["proj.weight"]
        token_ids = self.hparams["decision"]["readout_token_ids"]
        n_vocab = self.hparams["vocab_size"]
        assert readout.shape[0] == len(token_ids) == len(set(token_ids))
        lm_head = torch.zeros(n_vocab, readout.shape[1], dtype=readout.dtype)
        lm_head[token_ids] = readout
        yield "lm_head.weight", lm_head
