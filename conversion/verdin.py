"""Verdin-4B (victorbennett/Verdin-4B): Qwen3.5-4B text tower with a merged LoRA, released as `model.safetensors` with
bare tensor names (`layers.N.*`, `embed_tokens.weight`, `norm.weight`) and a `verdin_config.json` instead of an HF
config. The pointer head (`head.safetensors`, fp32) stays in the engine (torch): the GGUF is the plain text tower whose
final-norm hidden states are read through `/embedding` (pooling none, `embd_reads` server patch). The state-
bidirectional attention of the full-attention layers (`architecture.state_bidir`) is a request-time mask (server patch
`state_bidir`), it is not stored in the GGUF.

Point the converter at a staging directory holding `model.safetensors`, `verdin_config.json`, the tokenizer files and a
Qwen3.5-4B `config.json` (the converter needs an HF config; its architecture must match `verdin_config.json`, checked).
"""
from __future__ import annotations

import json

from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf, logger
from .qwen import Qwen3_5TextModel


def _is_verdin_checkpoint(dir_model: Path) -> bool:
    f = dir_model / "verdin_config.json"
    if not (f.is_file() and (dir_model / "config.json").is_file()):
        return False
    with open(f, encoding="utf-8") as fh:
        return json.load(fh).get("format") == "verdin"


@ModelBase.register_hparams_loader(_is_verdin_checkpoint)
def _load_verdin_hparams(dir_model: Path) -> dict[str, Any]:
    logger.info("gguf: detected Verdin checkpoint")
    hparams = ModelBase.load_hparams(dir_model, False, guess=False)
    hparams["architectures"] = ["VerdinModel"]
    with open(dir_model / "verdin_config.json", encoding="utf-8") as f:
        hparams["verdin"] = json.load(f)
    return hparams


@ModelBase.register("VerdinModel")
@ModelBase.example("victorbennett/Verdin-4B")
class VerdinModel(Qwen3_5TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN35
    no_mtp = True  # the checkpoint has no MTP head

    def __init__(self, dir_model: Path, *args, **kwargs):
        super().__init__(dir_model, *args, **kwargs)
        arch = self.hparams["verdin"]["architecture"]
        tc = self.hparams.get("text_config", self.hparams)
        for k in ("hidden_size", "num_hidden_layers", "intermediate_size", "num_attention_heads", "num_key_value_heads",
                  "head_dim", "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
                  "linear_value_head_dim", "linear_conv_kernel_dim", "vocab_size"):
            assert tc[k] == arch[k], (k, tc[k], arch[k])
        assert list(tc["layer_types"]) == list(arch["layer_types"])
        rope = tc.get("rope_parameters") or {}
        assert rope.get("rope_theta") == arch["rope_theta"] and rope.get("partial_rotary_factor") == arch["partial_rotary_factor"]
        assert self.hparams["verdin"]["weights"].get("lora_merged", False)
        self.seen = {"embed": False, "norm": False}

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if name.startswith(("layers.", "embed_tokens.", "norm.")):
            name = "model." + name
        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("visual.") or "mtp." in name or name.endswith("lm_head.weight"):
            return
        if name == "model.embed_tokens.weight":
            self.seen["embed"] = True
        elif name == "model.norm.weight":
            self.seen["norm"] = True
        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()
        if not all(self.seen.values()):
            raise ValueError(f"tower tensors missing: {self.seen}")
