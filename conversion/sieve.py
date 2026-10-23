"""Sieve decision models (sthanika-ai) released as LoRA adapters on Qwen3.5, converted with the adapter merged.

SieveCodesModel  Sieve-4B: sieve_config.json + adapter + head.safetensors (readout.weight, 255 x d).
                 The 255 option-code rows of the (tied) output head are replaced by the readout, so the logits of the
                 code tokens at a branch's last position are the model's option logits (softmax over the K listed codes
                 cancels the rest of the vocabulary). Prompting, temperatures and answer assembly stay in the engine.
SieveForkModel   Sieve-2B-v2 (sieve-hybrid-fork): config.json {base_model, read_layer, lora, ...} + adapter + head.
                 The base is truncated to read_layer blocks and the final norm weight is set to 1 (zero-centered: 0 + 1),
                 so `--embeddings --pooling last` returns h / rms(h) of the read layer at the last token; the scalar head
                 (LayerNorm -> Linear -> GELU -> Linear) is scale-invariant in its LayerNorm and is applied by the engine.
"""
from __future__ import annotations

import json
import re

from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, ModelBase, gguf, logger
from .lev import _DecisionLoraMixin, _load_decision_lora_hparams
from .qwen import Qwen3_5TextModel


def _is_sieve_codes_checkpoint(dir_model: Path) -> bool:
    return all((dir_model / name).is_file() for name in ("sieve_config.json", "adapter_config.json", "head.safetensors"))


@ModelBase.register_hparams_loader(_is_sieve_codes_checkpoint)
def _load_sieve_codes_hparams(dir_model: Path) -> dict[str, Any]:
    logger.info("gguf: detected Sieve (codes) checkpoint")
    return _load_decision_lora_hparams(dir_model, "SieveCodesModel")


@ModelBase.register("SieveCodesModel")
@ModelBase.example("sthanika-ai/Sieve-4B")
class SieveCodesModel(_DecisionLoraMixin, Qwen3_5TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN35

    def __init__(self, dir_model: Path, *args, **kwargs):
        super().__init__(dir_model, *args, **kwargs)
        from safetensors.torch import load_file
        with open(dir_model / "sieve_config.json", encoding="utf-8") as f:
            self.sieve_cfg = json.load(f)
        assert not self.sieve_cfg.get("pointer"), "the pointer ablation is not supported"
        self.readout = load_file(dir_model / "head.safetensors")["readout.weight"].float()
        self.code_ids = torch.tensor(self.sieve_cfg["code_ids"], dtype=torch.long)
        assert self.readout.shape[0] == len(self.code_ids) == 255
        self.output_written = False

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        assert not name.endswith("lm_head.weight"), "untied output head: the readout must replace its rows instead"
        if name.endswith("embed_tokens.weight"):
            assert not self.output_written
            emb = LazyTorchTensor.to_eager(data_torch).float()
            assert emb.shape[1] == self.readout.shape[1]
            out = emb.clone()
            out[self.code_ids] = self.readout
            self.output_written = True
            yield "output.weight", out
        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()
        if not self.output_written:
            raise ValueError("the token embedding was not seen, no output head with the readout was written")


def _is_sieve_fork_checkpoint(dir_model: Path) -> bool:
    if not all((dir_model / name).is_file() for name in ("config.json", "adapter_model.safetensors", "head.safetensors")):
        return False
    with open(dir_model / "config.json", encoding="utf-8") as f:
        return json.load(f).get("architecture") == "sieve-hybrid-fork"


@ModelBase.register_hparams_loader(_is_sieve_fork_checkpoint)
def _load_sieve_fork_hparams(dir_model: Path) -> dict[str, Any]:
    logger.info("gguf: detected Sieve (fork) checkpoint")
    return {"architectures": ["SieveForkModel"]}


@ModelBase.register("SieveForkModel")
@ModelBase.example("sthanika-ai/Sieve-2B-v2")
class SieveForkModel(Qwen3_5TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN35
    no_mtp = True

    def __init__(self, dir_model: Path, *args, **kwargs):
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file

        with open(dir_model / "config.json", encoding="utf-8") as f:
            self.sieve_cfg = json.load(f)
        assert self.sieve_cfg.get("architecture") == "sieve-hybrid-fork"
        assert not (dir_model / "full_weights.safetensors").is_file(), "full fine-tune releases are not supported"
        self.read_layer = int(self.sieve_cfg["read_layer"])

        logger.info(f"gguf: downloading the base model {self.sieve_cfg['base_model']}")
        dir_base = Path(snapshot_download(self.sieve_cfg["base_model"], allow_patterns=["*.json", "*.jinja", "*.safetensors"]))

        # the base, truncated to the read layer
        hparams = kwargs.pop("hparams", None) or ModelBase.load_hparams(dir_base, False)
        for h in (hparams, hparams.get("text_config")):
            if not isinstance(h, dict) or "num_hidden_layers" not in h:
                continue
            assert h["num_hidden_layers"] >= self.read_layer
            h["num_hidden_layers"] = self.read_layer
            if "layer_types" in h:
                h["layer_types"] = h["layer_types"][:self.read_layer]
            h["mtp_num_hidden_layers"] = 0
            h["tie_word_embeddings"] = True  # the output head is never read, the embedding stands in for it
        super().__init__(dir_base, *args, hparams=hparams, **kwargs)
        self.dir_adapter = dir_model
        self.dir_model_card = dir_model

        lora_cfg = self.sieve_cfg["lora"]
        self.lora_scale = lora_cfg["lora_alpha"] / lora_cfg["r"]
        self.lora_targets = re.compile(lora_cfg["target_modules"])
        adapter = lora_cfg.get("adapter_name", "default")
        # "layers.0.mlp.up_proj.lora_A.s0.weight" -> self.lora["layers.0.mlp.up_proj.weight"]["A"]
        self.lora: dict[str, dict[str, Tensor]] = {}
        for name, tensor in load_file(dir_model / "adapter_model.safetensors").items():
            base_name, _, part = name[name.index("layers."):].partition(".lora_")
            assert part in (f"A.{adapter}.weight", f"B.{adapter}.weight"), f"unexpected LoRA tensor: {name}"
            assert self.lora_targets.search(base_name), f"LoRA tensor outside the target modules: {name}"
            self.lora.setdefault(base_name + ".weight", {})[part[0]] = tensor.float()
        self.lora_merged: set[str] = set()
        self.norm_written = False

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.endswith("lm_head.weight") or "mtp." in name:
            return
        if bid is not None and bid >= self.read_layer:
            return  # past the read layer: never run
        if "layers." in name:
            lora = self.lora.get(name[name.index("layers."):])
            if lora is not None:
                assert set(lora) == {"A", "B"} and data_torch.shape == (lora["B"].shape[0], lora["A"].shape[1])
                delta = self.lora_scale * (lora["B"] @ lora["A"])
                data_torch = data_torch.float() + LazyTorchTensor.from_eager(delta)
                self.lora_merged.add(name[name.index("layers."):])
        elif name.endswith("norm.weight"):
            # final norm: weight 1 (Qwen3Next stores zero-centered norms, the base converter adds 1)
            data_torch = torch.zeros(data_torch.shape, dtype=torch.float32)
            self.norm_written = True
        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()
        if len(self.lora_merged) != len(self.lora):
            raise ValueError(f"only {len(self.lora_merged)} of {len(self.lora)} LoRA tensors were merged into the base model")
        if not self.norm_written:
            raise ValueError("the final norm was not seen")

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_pooling_type(gguf.PoolingType.LAST)
