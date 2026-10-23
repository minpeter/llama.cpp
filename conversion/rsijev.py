"""rsi-jev (shgao/rsi-jev-v6.1-vl-4b): a Qwen3.5-4B tower (full fine-tune, 32 layers, no MTP, no lm_head) released as
`tower.safetensors` with bare tensor names (`layers.N.*`, `embed_tokens.weight`, `norm.weight`), plus the option_xattn
scorers (`scorer.safetensors`, `aux_scorers.safetensors`) and calibration, which stay in the engine (torch fp32).

The GGUF is the plain text tower: the engine reads the final-norm hidden state of the prompt tokens via `/embedding`
(pooling none) and the residual stream after layers 16 and 20 (`hidden` of the same request, server patch) for the
adaptive exits, then runs the scorers itself. The output head is tied to the embedding; it is never read.

Point the converter at a staging directory that holds the snapshot files and `model.safetensors -> tower.safetensors`
(the base converter only reads `model*.safetensors`).
"""
from __future__ import annotations

import json

from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf, logger
from .qwen import Qwen3_5TextModel


def _is_rsijev_checkpoint(dir_model: Path) -> bool:
    return all((dir_model / name).is_file() for name in ("meta.json", "config.json", "scorer.safetensors", "tower.safetensors"))


@ModelBase.register_hparams_loader(_is_rsijev_checkpoint)
def _load_rsijev_hparams(dir_model: Path) -> dict[str, Any]:
    logger.info("gguf: detected rsi-jev checkpoint")
    hparams = ModelBase.load_hparams(dir_model, False, guess=False)
    hparams["architectures"] = ["RsiJevModel"]
    with open(dir_model / "meta.json", encoding="utf-8") as f:
        hparams["rsijev_meta"] = json.load(f)
    return hparams


@ModelBase.register("RsiJevModel")
@ModelBase.example("shgao/rsi-jev-v6.1-vl-4b")
class RsiJevModel(Qwen3_5TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN35
    no_mtp = True  # the tower has no MTP head

    def __init__(self, dir_model: Path, *args, **kwargs):
        super().__init__(dir_model, *args, **kwargs)
        meta = self.hparams["rsijev_meta"]
        spec = meta["spec"]
        assert spec["readout"] == "option_xattn" and spec["layout"] == "state_first", spec
        assert int(spec["arch_extra"]["exit_layer"]) == self.hparams["num_hidden_layers"], spec["arch_extra"]
        assert self.hparams.get("tie_word_embeddings", False), "the tower has no lm_head; the output must be tied"
        self.seen = {"embed": False, "norm": False}

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        # the tower file has bare names
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
