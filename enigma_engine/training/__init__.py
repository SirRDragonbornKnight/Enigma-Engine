"""Training package.

The Forge-era dispatcher (schema/registry/dispatch + the monolithic
Trainer) was retired in the 2026-07-18 compression pass. What remains
is the vision-align trainer used by align_vision.py; the live text
paths are the bespoke root scripts (pretrain_enigma.py,
finetune_enigma.py, dpo_enigma.py).

Re-exports are LAZY on purpose: importing this package must not pull in
torch (same rule as ``enigma_engine/core/__init__.py``). Callers that
want the trainer should prefer the direct module import:

    from enigma_engine.training.vision_align import Trainer, TrainingConfig
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, never executed at runtime
    from .vision_align import Trainer, TrainingConfig, TrainingState

__all__ = [
    "Trainer",
    "TrainingConfig",
    "TrainingState",
]


def __getattr__(name: str) -> Any:
    if name in set(__all__):
        from . import vision_align

        return getattr(vision_align, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
