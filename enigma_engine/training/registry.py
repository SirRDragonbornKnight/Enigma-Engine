"""Training mode registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ModeSpec:
    """Metadata for a training mode."""

    name: str
    experimental: bool
    description: str


def get_mode_registry() -> Dict[str, ModeSpec]:
    """Return mode metadata for dispatcher validation and UX surfacing."""
    return {
        "sft": ModeSpec("sft", False, "Supervised fine-tuning"),
        "dpo": ModeSpec("dpo", False, "Direct Preference Optimization"),
        "grpo": ModeSpec("grpo", False, "Group Relative Policy Optimization"),
        "lora": ModeSpec("lora", False, "LoRA adapter training"),
        "vision": ModeSpec("vision", False, "Vision encoder/projection training"),
        "audio": ModeSpec("audio", False, "Audio encoder/projection training"),
        "simpo": ModeSpec("simpo", True, "Reference-free preference optimization"),
        "kto": ModeSpec("kto", True, "Kahneman-Tversky Optimization"),
        "orpo": ModeSpec("orpo", True, "Odds Ratio Preference Optimization"),
        "rest": ModeSpec("rest", True, "Reinforced Self-Training loop"),
        "reward_model": ModeSpec("reward_model", True, "Reward model training"),
        "rlhf": ModeSpec("rlhf", True, "RLHF PPO-style policy training"),
        "self_play": ModeSpec("self_play", True, "Self-play policy training"),
        "remax": ModeSpec("remax", True, "ReMax policy training"),
    }
