"""Training config schema and YAML/JSON loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ModeName = Literal[
    "sft",
    "dpo",
    "grpo",
    "lora",
    "vision",
    "audio",
    "simpo",
    "kto",
    "orpo",
    "rest",
    "reward_model",
    "rlhf",
    "self_play",
    "remax",
    "adaptive",
]


class TrainingOverrides(BaseModel):
    """Subset of core TrainingConfig fields exposed in dispatcher config."""

    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(default=10, ge=1)
    batch_size: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_steps: int = Field(default=100, ge=1)
    gradient_clip: float = Field(default=1.0, ge=0)
    checkpoint_dir: str = "models/checkpoints"
    use_amp: bool = True
    max_grad_accumulation: int = Field(default=1, ge=1)
    min_lr_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    seed: int | None = None
    deterministic: bool = False
    # SFT-specific knobs (mirror TrainingConfig defaults)
    use_gradient_checkpointing: bool = True
    use_sequence_packing: bool = False
    ce_chunk_size: int = Field(default=0, ge=0)
    use_compile: bool = False
    rolling_best_k: int = Field(default=0, ge=0)
    general_mix_ratio: float = Field(default=0.2, ge=0.0, le=1.0)
    general_data: str = ""
    val_split: float = Field(default=0.1, ge=0.0, le=0.5)
    save_every: int = Field(default=0, ge=0)
    run_evaluation: bool = False


class DPOSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    beta: float = Field(default=0.1, gt=0)
    loss_type: Literal["dpo", "apo_zero"] = "dpo"
    dpo_online_interval: int = Field(default=0, ge=0)
    dpo_online_samples: int = Field(default=4, ge=1)


class SimPOSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    beta: float = Field(default=2.5, gt=0)
    gamma: float = Field(default=0.5)


class KTOSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    beta: float = Field(default=0.1, gt=0)
    desirable_weight: float = Field(default=1.0, gt=0)
    undesirable_weight: float = Field(default=1.0, gt=0)


class ORPOSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    beta: float = Field(default=0.1, gt=0)


class ReSTSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rounds: int = Field(default=3, ge=1)
    n_samples: int = Field(default=4, ge=2)
    beta: float = Field(default=0.1, gt=0)
    max_new_tokens: int = Field(default=128, ge=1)


class VisionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unfreeze_text_layers: int = Field(default=0, ge=0)


class AudioSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unfreeze_text_layers: int = Field(default=0, ge=0)


class LoraSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int = Field(default=8, ge=1, le=128)
    alpha: int = Field(default=16, ge=1, le=256)
    output_dir: str = "models/lora_adapters"
    learning_rate: float = Field(default=1e-4, gt=0)
    batch_size: int = Field(default=4, ge=1)
    epochs: int = Field(default=3, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    min_lr_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    max_length: int = Field(default=512, ge=2)


class GRPOSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_size: int = Field(default=4, ge=2)
    max_new_tokens: int = Field(default=128, ge=1)
    temperature: float = Field(default=0.8, gt=0)


class ReMaxSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_responses: int = Field(default=4, ge=2)
    max_new_tokens: int = Field(default=128, ge=1)
    temperature: float = Field(default=0.8, gt=0)


class TrainingJobConfig(BaseModel):
    """Unified training config consumed by the dispatcher."""

    model_config = ConfigDict(extra="forbid")

    mode: ModeName
    data: Any = None
    allow_experimental: bool = False
    resume_from: str | None = None

    training: TrainingOverrides = Field(default_factory=TrainingOverrides)

    dpo: DPOSettings = Field(default_factory=DPOSettings)
    simpo: SimPOSettings = Field(default_factory=SimPOSettings)
    kto: KTOSettings = Field(default_factory=KTOSettings)
    orpo: ORPOSettings = Field(default_factory=ORPOSettings)
    rest: ReSTSettings = Field(default_factory=ReSTSettings)
    vision: VisionSettings = Field(default_factory=VisionSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    lora: LoraSettings = Field(default_factory=LoraSettings)
    grpo: GRPOSettings = Field(default_factory=GRPOSettings)
    remax: ReMaxSettings = Field(default_factory=ReMaxSettings)

    @staticmethod
    def _require_non_empty_string(value: Any, mode: str, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"mode '{mode}' requires non-empty '{field_name}'")
        return value

    @classmethod
    def _validate_preference_rows(cls, mode: str, rows: Any) -> None:
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"mode '{mode}' expects 'data' to be a non-empty list of preference rows")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"mode '{mode}' preference rows must be dict objects")
            cls._require_non_empty_string(row.get("prompt", ""), mode, "prompt")
            cls._require_non_empty_string(row.get("chosen", ""), mode, "chosen")
            cls._require_non_empty_string(row.get("rejected", ""), mode, "rejected")

    @classmethod
    def _validate_multimodal_rows(cls, mode: str, rows: Any, field_name: str = "data") -> None:
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"mode '{mode}' expects '{field_name}' to be a non-empty list of rows")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"mode '{mode}' rows must be dict objects")

    @model_validator(mode="after")
    def _validate_mode_data(self) -> "TrainingJobConfig":
        if self.mode == "sft":
            # Trainer.train accepts raw text OR a list of rows; a .jsonl/.json
            # 'data' path is materialized into a list BEFORE validation runs
            # (materialize_dispatch_payload), so both shapes must pass here.
            if isinstance(self.data, list):
                if not self.data:
                    raise ValueError("mode 'sft' expects 'data' to be non-empty")
            else:
                self._require_non_empty_string(self.data, "sft", "data")

        if self.mode in {"dpo", "simpo", "orpo", "reward_model"}:
            self._validate_preference_rows(self.mode, self.data)

        if self.mode in {"grpo", "remax", "rest", "rlhf", "self_play"}:
            if not isinstance(self.data, list) or not self.data:
                raise ValueError(f"mode '{self.mode}' expects 'data' to be a non-empty list of prompts")
            for prompt in self.data:
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(f"mode '{self.mode}' expects 'data' to be non-empty string prompts")

        if self.mode in {"vision", "audio"}:
            if isinstance(self.data, dict):
                train_rows = self.data.get("train")
                self._validate_multimodal_rows(self.mode, train_rows, field_name="data.train")
                val_rows = self.data.get("val")
                if val_rows is not None and not isinstance(val_rows, list):
                    raise ValueError(f"mode '{self.mode}' expects 'data.val' to be a list when provided")
            else:
                self._validate_multimodal_rows(self.mode, self.data)

        if self.mode == "lora":
            if isinstance(self.data, str):
                self._require_non_empty_string(self.data, "lora", "data")
            elif isinstance(self.data, list) and self.data:
                for row in self.data:
                    if not isinstance(row, dict):
                        raise ValueError("mode 'lora' rows must be dict objects")
                    self._require_non_empty_string(
                        row.get("completion", ""),
                        "lora",
                        "completion",
                    )
            else:
                raise ValueError("mode 'lora' expects 'data' to be a non-empty string or list of rows")

        if self.mode == "kto":
            if not isinstance(self.data, list) or not self.data:
                raise ValueError("mode 'kto' expects 'data' to be a non-empty list of feedback rows")
            for row in self.data:
                if not isinstance(row, dict):
                    raise ValueError("mode 'kto' feedback rows must be dict objects")
                for key in ("prompt", "response"):
                    value = row.get(key, "")
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError(f"mode 'kto' rows require non-empty '{key}'")
                if not isinstance(row.get("desirable"), bool):
                    raise ValueError("mode 'kto' rows require boolean 'desirable'")
        return self


def load_training_config_raw(path: str | Path) -> dict[str, Any]:
    """Load a training config file as a raw mapping without validation."""
    cfg_path = Path(path)
    text = cfg_path.read_text(encoding="utf-8")

    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        raw = yaml.safe_load(text)
    elif cfg_path.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        # Try YAML first for convenience (it handles JSON too).
        raw = yaml.safe_load(text)

    if not isinstance(raw, dict):
        raise ValueError(f"Training config must be a mapping/object: {cfg_path}")

    return raw


_PROMPT_LIST_MODES = {"grpo", "remax", "rest", "rlhf", "self_play"}


def materialize_dispatch_payload(data: object, mode: str | None) -> object:
    """Resolve dispatcher 'data' payload from on-disk files when given a path.

    Centralizes the file-backed payload contract so every consumer of the
    dispatcher (CLI, API, GUI) gets the same materialization rules before
    strict schema validation runs.
    """
    if not isinstance(data, str):
        return data

    path = Path(data)
    if not path.exists():
        return data

    suffix = path.suffix.lower()
    if suffix == ".txt":
        text = path.read_text(encoding="utf-8")
        if mode in _PROMPT_LIST_MODES:
            return [ln.strip() for ln in text.splitlines() if ln.strip()]
        return text

    if suffix == ".jsonl":
        rows: list[Any] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
        return rows

    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))

    return path.read_text(encoding="utf-8")


def load_training_config(path: str | Path) -> TrainingJobConfig:
    """Load and validate a training config from YAML or JSON file."""
    raw = load_training_config_raw(path)
    raw["data"] = materialize_dispatch_payload(raw.get("data"), raw.get("mode"))

    return TrainingJobConfig.model_validate(raw)
