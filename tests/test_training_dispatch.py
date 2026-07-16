"""Tests for training schema/registry/dispatcher seam."""

from __future__ import annotations

from pathlib import Path

import pytest

from enigma_engine.training.dispatch import DispatchContext, run_training
from enigma_engine.training.registry import get_mode_registry
from enigma_engine.training.schema import TrainingJobConfig, load_training_config


def test_load_training_config_yaml(tmp_path: Path) -> None:
    cfg = tmp_path / "train.yaml"
    cfg.write_text(
        "\n".join(
            [
                "mode: sft",
                "data: data/training.txt",
                "training:",
                "  epochs: 2",
                "  learning_rate: 0.0002",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_training_config(cfg)

    assert loaded.mode == "sft"
    assert loaded.training.epochs == 2
    assert loaded.training.learning_rate == pytest.approx(0.0002)


def test_registry_core_and_experimental_flags() -> None:
    registry = get_mode_registry()

    assert registry["sft"].experimental is False
    assert registry["grpo"].experimental is False
    assert registry["simpo"].experimental is True
    assert registry["adaptive"].experimental is True


def test_grpo_requires_prompt_list() -> None:
    with pytest.raises(ValueError, match="expects 'data' to be a non-empty list"):
        TrainingJobConfig.model_validate(
            {
                "mode": "grpo",
                "data": "not-a-list",
            }
        )


def test_sft_rejects_zero_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        TrainingJobConfig.model_validate(
            {
                "mode": "sft",
                "data": "text",
                "training": {"batch_size": 0},
            }
        )


def test_dpo_requires_list_payload() -> None:
    with pytest.raises(ValueError, match="expects 'data' to be a non-empty list"):
        TrainingJobConfig.model_validate(
            {
                "mode": "dpo",
                "data": "oops",
            }
        )


def test_dpo_requires_prompt_chosen_rejected_keys() -> None:
    with pytest.raises(ValueError, match="prompt|chosen|rejected"):
        TrainingJobConfig.model_validate(
            {
                "mode": "dpo",
                "data": [{"prompt": "p", "chosen": "c"}],
            }
        )


def test_sft_rejects_blank_string_payload() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        TrainingJobConfig.model_validate(
            {
                "mode": "sft",
                "data": "   ",
            }
        )


def test_grpo_requires_non_empty_prompt_strings() -> None:
    with pytest.raises(ValueError, match="non-empty string prompts"):
        TrainingJobConfig.model_validate(
            {
                "mode": "grpo",
                "data": ["ok prompt", "", 42],
            }
        )


def test_reward_model_requires_preference_rows() -> None:
    with pytest.raises(ValueError, match="reward_model"):
        TrainingJobConfig.model_validate(
            {
                "mode": "reward_model",
                "allow_experimental": True,
                "data": "not-a-list",
            }
        )


def test_run_training_sft_routes_to_trainer(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class FakeTrainer:
        def __init__(self, model, tokenizer, config):
            calls["init_model"] = model
            calls["init_tokenizer"] = tokenizer
            calls["config_epochs"] = config.epochs
            self.on_progress = None
            self.on_epoch_complete = None
            self.on_loss = None

        def train(self, data, resume_from=None):
            calls["data"] = data
            calls["resume_from"] = resume_from
            return {"ok": True}

    monkeypatch.setattr("enigma_engine.training.dispatch.Trainer", FakeTrainer)

    ctx = DispatchContext(model=object(), tokenizer=object())
    result = run_training(
        {
            "mode": "sft",
            "data": "hello world",
            "resume_from": "models/checkpoints/last.pt",
            "training": {"epochs": 3},
        },
        ctx,
    )

    assert result == {"ok": True}
    assert calls["config_epochs"] == 3
    assert calls["resume_from"] == "models/checkpoints/last.pt"


def test_run_training_dpo_forwards_dpo_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class FakeTrainer:
        def __init__(self, model, tokenizer, config):
            self.on_progress = None
            self.on_epoch_complete = None
            self.on_loss = None

        def train_dpo(
            self,
            preference_data,
            beta,
            reward_fn,
            dpo_online_interval,
            dpo_online_samples,
            loss_type,
        ):
            calls["preference_data"] = preference_data
            calls["beta"] = beta
            calls["interval"] = dpo_online_interval
            calls["samples"] = dpo_online_samples
            calls["loss_type"] = loss_type
            calls["reward_fn"] = reward_fn
            return {"dpo": True}

    def reward_fn(prompt, response):
        return 1.0

    monkeypatch.setattr("enigma_engine.training.dispatch.Trainer", FakeTrainer)

    result = run_training(
        {
            "mode": "dpo",
            "data": [{"prompt": "p", "chosen": "c", "rejected": "r"}],
            "dpo": {
                "beta": 0.3,
                "loss_type": "apo_zero",
                "dpo_online_interval": 8,
                "dpo_online_samples": 6,
            },
        },
        DispatchContext(model=object(), tokenizer=object(), reward_fn=reward_fn),
    )

    assert result == {"dpo": True}
    assert calls["beta"] == pytest.approx(0.3)
    assert calls["interval"] == 8
    assert calls["samples"] == 6
    assert calls["loss_type"] == "apo_zero"
    assert calls["reward_fn"] is reward_fn


def test_run_training_vision_requires_encoder() -> None:
    with pytest.raises(ValueError, match="vision mode requires"):
        run_training(
            {
                "mode": "vision",
                "data": [{"image": "a.png", "text": "x"}],
            },
            DispatchContext(model=object(), tokenizer=object()),
        )


def test_run_training_vision_accepts_train_val_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeTrainer:
        def __init__(self, model, tokenizer, config):
            self.on_progress = None
            self.on_epoch_complete = None
            self.on_loss = None

        def train_vision(
            self,
            vision_encoder,
            data,
            unfreeze_text_layers,
            val_data=None,
            resume_from=None,
        ):
            captured["data"] = data
            captured["val_data"] = val_data
            captured["unfreeze"] = unfreeze_text_layers
            captured["resume_from"] = resume_from
            return {"ok": True}

    monkeypatch.setattr("enigma_engine.training.dispatch.Trainer", FakeTrainer)

    result = run_training(
        {
            "mode": "vision",
            "data": {
                "train": [{"image": "a.png", "text": "x"}],
                "val": [{"image": "b.png", "text": "y"}],
            },
            "vision": {"unfreeze_text_layers": 2},
        },
        DispatchContext(
            model=object(),
            tokenizer=object(),
            vision_encoder=object(),
        ),
    )

    assert result == {"ok": True}
    assert captured["data"] == [{"image": "a.png", "text": "x"}]
    assert captured["val_data"] == [{"image": "b.png", "text": "y"}]
    assert captured["unfreeze"] == 2
    assert captured["resume_from"] is None


def test_training_job_config_vision_rejects_non_list_val() -> None:
    with pytest.raises(ValueError, match="data.val"):
        TrainingJobConfig.model_validate(
            {
                "mode": "vision",
                "data": {
                    "train": [{"image": "a.png", "text": "x"}],
                    "val": "bad",
                },
            }
        )


def test_run_training_experimental_requires_opt_in() -> None:
    with pytest.raises(ValueError, match="experimental"):
        run_training(
            {
                "mode": "simpo",
                "data": [{"prompt": "p", "chosen": "c", "rejected": "r"}],
            },
            DispatchContext(model=object(), tokenizer=object()),
        )


def test_run_training_grpo_requires_reward_fn() -> None:
    with pytest.raises(ValueError, match="requires DispatchContext.reward_fn"):
        run_training(
            {
                "mode": "grpo",
                "allow_experimental": False,
                "data": ["prompt"],
            },
            DispatchContext(model=object(), tokenizer=object()),
        )


def test_run_training_remax_requires_reward_fn() -> None:
    with pytest.raises(ValueError, match="requires DispatchContext.reward_fn"):
        run_training(
            {
                "mode": "remax",
                "allow_experimental": True,
                "data": ["prompt"],
            },
            DispatchContext(model=object(), tokenizer=object()),
        )


def test_run_training_remax_routes_to_trainer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeReMaxTrainer:
        def __init__(self, model, tokenizer, reward_fn, config):
            self.on_progress = None
            self.on_epoch_complete = None
            self.on_loss = None
            captured["reward_fn"] = reward_fn
            captured["n_responses"] = config.n_responses

        def train(self, prompts):
            captured["prompts"] = prompts
            return {"ok": True}

    monkeypatch.setattr("enigma_engine.training.dispatch.ReMaxTrainer", FakeReMaxTrainer)

    result = run_training(
        {
            "mode": "remax",
            "allow_experimental": True,
            "data": ["prompt-a", "prompt-b"],
            "remax": {
                "n_responses": 5,
            },
        },
        DispatchContext(
            model=object(),
            tokenizer=object(),
            reward_fn=lambda p, r: 0.0,
        ),
    )

    assert result == {"ok": True}
    assert captured["prompts"] == ["prompt-a", "prompt-b"]
    assert captured["n_responses"] == 5


def test_run_training_materializes_jsonl_data_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Dispatcher must accept raw dict with data: <path> and load the file."""
    captured: dict[str, object] = {}

    data_path = tmp_path / "prefs.jsonl"
    data_path.write_text(
        '{"prompt": "p", "chosen": "c", "rejected": "r"}\n',
        encoding="utf-8",
    )

    class FakeTrainer:
        def __init__(self, model, tokenizer, config):
            self.on_progress = None
            self.on_epoch_complete = None
            self.on_loss = None

        def train_dpo(self, **kwargs):
            captured["preference_data"] = kwargs.get("preference_data")
            return {"ok": True}

    monkeypatch.setattr("enigma_engine.training.dispatch.Trainer", FakeTrainer)

    result = run_training(
        {
            "mode": "dpo",
            "data": str(data_path),
        },
        DispatchContext(model=object(), tokenizer=object()),
    )

    assert result == {"ok": True}
    assert captured["preference_data"] == [{"prompt": "p", "chosen": "c", "rejected": "r"}]


def test_run_training_materializes_prompt_text_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Dispatcher must split a .txt prompt list when mode expects list[str]."""
    captured: dict[str, object] = {}

    data_path = tmp_path / "prompts.txt"
    data_path.write_text("first prompt\n\nsecond prompt\n", encoding="utf-8")

    class FakeGRPOTrainer:
        def __init__(self, model, tokenizer, reward_fn, config):
            self.on_progress = None
            self.on_epoch_complete = None
            self.on_loss = None

        def train(self, prompts):
            captured["prompts"] = prompts
            return {"ok": True}

    monkeypatch.setattr("enigma_engine.training.dispatch.GRPOTrainer", FakeGRPOTrainer)

    result = run_training(
        {
            "mode": "grpo",
            "data": str(data_path),
        },
        DispatchContext(
            model=object(),
            tokenizer=object(),
            reward_fn=lambda p, r: 0.0,
        ),
    )

    assert result == {"ok": True}
    assert captured["prompts"] == ["first prompt", "second prompt"]


def test_build_dispatch_context_from_engine_object() -> None:
    """build_dispatch_context should pull model/tokenizer from an engine."""
    from enigma_engine.training import build_dispatch_context

    class FakeEngine:
        model = object()
        tokenizer = object()

    engine = FakeEngine()
    ctx = build_dispatch_context(engine=engine)

    assert ctx.model is engine.model
    assert ctx.tokenizer is engine.tokenizer


def test_build_dispatch_context_accepts_explicit_model_and_tokenizer() -> None:
    from enigma_engine.training import build_dispatch_context

    model = object()
    tokenizer = object()
    cb = lambda pct, msg: None  # noqa: E731

    ctx = build_dispatch_context(model=model, tokenizer=tokenizer, on_progress=cb)

    assert ctx.model is model
    assert ctx.tokenizer is tokenizer
    assert ctx.on_progress is cb


def test_build_dispatch_context_requires_model_and_tokenizer() -> None:
    import pytest as _pytest

    from enigma_engine.training import build_dispatch_context

    with _pytest.raises(ValueError, match="requires"):
        build_dispatch_context()


def test_run_training_lora_saves_adapter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    class FakeLoraTrainer:
        def __init__(
            self,
            model,
            tokenizer,
            lora_config,
            output_dir,
            learning_rate,
            batch_size,
            epochs,
            gradient_accumulation_steps,
            min_lr_ratio,
        ):
            calls["output_dir"] = output_dir
            calls["rank"] = lora_config.rank
            calls["alpha"] = lora_config.alpha
            self.on_progress = None
            self.on_epoch_complete = None
            self.on_loss = None

        def train(self, data, max_length=512):
            calls["data"] = data
            calls["max_length"] = max_length
            return {"ok": True}

        def save_adapter(self, path=None):
            calls["save_called"] = True
            calls["save_path"] = path
            if path is None:
                return tmp_path / "adapter_test"
            return Path(path)

    monkeypatch.setattr("enigma_engine.training.dispatch.LoraTrainer", FakeLoraTrainer)

    result = run_training(
        {
            "mode": "lora",
            "data": [{"prompt": "p", "completion": "c"}],
            "lora": {
                "rank": 12,
                "alpha": 24,
                "output_dir": str(tmp_path),
                "max_length": 256,
            },
        },
        DispatchContext(model=object(), tokenizer=object()),
    )

    assert calls.get("save_called") is True
    assert calls.get("save_path") is not None
    assert calls["rank"] == 12
    assert calls["alpha"] == 24
    assert result["train_result"] == {"ok": True}
    assert "adapter_" in Path(result["adapter_path"]).name


def test_on_trainer_ready_called_with_trainer_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """on_trainer_ready must be called with the Trainer instance after callbacks
    are wired but before .train() is invoked (ARCH-1.5c contract)."""
    from enigma_engine.training.dispatch import build_dispatch_context

    ready_calls: list = []
    train_called: list = []

    class FakeTrainer:
        def __init__(self, model, tokenizer, config):
            self.on_progress = None
            self.on_epoch_complete = None
            self.on_loss = None
            self.on_throughput = None

        def train(self, data, resume_from=None):
            # on_trainer_ready must have been called before we get here
            train_called.append(True)
            return "done"

    monkeypatch.setattr("enigma_engine.training.dispatch.Trainer", FakeTrainer)

    def on_ready(t):
        ready_calls.append(t)

    ctx = build_dispatch_context(
        model=object(),
        tokenizer=object(),
        on_trainer_ready=on_ready,
    )
    result = run_training({"mode": "sft", "data": "hello"}, ctx)

    assert result == "done"
    assert len(ready_calls) == 1, "on_trainer_ready must be called exactly once"
    assert isinstance(ready_calls[0], FakeTrainer)
    assert train_called, "train() must be called after on_trainer_ready"


def test_on_throughput_forwarded_to_trainer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """on_throughput from DispatchContext must be set on the trainer."""
    from enigma_engine.training.dispatch import build_dispatch_context

    wired: list = []

    class FakeTrainer:
        def __init__(self, model, tokenizer, config):
            self.on_progress = None
            self.on_epoch_complete = None
            self.on_loss = None
            self.on_throughput = None

        def train(self, data, resume_from=None):
            wired.append(self.on_throughput)
            return "done"

    monkeypatch.setattr("enigma_engine.training.dispatch.Trainer", FakeTrainer)

    def throughput_cb(tokens: int, step_time: float) -> None:
        pass

    ctx = build_dispatch_context(
        model=object(),
        tokenizer=object(),
        on_throughput=throughput_cb,
    )
    run_training({"mode": "sft", "data": "hello"}, ctx)

    assert len(wired) == 1
    assert wired[0] is throughput_cb


def test_training_overrides_accepts_sft_knobs() -> None:
    """TrainingOverrides must accept all SFT-specific fields added in ARCH-1.5c."""
    job = TrainingJobConfig.model_validate(
        {
            "mode": "sft",
            "data": "some text",
            "training": {
                "epochs": 2,
                "use_gradient_checkpointing": False,
                "use_sequence_packing": True,
                "ce_chunk_size": 512,
                "use_compile": True,
                "rolling_best_k": 3,
                "general_mix_ratio": 0.1,
                "general_data": "data/general.txt",
                "val_split": 0.05,
                "save_every": 2,
                "run_evaluation": True,
            },
        }
    )
    t = job.training
    assert t.use_gradient_checkpointing is False
    assert t.use_sequence_packing is True
    assert t.ce_chunk_size == 512
    assert t.use_compile is True
    assert t.rolling_best_k == 3
    assert t.general_mix_ratio == pytest.approx(0.1)
    assert t.general_data == "data/general.txt"
    assert t.val_split == pytest.approx(0.05)
    assert t.save_every == 2
    assert t.run_evaluation is True
