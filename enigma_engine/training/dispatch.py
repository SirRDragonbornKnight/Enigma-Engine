"""Training dispatcher entry-point."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from enigma_engine.core.lora_utils import LoraConfig, LoraTrainer
from enigma_engine.core.rl_training import (
    GRPOConfig,
    GRPOTrainer,
    RLHFConfig,
    RLHFTrainer,
    ReMaxConfig,
    ReMaxTrainer,
    RewardModel,
    RewardTrainer,
    RewardTrainerConfig,
    SelfPlayConfig,
    SelfPlayTrainer,
)
from enigma_engine.training.training import Trainer, TrainingConfig

from .registry import get_mode_registry
from .schema import TrainingJobConfig, materialize_dispatch_payload


@dataclass
class DispatchContext:
    """Runtime objects required to execute training modes."""

    model: Any
    tokenizer: Any
    reward_fn: Callable[[str, str], float] | None = None
    reward_model: RewardModel | None = None
    trainer_engine: Any = None
    vision_encoder: Any = None
    audio_encoder: Any = None
    on_progress: Callable[[int, str], None] | None = None
    on_epoch_complete: Callable[[int, float], None] | None = None
    on_loss: Callable[[float], None] | None = None
    on_throughput: Callable[[int, float], None] | None = None
    on_trainer_ready: Callable[[Any], None] | None = None


def _core_training_config(job: TrainingJobConfig) -> TrainingConfig:
    return TrainingConfig(**job.training.model_dump())


def _apply_callbacks(trainer: Any, ctx: DispatchContext) -> None:
    if hasattr(trainer, "on_progress"):
        trainer.on_progress = ctx.on_progress
    if hasattr(trainer, "on_epoch_complete"):
        trainer.on_epoch_complete = ctx.on_epoch_complete
    if hasattr(trainer, "on_loss"):
        trainer.on_loss = ctx.on_loss
    if hasattr(trainer, "on_throughput") and ctx.on_throughput is not None:
        trainer.on_throughput = ctx.on_throughput
    if ctx.on_trainer_ready is not None:
        ctx.on_trainer_ready(trainer)


def build_dispatch_context(
    engine: Any | None = None,
    *,
    model: Any = None,
    tokenizer: Any = None,
    reward_fn: Callable[[str, str], float] | None = None,
    reward_model: RewardModel | None = None,
    vision_encoder: Any = None,
    audio_encoder: Any = None,
    on_progress: Callable[[int, str], None] | None = None,
    on_epoch_complete: Callable[[int, float], None] | None = None,
    on_loss: Callable[[float], None] | None = None,
    on_throughput: Callable[[int, float], None] | None = None,
    on_trainer_ready: Callable[[Any], None] | None = None,
) -> DispatchContext:
    """Build a DispatchContext from a loaded engine or raw model+tokenizer.

    `engine` is any object exposing `.model` and `.tokenizer` (e.g. EnigmaEngine).
    Pass `model` and `tokenizer` directly when no engine wrapper exists.
    """
    if engine is not None:
        if model is None:
            model = getattr(engine, "model", None)
        if tokenizer is None:
            tokenizer = getattr(engine, "tokenizer", None)

    if model is None or tokenizer is None:
        raise ValueError(
            "build_dispatch_context requires an engine with .model/.tokenizer "
            "or explicit model= and tokenizer= arguments"
        )

    return DispatchContext(
        model=model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        reward_model=reward_model,
        vision_encoder=vision_encoder,
        audio_encoder=audio_encoder,
        on_progress=on_progress,
        on_epoch_complete=on_epoch_complete,
        on_loss=on_loss,
        on_throughput=on_throughput,
        on_trainer_ready=on_trainer_ready,
    )


def run_training(config: TrainingJobConfig | dict[str, Any], ctx: DispatchContext) -> Any:
    """Dispatch a validated job config to the corresponding trainer."""
    if isinstance(config, TrainingJobConfig):
        job = config
    else:
        raw = dict(config)
        raw["data"] = materialize_dispatch_payload(raw.get("data"), raw.get("mode"))
        job = TrainingJobConfig.model_validate(raw)

    registry = get_mode_registry()
    spec = registry.get(job.mode)
    if spec is None:
        raise ValueError(f"Unknown training mode: {job.mode}")
    if spec.experimental and not job.allow_experimental:
        raise ValueError(f"Mode '{job.mode}' is experimental. Set allow_experimental=true to run it.")

    train_cfg = _core_training_config(job)

    if job.mode == "sft":
        trainer = Trainer(ctx.model, ctx.tokenizer, train_cfg)
        _apply_callbacks(trainer, ctx)
        return trainer.train(job.data, resume_from=job.resume_from)

    if job.mode == "dpo":
        trainer = Trainer(ctx.model, ctx.tokenizer, train_cfg)
        _apply_callbacks(trainer, ctx)
        return trainer.train_dpo(
            preference_data=job.data,
            beta=job.dpo.beta,
            reward_fn=ctx.reward_fn,
            dpo_online_interval=job.dpo.dpo_online_interval,
            dpo_online_samples=job.dpo.dpo_online_samples,
            loss_type=job.dpo.loss_type,
        )

    if job.mode == "simpo":
        trainer = Trainer(ctx.model, ctx.tokenizer, train_cfg)
        _apply_callbacks(trainer, ctx)
        return trainer.train_simpo(
            preference_data=job.data,
            beta=job.simpo.beta,
            gamma=job.simpo.gamma,
        )

    if job.mode == "kto":
        trainer = Trainer(ctx.model, ctx.tokenizer, train_cfg)
        _apply_callbacks(trainer, ctx)
        return trainer.train_kto(
            feedback_data=job.data,
            beta=job.kto.beta,
            desirable_weight=job.kto.desirable_weight,
            undesirable_weight=job.kto.undesirable_weight,
        )

    if job.mode == "orpo":
        trainer = Trainer(ctx.model, ctx.tokenizer, train_cfg)
        _apply_callbacks(trainer, ctx)
        return trainer.train_orpo(preference_data=job.data, beta=job.orpo.beta)

    if job.mode == "rest":
        if ctx.reward_fn is None:
            raise ValueError("rest mode requires DispatchContext.reward_fn")
        trainer = Trainer(ctx.model, ctx.tokenizer, train_cfg)
        _apply_callbacks(trainer, ctx)
        return trainer.train_rest(
            prompts=job.data,
            reward_fn=ctx.reward_fn,
            rounds=job.rest.rounds,
            n_samples=job.rest.n_samples,
            beta=job.rest.beta,
            max_new_tokens=job.rest.max_new_tokens,
        )

    if job.mode == "vision":
        if ctx.vision_encoder is None:
            raise ValueError("vision mode requires DispatchContext.vision_encoder")
        trainer = Trainer(ctx.model, ctx.tokenizer, train_cfg)
        _apply_callbacks(trainer, ctx)
        vision_data = job.data
        val_data = None
        if isinstance(vision_data, dict):
            val_data = vision_data.get("val")
            vision_data = vision_data.get("train", [])
        return trainer.train_vision(
            vision_encoder=ctx.vision_encoder,
            data=vision_data,
            val_data=val_data,
            unfreeze_text_layers=job.vision.unfreeze_text_layers,
        )

    if job.mode == "audio":
        if ctx.audio_encoder is None:
            raise ValueError("audio mode requires DispatchContext.audio_encoder")
        trainer = Trainer(ctx.model, ctx.tokenizer, train_cfg)
        _apply_callbacks(trainer, ctx)
        audio_data = job.data
        val_data = None
        if isinstance(audio_data, dict):
            val_data = audio_data.get("val")
            audio_data = audio_data.get("train", [])
        return trainer.train_audio(
            audio_encoder=ctx.audio_encoder,
            data=audio_data,
            val_data=val_data,
            unfreeze_text_layers=job.audio.unfreeze_text_layers,
        )

    if job.mode == "lora":
        lora_config = LoraConfig(
            rank=job.lora.rank,
            alpha=job.lora.alpha,
        )
        lora_trainer = LoraTrainer(
            model=ctx.model,
            tokenizer=ctx.tokenizer,
            lora_config=lora_config,
            output_dir=job.lora.output_dir,
            learning_rate=job.lora.learning_rate,
            batch_size=job.lora.batch_size,
            epochs=job.lora.epochs,
            gradient_accumulation_steps=job.lora.gradient_accumulation_steps,
            min_lr_ratio=job.lora.min_lr_ratio,
        )
        _apply_callbacks(lora_trainer, ctx)
        train_result = lora_trainer.train(job.data, max_length=job.lora.max_length)
        adapter_name = f"adapter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        adapter_target = Path(job.lora.output_dir) / adapter_name
        adapter_path = lora_trainer.save_adapter(adapter_target)
        return {
            "train_result": train_result,
            "adapter_path": str(adapter_path),
        }

    if job.mode == "reward_model":
        reward_model = ctx.reward_model or RewardModel(ctx.model)
        reward_trainer = RewardTrainer(
            reward_model=reward_model,
            tokenizer=ctx.tokenizer,
            config=RewardTrainerConfig(
                epochs=job.training.epochs,
                learning_rate=job.training.learning_rate,
                batch_size=max(1, job.training.batch_size),
                gradient_clip=job.training.gradient_clip,
                use_amp=job.training.use_amp,
            ),
        )
        _apply_callbacks(reward_trainer, ctx)
        return reward_trainer.train(job.data)

    if job.mode == "grpo":
        if ctx.reward_fn is None:
            raise ValueError("grpo mode requires DispatchContext.reward_fn")
        grpo_trainer = GRPOTrainer(
            model=ctx.model,
            tokenizer=ctx.tokenizer,
            reward_fn=ctx.reward_fn,
            config=GRPOConfig(
                epochs=job.training.epochs,
                learning_rate=job.training.learning_rate,
                max_new_tokens=job.grpo.max_new_tokens,
                group_size=job.grpo.group_size,
                temperature=job.grpo.temperature,
                use_amp=job.training.use_amp,
            ),
        )
        _apply_callbacks(grpo_trainer, ctx)
        return grpo_trainer.train(job.data)

    if job.mode == "remax":
        if ctx.reward_fn is None:
            raise ValueError("remax mode requires DispatchContext.reward_fn")
        remax_trainer = ReMaxTrainer(
            model=ctx.model,
            tokenizer=ctx.tokenizer,
            reward_fn=ctx.reward_fn,
            config=ReMaxConfig(
                epochs=job.training.epochs,
                learning_rate=job.training.learning_rate,
                max_new_tokens=job.remax.max_new_tokens,
                n_responses=job.remax.n_responses,
                temperature=job.remax.temperature,
                use_amp=job.training.use_amp,
            ),
        )
        _apply_callbacks(remax_trainer, ctx)
        return remax_trainer.train(job.data)

    if job.mode == "rlhf":
        if ctx.reward_model is None:
            raise ValueError("rlhf mode requires DispatchContext.reward_model")
        rlhf_trainer = RLHFTrainer(
            model=ctx.model,
            tokenizer=ctx.tokenizer,
            reward_model=ctx.reward_model,
            config=RLHFConfig(
                epochs=job.training.epochs,
                learning_rate=job.training.learning_rate,
                use_amp=job.training.use_amp,
            ),
        )
        _apply_callbacks(rlhf_trainer, ctx)
        return rlhf_trainer.train(job.data)

    if job.mode == "self_play":
        if ctx.trainer_engine is None:
            raise ValueError("self_play mode requires DispatchContext.trainer_engine")
        sp_trainer = SelfPlayTrainer(
            student=ctx.model,
            tokenizer=ctx.tokenizer,
            trainer_engine=ctx.trainer_engine,
            config=SelfPlayConfig(
                epochs=job.training.epochs,
                learning_rate=job.training.learning_rate,
                use_amp=job.training.use_amp,
            ),
        )
        _apply_callbacks(sp_trainer, ctx)
        return sp_trainer.train(job.data)

    if job.mode == "adaptive":
        raise NotImplementedError(
            "adaptive mode is a GUI/meta scheduler path and is not yet supported by the dispatcher"
        )

    raise ValueError(f"Unhandled training mode: {job.mode}")
