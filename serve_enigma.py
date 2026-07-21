#!/usr/bin/env python
"""Serve the REAL Enigma -- the from-scratch transformer -- as an OpenAI-compatible
/v1 endpoint, so Odysseus (or any OpenAI client) can talk to her.

  python serve_enigma.py                       # models/enigma_dpo/model.pth (the adopted model)
  python serve_enigma.py --model models/enigma_pretrain_base_v2/latest.pth
  # then, in Odysseus chat:  /setup local http://127.0.0.1:8000/v1

She is a BASE model (mid-pretraining): no chat template and no tool tokens yet --
those arrive with the instruct pass (special-token IDs 4718-4735 are reserved in
the padded embedding). /v1/chat/completions therefore bridges by rendering the
conversation as a plain-text transcript she continues; /v1/completions is her
native shape.

Replaces the rejected Qwen-wrapper server (the "Muppet"; its <tool_call>
parsing lives in git history and returns with the instruct pass).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import queue
import re
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from enigma_engine.core.chat_format import (
    CHAT_FORMAT_NAME,
    ROLES,
    attach_chat_tokens,
    chat_token_ids,
    parse_assistant_ids,
    render_chat,
    render_tools_system,
    think_token_ids,
)
from enigma_engine.core.asr import ASRError, Ears
from enigma_engine.core.calculator import CalcError, evaluate, format_result
from enigma_engine.core.eyes import Eyes, EyesError, flatten_image_content
from enigma_engine.core.imagegen import ImageGenError, Painter
from enigma_engine.core.tts import Speaker, TTSError
from enigma_engine.core.model import Enigma
from enigma_engine.core.model_presets import ForgeConfig
from enigma_engine.core.tokenizer import get_tokenizer, vocab_file_for_size

try:  # Windows consoles default to cp1252 and crash printing unicode.
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
MODEL_ID = "enigma"

_p = argparse.ArgumentParser()
_p.add_argument(
    "--model",
    # The ADOPTED instruct/DPO model -- same posture as every launcher. The old
    # default was the raw pretrain checkpoint, so a bare `enigma` console
    # script served the wrong brain (audit 2026-07-17).
    default=str(ROOT / "models" / "enigma_dpo" / "model.pth"),
    help="Enigma checkpoint (.pth with model_state_dict + config); default = the adopted instruct/DPO model",
)
_p.add_argument("--host", default="127.0.0.1")
_p.add_argument("--port", type=int, default=8000)
_p.add_argument(
    "--max-context",
    type=int,
    default=1024,
    help="prompt+generation token budget; she trains at block 1024 -- longer is mechanically possible but untested",
)
_p.add_argument(
    "--memory-dir",
    default=None,
    help="enable the local memory store (JSONL + BM25); relevant memories are injected into her system context",
)
_p.add_argument(
    "--voice",
    action="store_true",
    help="enable the voice organ: the speak built-in tool + /v1/audio/speech (local pyttsx3/SAPI)",
)
_p.add_argument(
    "--voice-name",
    default=None,
    help="pick the TTS voice by name or id substring, e.g. 'zira' (default: system default voice)",
)
_p.add_argument(
    "--ears",
    action="store_true",
    help="enable the ears organ: /v1/audio/transcriptions (local faster-whisper)",
)
_p.add_argument(
    "--eyes",
    action="store_true",
    help="enable the eyes organ: image messages are captioned into her context + /v1/images/describe "
    "(HER OWN aligned vision encoder + projection + the served model -- no external captioner)",
)
_p.add_argument(
    "--eyes-model",
    default=None,
    help="align checkpoint carrying vision_encoder_state_dict + vision_projection weights "
    "(default: models/enigma_vision_align/enigma_vision_align_vision_best.pt in the repo)",
)
_p.add_argument(
    "--eyes-preset",
    default="medium",
    help="VisionEncoder preset the align checkpoint was trained with (see vision_encoder.VISION_PRESETS)",
)
_p.add_argument(
    "--image-gen",
    action="store_true",
    help="enable the imagination organ: the imagine built-in tool + /v1/images/generations (local Stable Diffusion)",
)
_p.add_argument(
    "--allow-downloads",
    action="store_true",
    help="permit a one-time organ weight download from HuggingFace; WITHOUT this flag the server is fully offline",
)
_p.add_argument(
    "--fp32",
    action="store_true",
    help="generate in full fp32 instead of the default bf16 autocast (slower; numerics escape hatch)",
)
# ---------------------------------------------------------------------------
# Runtime state -- populated by boot(). This module used to do ALL of its
# startup (argv parse, checkpoint load, organ construction) at import time,
# which made it a zero-coverage island: a bare `import serve_enigma` needed a
# real model file (test-suite audit 2026-07-17). boot() owns startup now and
# main() calls it; tests import the module cheaply, then either boot() a tiny
# checkpoint or set these globals directly.
# ---------------------------------------------------------------------------
ARGS: argparse.Namespace | None = None
CONFIG = None
model = None
tokenizer = None
DEVICE = "cpu"
_BF16_GEN = False
STEP = None
META: dict = {}
INSTRUCT = False
MEMORY = None
SPEAKER = None
MUTED = False
EARS = None
EYES = None
PAINTER = None
EOS_ID = 2
BOS_ID = 1
# True only when boot() ran to COMPLETION. `model is None` was the readiness
# signal at first, but boot() assigns model early and can still die at the
# max-context guard / tokenizer / organs -- leaving exactly the deceptive
# half-up app the middleware exists to kill (round-2 re-audit 2026-07-18).
_BOOTED = False

# Always keep this many ids of the context free for the reply. Prompt and
# generation share the fixed max_context window; without a reserve a large
# client max_tokens would shrink the prompt budget toward zero and the model
# would answer from a near-empty context (confident garbage). We keep the
# prompt intact (up to max_context - MIN_GEN_TOKENS) and let generation take
# whatever room is left -- never the other way around.
MIN_GEN_TOKENS = 64

def _stop_ids() -> tuple[int, int]:
    """(EOS, <|im_end|>) for the CURRENTLY attached tokenizer.

    Derived per call rather than cached in a module global: the chat ids
    belong to the tokenizer instance (they move with vocab size -- see
    chat_format HIGH-2), and boot() can re-run with a different vocab. A
    rebound global would also leak across boots, since serve's test
    snapshot/restore list cannot see a name that only boot() writes.
    """
    return EOS_ID, chat_token_ids(tokenizer)["<|im_end|>"]


# One model, one KV-cache -- generation must be serialized across requests.
# Defined BEFORE the organs: the eyes borrow the served model for caption
# generation and share this lock.
_GEN_LOCK = threading.Lock()

# Runtime mute (POST /v1/audio/mute -- the chat page's Mute button and the
# tray icon): silences the server-side speak TOOL, and /v1/audio/speech
# answers 204 (no audio) so muting from anywhere silences every open window.
# The server is the single source of truth; the page polls and adopts it.
# The truth survives restarts: best-effort persisted to a tiny state file
# (a crash-relaunch must not silently unmute a muted gaming session).
# Anchored to the repo (this file's directory), NOT the CWD -- the enigma /
# enigma-ai console scripts can be launched from anywhere and must still see
# the same state file (2026-07-17 audit).
_MUTE_STATE = Path(__file__).resolve().parent / "data" / "mute_state.json"

# Where the imagine tool and /v1/images/generations drop their PNGs: the
# engine's data home, not the repo checkout.
IMAGES_DIR = Path.home() / ".enigma_engine" / "images"


def _load_eyes(ckpt_path: Path, preset: str):
    """Load an align checkpoint: her aligned VisionEncoder, the trained
    vision_projection weights, and the encoder dim for the model's vision
    port. Raises EyesError for a missing file / non-align checkpoint / absent
    projection, and for a stored encoder config that will not rebuild (any
    failure while turning it into an encoder). An unknown preset raises KeyError
    and a mismatched encoder state dict RuntimeError -- boot() catches all of
    these and degrades to text-only with one honest WARN (extracted for
    testability, 2026-07-17)."""
    from enigma_engine.core.vision_encoder import VISION_PRESETS, VisionEncoder

    if not ckpt_path.exists():
        raise EyesError(
            f"align checkpoint not found: {ckpt_path} (run distill_vision_encoder.py then align_vision.py)"
        )
    # The file is untrusted: a truncated, empty or wrong-format checkpoint
    # raises EOFError/UnpicklingError, neither of which boot's degrade catch
    # covers, so an unreadable eye would take text serving down with it.
    try:
        eck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise EyesError(
            f"{ckpt_path} could not be read as a checkpoint "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    if not isinstance(eck, dict) or "vision_encoder_state_dict" not in eck:
        raise EyesError(f"{ckpt_path} carries no vision_encoder_state_dict (not an align checkpoint)")
    # Prefer the encoder config STORED IN the checkpoint (align runs
    # persist it since 2026-07-20) over the hand-passed preset name -- a
    # preset/checkpoint mismatch then cannot exist. Older checkpoints
    # without the key keep the preset path.
    stored_cfg = eck.get("vision_encoder_config")
    if stored_cfg:
        from enigma_engine.core.vision_encoder import VisionEncoderConfig

        # The stored config is UNTRUSTED input: only patch_size is checked by
        # the dataclass, while image_size/patch_size divisibility, head and
        # layer counts and dropout range are not validated until the encoder is
        # built. Both steps therefore sit inside the guard, and ANY failure
        # becomes EyesError so boot degrades to text-only rather than dying on
        # a raw ValueError/TypeError/ZeroDivisionError.
        try:
            vcfg = VisionEncoderConfig(**stored_cfg)
            venc = VisionEncoder(vcfg)
        except Exception as exc:
            raise EyesError(
                f"{ckpt_path} carries a vision_encoder_config that will not "
                f"rebuild ({type(exc).__name__}: {exc}); if the config itself is "
                f"stale, re-run the align to rewrite it or drop the key to fall "
                f"back to --eyes-preset"
            ) from exc
    else:
        # Any falsy stored config (absent, or an empty dict from a writer that
        # had nothing to record) takes the preset path: building from {} would
        # produce an all-defaults encoder while ignoring --eyes-preset.
        vcfg = VISION_PRESETS[preset]
        venc = VisionEncoder(vcfg)
    venc.load_state_dict(eck["vision_encoder_state_dict"], strict=True)
    proj_sd = {
        k[len("vision_projection."):]: v
        for k, v in eck["model_state_dict"].items()
        if k.startswith("vision_projection.")
    }
    if not proj_sd:
        raise EyesError(f"{ckpt_path} carries no vision_projection weights")
    return venc, proj_sd, vcfg.dim


# Env keys boot() itself wrote: key -> (value displaced by our write, or None
# if the key was absent; the value we wrote). setdefault must respect an
# OPERATOR's shell export but not a leftover from a previous boot() in this
# same process -- an --allow-downloads boot writes "0", and a later flagless
# boot's setdefault would silently keep the network open. Tracking the
# DISPLACED value lets a re-boot RESTORE the operator's export instead of
# deleting it, and a write that matches the current value claims nothing, so
# an export that already agrees with us stays the operator's (round-2
# re-audit 2026-07-18: the delete-only scheme destroyed a genuine
# HF_HUB_OFFLINE=0 export across an allow-downloads -> flagless boot pair).
_BOOT_ENV_WRITES: dict[str, tuple[str | None, str]] = {}


def boot(argv: list[str] | None = None) -> None:
    """Parse args, load the checkpoint, bring the organs up. main() calls
    this; tests call it with an explicit argv (or skip it and set globals
    directly). argv=None reads sys.argv -- byte-identical behavior to the
    old import-time startup."""
    global ARGS, CONFIG, model, tokenizer, DEVICE, _BF16_GEN, STEP, META
    global INSTRUCT, MEMORY, SPEAKER, MUTED, EARS, EYES, PAINTER, EOS_ID, BOS_ID
    global _BOOTED

    _BOOTED = False  # a re-boot is unready until it completes

    # parse_known_args is deliberate (the server must start under runners
    # that carry their own argv) -- but a typo'd flag must not silently
    # disable an organ, so unknowns are named out loud.
    ARGS, unknown = _p.parse_known_args(argv)
    if unknown:
        print(f"  WARN: ignoring unrecognized args: {' '.join(unknown)}", flush=True)

    # PRIVACY: she is local, fully. Her own weights never touch the network; the
    # organ libraries (transformers/diffusers/faster-whisper) would by default
    # phone HuggingFace at LOAD time for update checks and telemetry even when
    # the weights are already cached on disk. Offline is therefore the DEFAULT:
    # organs load from cache only. --allow-downloads exists solely for the
    # one-time first fetch of an organ's weights, on purpose, out loud.
    for _k, (_prior, _ours) in list(_BOOT_ENV_WRITES.items()):
        if os.environ.get(_k) == _ours:  # untouched since we wrote it
            if _prior is None:
                os.environ.pop(_k, None)
            else:
                os.environ[_k] = _prior  # give the operator's export back
        # rewritten externally since -> theirs now, our claim is void
    _BOOT_ENV_WRITES.clear()

    def _own_setdefault(key: str, value: str) -> None:
        if key not in os.environ:
            _BOOT_ENV_WRITES[key] = (None, value)
            os.environ[key] = value

    def _own_set(key: str, value: str) -> None:
        prior = os.environ.get(key)
        if prior == value:
            return  # an export already agreeing with us stays the operator's
        _BOOT_ENV_WRITES[key] = (prior, value)
        os.environ[key] = value

    _own_setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    _own_setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    if not ARGS.allow_downloads:
        _own_setdefault("HF_HUB_OFFLINE", "1")
        _own_setdefault("TRANSFORMERS_OFFLINE", "1")
    else:
        # The flag must WIN: a shell exporting HF_HUB_OFFLINE=1 would otherwise
        # silently block the one fetch the operator just asked for out loud.
        _own_set("HF_HUB_OFFLINE", "0")
        _own_set("TRANSFORMERS_OFFLINE", "0")

    print(f"Loading Enigma from {ARGS.model} ...", flush=True)
    if not Path(ARGS.model).exists():
        raise SystemExit(
            f"checkpoint not found: {ARGS.model}\n"
            "Pass --model <path to an Enigma .pth checkpoint> (the default only "
            "exists inside a repo checkout with trained models)"
        )
    _ck = torch.load(ARGS.model, map_location="cpu", weights_only=False)  # our own checkpoint
    if not (isinstance(_ck, dict) and "model_state_dict" in _ck and "config" in _ck):
        raise SystemExit(f"{ARGS.model} is not an Enigma checkpoint (need model_state_dict + config)")
    CONFIG = ForgeConfig.from_dict(_ck["config"])

    # Her own eyes (Phase 4.5 step 5): --eyes opens the model's vision port and
    # grafts the encoder + projection trained by align_vision.py onto the SERVED
    # text weights. The align pass kept every text weight frozen on purpose, so
    # the projection targets exactly this checkpoint's embedding space and text
    # behavior is byte-unchanged. Failure here degrades to text-only (WARN), the
    # organ pattern -- text serving never dies for a missing eye.
    _VISION_ENCODER = None
    _VISION_PROJ_SD = None
    if ARGS.eyes:
        _eyes_ckpt = Path(ARGS.eyes_model) if ARGS.eyes_model else (
            ROOT / "models" / "enigma_vision_align" / "enigma_vision_align_vision_best.pt"
        )
        try:
            _VISION_ENCODER, _VISION_PROJ_SD, _vdim = _load_eyes(_eyes_ckpt, ARGS.eyes_preset)
            CONFIG.vision_hidden_size = _vdim
        except (EyesError, KeyError, RuntimeError, OSError) as exc:
            print(f"  WARN: eyes disabled -- {exc}", flush=True)
            _VISION_ENCODER = None
            _VISION_PROJ_SD = None

    model = Enigma(CONFIG)
    if _VISION_ENCODER is not None:
        # Text weights come from the SERVED checkpoint; only the projection is new.
        _missing, _unexpected = model.load_state_dict(_ck["model_state_dict"], strict=False)
        _bad = [k for k in _missing if "vision_projection" not in k]
        if _bad or _unexpected:
            raise SystemExit(f"checkpoint mismatch with vision port open: missing={_bad[:5]} unexpected={list(_unexpected)[:5]}")
        model.vision_projection.load_state_dict(_VISION_PROJ_SD, strict=True)
    else:
        model.load_state_dict(_ck["model_state_dict"], strict=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(DEVICE).eval()
    # Serve was the only fp32 stage in the whole stack (ultrareview #36): every
    # training pass runs bf16 autocast with TF32 matmuls, and batch-1 decode is
    # weight-read bound, so fp32 roughly halves the tokens/s ceiling for nothing.
    # Generation runs under the same numerics training validated; --fp32 is the
    # escape hatch if a regression ever points here.
    if DEVICE == "cuda" and not ARGS.fp32:
        # TF32 rides the same flag: --fp32 must reproduce the true fp32 baseline
        # (TF32 truncates matmul mantissas, so leaving it on would defeat the
        # escape hatch; re-audit 2026-07-17).
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # device_count guard: is_available() can be True with zero visible devices
    # (CUDA_VISIBLE_DEVICES=""), where is_bf16_supported() raises instead of
    # returning False (measured on torch 2.10).
    _BF16_GEN = (
        DEVICE == "cuda" and not ARGS.fp32 and torch.cuda.device_count() > 0 and torch.cuda.is_bf16_supported()
    )
    STEP = _ck.get("step")
    META = _ck.get("meta") or {}  # finetune_enigma stamps chat_format here
    del _ck

    # Prompt truncation budgets against ARGS.max_context, and the model's KV
    # cache holds exactly config.max_seq_len positions: a single oversize
    # prefill is refused outright, and incremental overflow slides the window.
    # Clamp so neither can be reached from an oversize --max-context.
    # Attention.MAX_CACHE_SEQ_LEN must NOT bound this -- it is only the cache's
    # own fallback for configs lacking max_seq_len, which ForgeConfig always has.
    _cache_cap = CONFIG.max_seq_len
    if ARGS.max_context > _cache_cap:
        print(
            f"  WARN: --max-context {ARGS.max_context} exceeds the model's KV cache "
            f"capacity {_cache_cap}; clamping to {_cache_cap}",
            flush=True,
        )
        ARGS.max_context = _cache_cap
    if ARGS.max_context <= MIN_GEN_TOKENS:
        raise SystemExit(
            f"max_context {ARGS.max_context} leaves no prompt budget after the "
            f"{MIN_GEN_TOKENS}-token generation reserve; this model context is too small to serve"
        )

    # The vocabulary belongs to the WEIGHTS: pick the file matching this
    # checkpoint's vocab_size rather than whatever the repo directory defaults
    # to, so a v1 and a v2 model can each be served from the same checkout.
    try:
        _vocab_file = vocab_file_for_size(CONFIG.vocab_size)
        tokenizer = get_tokenizer("bpe", vocab_path=_vocab_file)
        print(f"  tokenizer: {_vocab_file.name} (model vocab {CONFIG.vocab_size})", flush=True)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  WARN: no vocab matches model vocab {CONFIG.vocab_size} ({exc}); using the default", flush=True)
        tokenizer = get_tokenizer("bpe")
    if getattr(tokenizer, "vocab_size", None) != CONFIG.vocab_size:
        print(
            f"  WARN: tokenizer vocab {getattr(tokenizer, 'vocab_size', '?')} != model vocab {CONFIG.vocab_size}",
            flush=True,
        )
    EOS_ID = getattr(tokenizer, "eos_token_id", 2)
    BOS_ID = getattr(tokenizer, "bos_token_id", 1)

    # Instruct mode: SFT checkpoints (finetune_enigma.py) carry meta.chat_format.
    # Base checkpoints get the plain-transcript bridge below. Attaching the chat
    # tokens is safe either way -- plain text encodes byte-identically.
    INSTRUCT = META.get("chat_format") == CHAT_FORMAT_NAME
    attach_chat_tokens(tokenizer)

    MEMORY = None
    if ARGS.memory_dir:
        from enigma_engine.core.memory_store import MemoryStore

        MEMORY = MemoryStore(ARGS.memory_dir)

    # Organs: constructed eagerly so a broken backend surfaces at startup, not
    # mid-conversation. She still serves text if an organ fails to come up.
    SPEAKER = None
    if ARGS.voice:
        try:
            SPEAKER = Speaker(voice=ARGS.voice_name)
        except TTSError as exc:
            print(f"  WARN: voice disabled -- {exc}", flush=True)

    MUTED = False
    try:
        _state = json.loads(_MUTE_STATE.read_text(encoding="utf-8"))
        if isinstance(_state, dict):
            MUTED = bool(_state.get("muted", False))
    except (OSError, ValueError):
        pass  # best-effort: a missing or corrupt state file must never stop serve

    EARS = None
    if ARGS.ears:
        try:
            EARS = Ears()
        except ASRError as exc:
            print(f"  WARN: ears disabled -- {exc}", flush=True)

    EYES = None
    if ARGS.eyes and _VISION_ENCODER is not None:
        try:
            # Her own eyes: aligned encoder + grafted projection + the served
            # frozen model, sharing the generation lock.
            EYES = Eyes(model=model, tokenizer=tokenizer, encoder=_VISION_ENCODER, gen_lock=_GEN_LOCK)
        except EyesError as exc:
            print(f"  WARN: eyes disabled -- {exc}", flush=True)

    PAINTER = None
    if ARGS.image_gen:
        try:
            PAINTER = Painter()
        except ImageGenError as exc:
            print(f"  WARN: image-gen disabled -- {exc}", flush=True)

    _n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Enigma loaded: {_n_params / 1e6:.1f}M params on {DEVICE}"
        + (f", checkpoint step {STEP:,}" if STEP is not None else "")
        + (f" | INSTRUCT ({META.get('chat_format')})" if INSTRUCT else " | base (transcript bridge)")
        + (f" | memory: {len(MEMORY)} entries" if MEMORY is not None else "")
        + (" | voice: on" if SPEAKER is not None else "")
        + (" | ears: on" if EARS is not None else "")
        + (" | eyes: on" if EYES is not None else "")
        + (" | image-gen: on" if PAINTER is not None else ""),
        flush=True,
    )
    _BOOTED = True  # LAST statement: readiness means boot ran to completion

app = FastAPI(title="Enigma (from-scratch)")


@app.middleware("http")
async def _require_boot(request, call_next):
    # An ASGI runner can mount this module without ever calling boot()
    # (`uvicorn serve_enigma:app`) -- which WORKED before startup moved out
    # of import time. Serving half-up is worse than refusing: health
    # endpoints would 200 while every generation request 500s with an opaque
    # NoneType error (re-audit 2026-07-18). Gate on boot COMPLETION, not on
    # `model is None` -- boot() assigns model early and can still die at the
    # max-context guard or tokenizer load (round-2 re-audit).
    if not _BOOTED:
        return JSONResponse(
            status_code=503,
            content={
                "detail": "server not booted -- start with 'python serve_enigma.py' "
                "(or call serve_enigma.boot() before mounting the app)"
            },
        )
    return await call_next(request)


# Transcript turn markers: a base model will happily continue the whole
# conversation, so cut her off when she starts writing the next turn.
_STOP_TEXTS = ("\nUser:", "\nEnigma:")


class Msg(BaseModel):
    role: str
    # str is the native shape; a list is the OpenAI multimodal content form,
    # flattened to str (eyes organ captions the images) before anything
    # downstream -- gates, memory retrieval, the renderer -- touches it.
    content: str | list | None = None
    tool_calls: list[dict] | None = None  # assistant history (instruct mode)


class ChatReq(BaseModel):
    model: str = MODEL_ID
    messages: list[Msg]
    # Clarity defaults for a 182M model (2026-07-15): 0.8/no-min_p read as
    # rambling; 0.3 + min_p keeps her coherent while staying non-greedy.
    temperature: float = 0.3
    top_p: float = 0.9
    min_p: float = 0.05  # 0 = off; prunes tokens below min_p * max_prob
    top_k: int = 50  # 0 = off
    repetition_penalty: float = 1.1  # applied to HER tokens only, never the prompt
    max_tokens: int = 256
    stream: bool = False
    tools: list[dict] | None = None  # OpenAI tool specs (instruct mode)


class CompletionReq(BaseModel):
    model: str = MODEL_ID
    prompt: str
    temperature: float = 0.8  # raw continuation keeps the exploratory default
    top_p: float = 0.9
    min_p: float = 0.0  # 0 = off; prunes tokens below min_p * max_prob
    top_k: int = 50  # 0 = off
    repetition_penalty: float = 1.1  # applied to generated tokens only
    max_tokens: int = 256
    stream: bool = False


def _render_transcript(messages: list[Msg]) -> str:
    """Plain-text bridge for a base model: render the conversation as a
    transcript she can continue. Replaced by a real chat template once the
    instruct pass defines special tokens."""
    lines = []
    for m in messages:
        text = (m.content or "").strip()
        if not text:
            continue
        if m.role == "system":
            lines.append(text)
        elif m.role == "assistant":
            lines.append(f"Enigma: {text}")
        else:
            lines.append(f"User: {text}")
    lines.append("Enigma:")
    return "\n".join(lines)


def _find_stop(text: str, stop_texts: tuple[str, ...]) -> int:
    hits = [i for i in (text.find(s) for s in stop_texts) if i != -1]
    return min(hits) if hits else -1


_GEN_DONE = object()


def _stream_ids_locked(
    x: torch.Tensor,
    max_tokens: int,
    temperature: float,
    top_p: float,
    min_p: float,
    stop_tokens: list[int],
    top_k: int = 50,
    repetition_penalty: float = 1.1,
):
    """Yield token ids from the model without holding _GEN_LOCK across
    consumer waits. A worker thread owns the lock for the whole generation
    and hands ids over an unbounded queue, so a slow or stalled SSE client
    can never keep other requests blocked on the lock; the ids themselves
    are bounded by max_tokens. A consumer that goes away (client disconnect)
    sets the cancel flag and the worker stops at the next token."""
    q: queue.Queue = queue.Queue()
    cancel = threading.Event()

    def worker():
        try:
            # bf16 autocast on CUDA (the numerics every training pass uses);
            # enabled=False leaves the fp32 path untouched on CPU or --fp32.
            with _GEN_LOCK, torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=_BF16_GEN
            ):
                for t in model.generate_stream(
                    x,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    stop_tokens=stop_tokens,
                    min_p=min_p,
                ):
                    if cancel.is_set():
                        break
                    q.put(int(t.item()))
        except BaseException as exc:
            # Also print: after a client disconnect nobody drains the queue,
            # and a generation error (e.g. CUDA state) must not vanish.
            print(f"generation worker error: {exc!r}", flush=True)
            q.put(exc)
        finally:
            q.put(_GEN_DONE)

    threading.Thread(target=worker, daemon=True).start()
    try:
        while True:
            item = q.get()
            if item is _GEN_DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        cancel.set()


def _generate_text(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop_texts: tuple[str, ...] = (),
    min_p: float = 0.0,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    stats: dict | None = None,
):
    """Yield text deltas from her KV-cached streaming path.

    Decoding re-decodes the full output each token (O(n²) chars, trivial at
    n<=max_tokens) so BPE merges never split mid-character across deltas. The
    last len(stop)-1 chars are held back until we know a stop marker isn't
    forming, then flushed.

    ``stats`` (if given) is filled once the generator finishes:
    prompt_tokens = ids actually fed, completion_tokens = ids actually
    sampled, finish = "stop" (eos / stop marker) or "length" (budget spent).
    Re-encoding the decoded text is NOT a substitute -- strip() plus BPE
    re-merge under-count what the model really produced.
    """
    # encode() brackets text as [BOS]...[EOS]; drop the trailing EOS so she
    # CONTINUES the prompt instead of seeing a finished document, and ensure
    # BOS survives any context trim (mirrors sample_enigma.py).
    ids = tokenizer.encode(prompt)
    if ids and ids[-1] == EOS_ID:
        ids = ids[:-1]
    # Clamp the GENERATION side too: she trains at block 1024, and the RoPE
    # table ends at 2x max_seq_len -- an unclamped client max_tokens could walk
    # past both. (2026-06-11 audit finding.)
    # Reserve the prompt first, generation second: keep the most recent
    # prompt context (up to max_context - MIN_GEN_TOKENS ids, leaving 1 for
    # BOS), then give generation whatever room is left. Prompt + generation
    # always fit in max_context; the prompt is never squeezed to a stub.
    prompt_cap = ARGS.max_context - MIN_GEN_TOKENS
    if len(ids) > prompt_cap - 1:
        ids = ids[-(prompt_cap - 1) :]  # keep the most recent context
    if not ids or ids[0] != BOS_ID:
        ids = [BOS_ID] + ids
    max_tokens = max(1, min(int(max_tokens), ARGS.max_context - len(ids)))
    if stats is None:
        stats = {}
    stats["prompt_tokens"] = len(ids)
    stats["completion_tokens"] = 0
    stats["finish"] = "stop"
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    # 0 (or below) means GREEDY -- the model argmaxes. Positive-but-tiny is
    # clamped: dividing logits by ~1e-9 overflows fp32.
    temperature = 0.0 if float(temperature) <= 0 else max(float(temperature), 1e-3)
    hold = max((len(s) for s in stop_texts), default=1) - 1
    emitted = 0
    saw_eos = False
    out_ids: list[int] = []
    for tid in _stream_ids_locked(
        x, max_tokens, temperature, top_p, min_p, [EOS_ID], top_k=top_k, repetition_penalty=repetition_penalty
    ):
        if tid == EOS_ID:
            saw_eos = True
            break
        out_ids.append(tid)
        text = tokenizer.decode(out_ids)
        cut = _find_stop(text, stop_texts)
        if cut != -1:
            stats["completion_tokens"] = len(out_ids)
            if cut > emitted:
                yield text[emitted:cut]
            return
        safe_end = max(emitted, len(text) - hold)
        if safe_end > emitted:
            yield text[emitted:safe_end]
            emitted = safe_end
    stats["completion_tokens"] = len(out_ids)
    if not saw_eos and len(out_ids) >= max_tokens:
        stats["finish"] = "length"  # budget spent, not a natural end
    # Natural end (eos or token budget): flush the held tail.
    text = tokenizer.decode(out_ids)
    cut = _find_stop(text, stop_texts)
    if cut != -1:
        # A stop marker completed inside the held tail: this is a marker
        # stop, not a budget cut.
        text = text[:cut]
        stats["finish"] = "stop"
    if len(text) > emitted:
        yield text[emitted:]


def _gen_ids(
    ids: list[int],
    max_tokens: int,
    temperature: float,
    top_p: float,
    min_p: float,
    stop_ids: tuple[int, ...],
    top_k: int = 50,
    repetition_penalty: float = 1.1,
):
    """ID-level generation for instruct mode: render_chat already built the
    exact prompt (BOS included, no trailing EOS -- the whole encode() EOS
    gotcha is bypassed). Yields raw token ids; stops on EOS/<|im_end|>."""
    # Defensive: prompt + generation must fit in max_context (caller already
    # sized max_tokens against len(ids), but never let a bad caller overflow).
    max_tokens = max(1, min(int(max_tokens), ARGS.max_context - len(ids)))
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    # 0 (or below) means GREEDY; positive-but-tiny is clamped (fp32 overflow).
    temperature = 0.0 if float(temperature) <= 0 else max(float(temperature), 1e-3)
    for tid in _stream_ids_locked(
        x, max_tokens, temperature, top_p, min_p, list(stop_ids), top_k=top_k, repetition_penalty=repetition_penalty
    ):
        if tid in stop_ids:
            break
        yield tid


def _sse_error_end(cid: str, created: int, object_name: str, exc: BaseException):
    """Terminal SSE frames for a mid-stream failure. HTTP 200 and a partial
    stream are already on the wire by then; without an explicit finish the
    client cannot tell a crash from a normal end (audit 2026-07-15). Emits a
    finish_reason "error" chunk, then [DONE]."""
    print(f"stream error: {exc!r}", flush=True)
    if object_name == "chat.completion.chunk":
        choice = {"index": 0, "delta": {}, "finish_reason": "error"}
    else:
        choice = {"index": 0, "text": "", "finish_reason": "error"}
    yield (
        "data: "
        + json.dumps({"id": cid, "object": object_name, "created": created, "model": MODEL_ID, "choices": [choice]})
        + "\n\n"
    )
    yield "data: [DONE]\n\n"


def _last_user_text(messages: list[Msg]) -> str:
    for m in reversed(messages):
        if m.role == "user" and m.content:
            return m.content
    return ""


# Built-in tools serve executes ITSELF (no client round-trip), in the same
# spec shape make_sft_data trains on (flat params). calculate: a from-scratch
# 182M model can't compute arithmetic in-weights (tokenizer splits numbers
# inconsistently). remember: the ChatGPT-bio-tool pattern -- she calls it when
# the user states a fact worth keeping, serve writes it to the MemoryStore,
# and render_context injects it back on every future relevant ask.
_CALC_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression and return the exact result.",
        "parameters": {"expression": "string"},
    },
}
_REMEMBER_TOOL = {
    "type": "function",
    "function": {
        "name": "remember",
        "description": "Save a fact about the user to long-term memory.",
        "parameters": {"text": "string"},
    },
}
_SPEAK_TOOL = {
    "type": "function",
    "function": {
        "name": "speak",
        "description": "Speak text out loud through the computer speakers.",
        "parameters": {"text": "string"},
    },
}
_IMAGINE_TOOL = {
    "type": "function",
    "function": {
        "name": "imagine",
        "description": "Generate an image from a text description and save it as a file.",
        "parameters": {"prompt": "string"},
    },
}
_BUILTIN_NAMES = {"calculate", "remember", "speak", "imagine"}
_MAX_TOOL_HOPS = 3  # bound the execute->regenerate loop so it can't spin

# The calculate tool is offered ONLY when the ask looks arithmetic. Injecting
# it on EVERY request poisoned normal chat: identity/factual training never saw
# a tool system-prompt, so a permanent one dragged the model into "calculator
# mode" (garbage like "your syntax violates the validity criterion" for
# "sum yourself up"). Gate on intent instead -- non-math chat gets no tool
# prompt and behaves; arithmetic gets the calculator.
_ARITH_KEYWORDS = re.compile(
    r"\b(plus|minus|times|divided|divide|multiply|multiplied|product|sum|subtract|"
    r"square root|squared|cubed|percent|modulo|remainder|to the power)\b",
    re.IGNORECASE,
)
# A digit next to an arithmetic operator, e.g. "7 * 8", "100 / 4", "2 ** 10".
_ARITH_SYMBOLS = re.compile(r"\d\s*[-+*/%^]\s*\d|\d\s*(?:x|X)\s*\d")


def _looks_arithmetic(text: str) -> bool:
    if not text:
        return False
    has_digit = any(c.isdigit() for c in text)
    return has_digit and bool(_ARITH_KEYWORDS.search(text) or _ARITH_SYMBOLS.search(text))


# remember is offered only when the message states something save-worthy:
# an explicit remember ask, or a first-person fact/preference. Same rationale
# as the calculate gate -- an ever-present tool prompt degrades normal chat.
_MEMORABLE = re.compile(
    r"\b(remember|don'?t forget|note (that|this)|keep in mind|save (this|that)|"
    r"call me|my name('s| is)|"
    # "my <up to 3 words> is/are": covers "my dog's name is", "my favorite
    # season is" (two attribute words -- a single-\w+ pattern missed it,
    # measured 2026-07-06). Offering is cheap; she decides whether to call.
    r"my (\w+('s)? ){1,3}(is|are)\b|"
    r"i (like|love|hate|prefer|live|work|drive|play|always|never|usually)|"
    r"i'?m (allergic|from|married|working))",
    re.IGNORECASE,
)


def _looks_memorable(text: str) -> bool:
    return bool(text) and MEMORY is not None and bool(_MEMORABLE.search(text))


# speak is offered only on an explicit say-it-out-loud ask, same rationale as
# the other gates. Shapes mirror the avatar_say SFT cases (say X / announce /
# read back / tell the room) plus "out loud"/"aloud" anywhere. Offering is
# cheap; she decides whether to call.
_SPEAKABLE = re.compile(
    r"\b(out loud|aloud|announce|use your voice|speak up|"
    r"say (it|this|that|something|hello|hi|good)|"
    r"tell (the room|everyone|everybody)|"
    r"read (it|this|that|.{0,30}) (back )?to me)\b",
    re.IGNORECASE,
)


def _looks_speakable(text: str) -> bool:
    return bool(text) and SPEAKER is not None and bool(_SPEAKABLE.search(text))


# imagine is offered on a make-me-a-picture ask: a creation verb within reach
# of an image noun, or the "image/picture of" idiom. Same philosophy as the
# other gates -- offering is cheap; she decides whether to call.
_IMAGINABLE = re.compile(
    r"\b(draw|paint|sketch|render|generate|create|make|imagine)\b[^.?!]{0,50}"
    r"\b(image|picture|photo|drawing|painting|art|illustration|logo|icon|wallpaper|scene)\b"
    r"|\b(image|picture|photo|drawing|painting) of\b",
    re.IGNORECASE,
)


def _looks_imaginable(text: str) -> bool:
    return bool(text) and PAINTER is not None and bool(_IMAGINABLE.search(text))


def _builtin_tools(user_text: str, client_mode: bool) -> list[dict]:
    """The built-ins to offer for this request. calculate rides along in
    client tool-mode (a tool prompt exists anyway; math grammar is distinctive
    enough that it never steals calls). remember is intent-gated ALWAYS:
    offered merely because other tools were, it stole tool calls -- measured
    2026-07-06, 'Check the weather in Toronto' -> remember("User's weather
    for Toronto is correct") instead of get_weather."""
    tools = []
    if client_mode or _looks_arithmetic(user_text):
        tools.append(_CALC_TOOL)
    if _looks_memorable(user_text):  # checks MEMORY is enabled too
        tools.append(_REMEMBER_TOOL)
    if _looks_speakable(user_text):  # checks SPEAKER is enabled too
        tools.append(_SPEAK_TOOL)
    if _looks_imaginable(user_text):  # checks PAINTER is enabled too
        tools.append(_IMAGINE_TOOL)
    return tools


def _execute_builtin(name: str, arguments: dict) -> str:
    """Run a server-side built-in tool and return its result string. Errors
    come back as text (fed to the model) rather than raising -- an engine that
    fails honestly beats one that 500s mid-conversation."""
    # The model can emit any valid JSON as "arguments" (a bare string parses
    # fine); only an object supports .get(), so anything else is a tool error
    # fed back to her, not an exception.
    if not isinstance(arguments, dict):
        return f"error: tool arguments must be a JSON object, got {type(arguments).__name__}"
    if name == "calculate":
        expr = str(arguments.get("expression", "")).strip()
        try:
            return format_result(evaluate(expr))
        except CalcError as exc:
            return f"error: {exc}"
    if name == "remember":
        if MEMORY is None:
            return "error: memory disabled (start serve with --memory-dir)"
        text = str(arguments.get("text", "")).strip()
        if not text:
            return "error: nothing to remember"
        rec = MEMORY.remember(text, source="chat")
        return f"updated: {rec['text']}" if rec.get("superseded") else f"saved: {rec['text']}"
    if name == "speak":
        if SPEAKER is None:
            return "error: voice disabled (start serve with --voice)"
        if MUTED:
            return "muted: voice is muted right now, so nothing was said out loud"
        text = str(arguments.get("text", "")).strip()
        if not text:
            return "error: nothing to say"
        try:
            SPEAKER.speak(text)  # fire-and-forget; playback runs while she writes her final turn
        except TTSError as exc:
            return f"error: {exc}"
        return "speaking"
    if name == "imagine":
        if PAINTER is None:
            return "error: image generation disabled (start serve with --image-gen)"
        prompt = str(arguments.get("prompt", "")).strip()
        if not prompt:
            return "error: empty prompt"
        try:
            path = PAINTER.generate(prompt, IMAGES_DIR / f"imagine_{uuid.uuid4().hex[:8]}.png")
        except ImageGenError as exc:
            return f"error: {exc}"
        return f"image saved to {path}"
    return f"error: unknown tool {name!r}"


def _with_context(msgs: list[dict], req: ChatReq) -> list[dict]:
    """Fold tool specs and retrieved memories into the system message. The
    built-in calculate tool is ALWAYS offered alongside any client tools."""
    extra = []
    if MEMORY is not None:
        mem = MEMORY.render_context(_last_user_text(req.messages), tokenizer, max_ids=128)
        if mem:
            extra.append(mem)
    # Built-ins are gated on intent (see _builtin_tools); client tools are
    # always honored.
    client_tools = list(req.tools or [])
    all_tools = _builtin_tools(_last_user_text(req.messages), bool(client_tools)) + client_tools
    if all_tools:
        tools_block = render_tools_system(all_tools)
        if not (msgs and msgs[0].get("role") == "system"):
            # Training's tool examples ALWAYS lead with this exact preamble
            # (make_sft_data._system, single \n before "Available tools:");
            # a system message that OPENS with "Available tools:" is a shape
            # the model never saw.
            tools_block = (
                "You are Enigma. You can use tools when they are needed; "
                "answer directly when they are not.\n" + tools_block
            )
        extra.append(tools_block)
    if not extra:
        return msgs
    if msgs and msgs[0].get("role") == "system":
        head = dict(msgs[0])
        head["content"] = "\n\n".join([head.get("content") or ""] + extra).strip()
        return [head] + msgs[1:]
    return [{"role": "system", "content": "\n\n".join(extra)}] + msgs


def _openai_tool_calls(calls: list[dict]) -> list[dict]:
    return [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": c.get("name"), "arguments": json.dumps(c.get("arguments", {}), ensure_ascii=False)},
        }
        for i, c in enumerate(calls)
        if c.get("name")
    ]


def _chat_instruct(req: ChatReq):
    msgs = _with_context([m.model_dump(exclude_none=True) for m in req.messages], req)
    created = int(time.time())
    cid = f"chatcmpl-{created}"

    def _hop(cur_msgs: list[dict]):
        # Reserve the prompt first (render into max_context - MIN_GEN_TOKENS ids),
        # then let generation take the room that's left; a large client max_tokens
        # can no longer starve the prompt.
        prompt_ids = render_chat(
            tokenizer, cur_msgs, add_generation_prompt=True, max_ids=ARGS.max_context - MIN_GEN_TOKENS
        )
        hop_max = max(1, min(int(req.max_tokens), ARGS.max_context - len(prompt_ids)))
        return prompt_ids, hop_max

    def _apply_builtins(
        cur_msgs: list[dict], out: dict, parsed: list[dict], results: list | None = None
    ) -> list[dict]:
        # Append the assistant turn that made the calls, then each built-in's
        # result, so the next hop sees a coherent tool trace. `results`
        # collects (name, result) so the caller can see what actually ran.
        named = [{"name": c["name"], "arguments": c.get("arguments") or {}} for c in parsed if c.get("name")]
        nxt = cur_msgs + [{"role": "assistant", "content": out.get("content") or "", "tool_calls": named}]
        for c in parsed:
            if c.get("name") in _BUILTIN_NAMES:
                result = _execute_builtin(c["name"], c.get("arguments") or {})
                if results is not None:
                    results.append((c["name"], result))
                nxt = nxt + [{"role": "tool", "content": result}]
        return nxt

    def _loop_on_builtins(parsed: list[dict], hop: int) -> bool:
        # Execute-and-loop ONLY when every named call is a built-in and hops
        # remain. A mixed batch (built-in + client tool) is surfaced whole
        # instead: executing half and looping would leave the client's call
        # dangling in the trace and never delivered. Likewise a built-in call
        # on the final hop is surfaced, not dropped -- the client seeing a
        # "calculate" call it can answer beats the action silently vanishing.
        named = [c for c in parsed if c.get("name")]
        return bool(named) and all(c["name"] in _BUILTIN_NAMES for c in named) and hop < _MAX_TOOL_HOPS

    if req.stream:

        def _events_body():
            # Span ids from the ATTACHED tokenizer (v1: identical to the
            # module constants; bigger vocab: derived rows -- HIGH-2).
            _ct = chat_token_ids(tokenizer)
            THINK, THINK_END = think_token_ids(tokenizer)
            TOOL_CALL, TOOL_CALL_END = _ct["<|tool_call|>"], _ct["<|/tool_call|>"]
            stop_ids = _stop_ids()

            cur_msgs = msgs
            raw_all: list[str] = []
            # Whether any content delta is already on the wire (across hops).
            # The non-stream path joins hop contents and raw calls with "\n";
            # emitting the same separator here keeps the two paths returning
            # byte-identical content for the same request (ultrareview #31).
            emitted_any = False
            for hop in range(_MAX_TOOL_HOPS + 1):
                prompt_ids, hop_max = _hop(cur_msgs)
                gen = _gen_ids(
                    prompt_ids, hop_max, req.temperature, req.top_p, req.min_p, stop_ids,
                    top_k=req.top_k, repetition_penalty=req.repetition_penalty,
                )
                all_ids: list[int] = []
                content_ids: list[int] = []
                emitted = 0
                depth = 0
                # Parity with non-stream, which returns parse_assistant_ids'
                # STRIPPED per-hop content: drop leading whitespace, hold any
                # trailing-whitespace run back until more non-whitespace
                # arrives, and discard it at hop end. Without this, a hop
                # ending in "\n" or a whitespace-only hop breaks the
                # byte-identical guarantee (re-audit 2026-07-17).
                hop_started = False
                pending_ws = ""
                for tid in gen:
                    all_ids.append(tid)
                    if tid in (THINK, TOOL_CALL):
                        depth += 1
                        continue
                    if tid in (THINK_END, TOOL_CALL_END):
                        depth = max(0, depth - 1)
                        continue
                    if depth:
                        continue  # span ids surface at the end, parsed -- not as text
                    content_ids.append(tid)
                    text = tokenizer.decode(content_ids, skip_special_tokens=True)
                    if len(text) <= emitted:
                        continue
                    delta = text[emitted:]
                    emitted = len(text)
                    chunk = pending_ws + delta
                    if not hop_started:
                        chunk = chunk.lstrip()
                        if not chunk:
                            pending_ws = ""
                            continue
                    body = chunk.rstrip()
                    pending_ws = chunk[len(body):]
                    if not body:
                        continue
                    if not hop_started and emitted_any:
                        body = "\n" + body  # hop-content separator, matches non-stream join
                    hop_started = True
                    emitted_any = True
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "id": cid,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": MODEL_ID,
                                "choices": [
                                    {"index": 0, "delta": {"content": body}, "finish_reason": None}
                                ],
                            }
                        )
                        + "\n\n"
                    )
                out = parse_assistant_ids(tokenizer, all_ids)
                parsed = out["tool_calls"]
                # Unparsable call text is collected across hops -- a malformed
                # action generated alongside an executed built-in surfaces at
                # the end instead of vanishing with the loop.
                raw_all += [c["raw"] for c in parsed if not c.get("name") and c.get("raw")]
                # A built-in-only batch (calculate) is executed here, then we
                # loop to let the model answer from the result -- the client
                # never sees it. Anything else is surfaced whole.
                if _loop_on_builtins(parsed, hop):
                    cur_msgs = _apply_builtins(cur_msgs, out, parsed)
                    continue
                calls = _openai_tool_calls(parsed)
                if raw_all:
                    raw_text = "\n".join(raw_all)
                    if emitted_any:
                        raw_text = "\n" + raw_text  # separator, matches non-stream join
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "id": cid,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": MODEL_ID,
                                "choices": [
                                    {"index": 0, "delta": {"content": raw_text}, "finish_reason": None}
                                ],
                            }
                        )
                        + "\n\n"
                    )
                if calls:
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "id": cid,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": MODEL_ID,
                                "choices": [{"index": 0, "delta": {"tool_calls": calls}, "finish_reason": None}],
                            }
                        )
                        + "\n\n"
                    )
                finish = "tool_calls" if calls else ("length" if len(all_ids) >= hop_max else "stop")
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "id": cid,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": MODEL_ID,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                        }
                    )
                    + "\n\n"
                )
                break
            yield "data: [DONE]\n\n"

        def events():
            try:
                yield from _events_body()
            except Exception as exc:
                yield from _sse_error_end(cid, created, "chat.completion.chunk", exc)

        return StreamingResponse(events(), media_type="text/event-stream")

    # Non-stream: run the built-in tool loop to completion, accumulating usage.
    cur_msgs = msgs
    n_prompt = n_out = 0
    out: dict = {"content": "", "tool_calls": []}
    out_ids: list[int] = []
    last_max = 1
    raw_all: list[str] = []
    # Spoken content from executed (looped) hops. The stream path has already
    # sent it by the time the loop decision lands, so keeping it here is what
    # makes stream and non-stream return the same content (ultrareview #31).
    # Trained tool calls carry empty content, so this is usually empty.
    hop_texts: list[str] = []
    spoke_server_side = False
    for hop in range(_MAX_TOOL_HOPS + 1):
        prompt_ids, last_max = _hop(cur_msgs)
        out_ids = list(
            _gen_ids(
                prompt_ids, last_max, req.temperature, req.top_p, req.min_p, _stop_ids(),
                top_k=req.top_k, repetition_penalty=req.repetition_penalty,
            )
        )
        out = parse_assistant_ids(tokenizer, out_ids)
        n_prompt += len(prompt_ids)
        n_out += len(out_ids)
        parsed = out["tool_calls"]
        # A tool call whose JSON didn't parse has no name -- its raw text is
        # collected across hops and surfaced as content instead of silently
        # dropping the model's action.
        raw_all += [c["raw"] for c in parsed if not c.get("name") and c.get("raw")]
        if _loop_on_builtins(parsed, hop):
            if out.get("content"):
                hop_texts.append(out["content"])
            tool_results: list = []
            cur_msgs = _apply_builtins(cur_msgs, out, parsed, tool_results)
            # Flag on the EXECUTED result, not the intent -- "error: nothing
            # to say" must not silence the page's own TTS (2026-07-17 audit).
            if any(name == "speak" and result == "speaking" for name, result in tool_results):
                spoke_server_side = True
            continue
        break

    calls = _openai_tool_calls(out["tool_calls"])
    content = "\n".join(t for t in [*hop_texts, out.get("content"), *raw_all] if t)
    message = {"role": "assistant", "content": content or (None if calls else "")}
    if calls:
        message["tool_calls"] = calls
    # honest finish_reason: a generation that spent the whole budget was cut
    # off ("length"), not naturally finished ("stop")
    finish = "tool_calls" if calls else ("length" if len(out_ids) >= last_max else "stop")
    resp = {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_out, "total_tokens": n_prompt + n_out},
    }
    if spoke_server_side:
        # Non-standard extension the chat page reads so it never double-voices
        # a reply the speak tool already said on the server's speakers.
        resp["enigma"] = {"spoke": True}
    return resp


# Built-in chat page (GET /): self-contained HTML+JS, no external assets
# (the server is offline by default and stays that way). Talks to the same
# /v1 API as any client; spoken replies are fetched from /v1/audio/speech
# and played IN THE BROWSER, so the mute button silences instantly and the
# volume mixes like any app (fine while gaming). Mute state lives on the
# SERVER (the tray icon can flip it too); the page polls it every 3 seconds
# and adopts changes, so a tray mute silences an already-open window.
_CHAT_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Enigma</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --bg:#101418; --panel:#1a2129; --me:#2b4a6f; --her:#232d38;
          --text:#dde5ec; --dim:#8899aa; --accent:#5aa9e6; --warn:#e6a55a; }
  * { box-sizing: border-box; margin: 0; }
  body { background:var(--bg); color:var(--text); font:16px/1.45 system-ui,sans-serif;
         display:flex; flex-direction:column; height:100vh; }
  header { display:flex; align-items:center; gap:12px; padding:10px 16px;
           background:var(--panel); border-bottom:1px solid #000; }
  header h1 { font-size:18px; font-weight:600; flex:1; }
  #voice-state { color:var(--dim); font-size:13px; }
  #mute { background:var(--accent); color:#08121c; border:0; border-radius:8px;
          padding:8px 18px; font-size:15px; font-weight:700; cursor:pointer; }
  #mute.muted { background:var(--warn); }
  #log { flex:1; overflow-y:auto; padding:16px; display:flex; flex-direction:column; gap:10px; }
  .msg { max-width:72%; padding:9px 13px; border-radius:12px; white-space:pre-wrap;
         overflow-wrap:break-word; }
  .me { background:var(--me); align-self:flex-end; border-bottom-right-radius:4px; }
  .her { background:var(--her); align-self:flex-start; border-bottom-left-radius:4px; }
  .sys { color:var(--dim); font-size:13px; align-self:center; }
  form { display:flex; gap:8px; padding:12px 16px; background:var(--panel); }
  #box { flex:1; background:var(--bg); color:var(--text); border:1px solid #333c46;
         border-radius:8px; padding:10px 12px; font-size:16px; outline:none; }
  #box:focus { border-color:var(--accent); }
  #send { background:var(--accent); color:#08121c; border:0; border-radius:8px;
          padding:0 22px; font-size:15px; font-weight:700; cursor:pointer; }
  button:disabled { opacity:.5; cursor:default; }
</style></head><body>
<header>
  <h1>Enigma</h1>
  <span id="voice-state">voice: checking...</span>
  <button id="mute" type="button">Mute</button>
</header>
<div id="log"></div>
<form id="f"><input id="box" autocomplete="off" placeholder="Say something to her..." autofocus>
<button id="send" type="submit">Send</button></form>
<script>
"use strict";
var history_ = [];
var muted = false;  // the SERVER owns mute; syncMute adopts the truth within 3s
var voiceReady = false;
var currentAudio = null;
var currentUrl = null;   // blob URL of the playing reply, revoked in stopAudio
var muteEpoch = 0;       // clicks invalidate in-flight polls (no stale revert)
var log = document.getElementById("log");
var box = document.getElementById("box");
var send = document.getElementById("send");
var muteBtn = document.getElementById("mute");
var voiceState = document.getElementById("voice-state");

function add(cls, text) {
  var d = document.createElement("div");
  d.className = "msg " + cls;
  d.textContent = text;
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
  return d;
}
function stopAudio() {
  if (currentAudio) { currentAudio.pause(); currentAudio = null; }
  if (currentUrl) { URL.revokeObjectURL(currentUrl); currentUrl = null; }
}
function paintMute() {
  muteBtn.textContent = muted ? "Muted" : "Mute";
  muteBtn.className = muted ? "muted" : "";
  if (voiceReady) voiceState.textContent = muted ? "voice: muted" : "voice: on";
}
function pushMute() {
  fetch("/v1/audio/mute", { method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ muted: muted }) }).catch(function () {});
}
function syncMute() {
  var epoch = muteEpoch;
  fetch("/v1/audio/mute")
    .then(function (r) { if (!r.ok) throw new Error(); return r.json(); })
    .then(function (s) {
      if (epoch !== muteEpoch) return;  // a click won since this poll left
      if (s.muted === muted) return;
      muted = s.muted;
      if (muted) stopAudio();
      paintMute();
    }).catch(function () {});
}
muteBtn.onclick = function () {
  muted = !muted;
  muteEpoch += 1;
  if (muted) stopAudio();
  paintMute();
  pushMute();
};
function speak(text) {
  if (!voiceReady || muted || !text) return;
  fetch("/v1/audio/speech", { method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input: text }) })
    .then(function (r) {
      if (r.status === 204) return null;
      if (!r.ok) throw new Error();
      return r.blob(); })
    .then(function (b) {
      if (!b || muted) return;
      stopAudio();
      currentUrl = URL.createObjectURL(b);
      var a = new Audio(currentUrl);
      currentAudio = a;
      a.addEventListener("ended", function () { if (currentAudio === a) stopAudio(); });
      a.play().catch(function () {});
    }).catch(function () {});
}
document.getElementById("f").onsubmit = function (ev) {
  ev.preventDefault();
  var text = box.value.trim();
  if (!text || send.disabled) return;
  box.value = "";
  add("me", text);
  history_.push({ role: "user", content: text });
  send.disabled = true;
  var thinking = add("sys", "...");
  fetch("/v1/chat/completions", { method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: "enigma", messages: history_ }) })
    .then(function (r) {
      if (!r.ok) return r.text().then(function (t) { throw new Error(t.slice(0, 200)); });
      return r.json();
    })
    .then(function (data) {
      thinking.remove();
      var msg = (data.choices && data.choices[0] && data.choices[0].message) || {};
      var reply = (typeof msg.content === "string") ? msg.content : "";
      if (!reply) {
        add("sys", "[no text reply]");  // shown, never spoken or remembered
        return;
      }
      add("her", reply);
      history_.push({ role: "assistant", content: reply });
      if (!(data.enigma && data.enigma.spoke)) speak(reply);
    })
    .catch(function (e) {
      thinking.remove();
      add("sys", "error: " + e.message);
    })
    .then(function () { send.disabled = false; box.focus(); });
};
fetch("/v1/audio/voices")
  .then(function (r) { if (!r.ok) throw new Error(); voiceReady = true; })
  .catch(function () { voiceReady = false; })
  .then(function () {
    voiceState.textContent = voiceReady
      ? (muted ? "voice: muted" : "voice: on")
      : "voice: off (start with --voice)";
    if (voiceReady) syncMute();
  });
paintMute();
setInterval(syncMute, 3000);
</script></body></html>
"""


@app.get("/", response_class=HTMLResponse)
def chat_page():
    return _CHAT_PAGE


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "enigma"}]}


@app.post("/v1/chat/completions")
def chat(req: ChatReq):
    # Validate roles up front: an unknown role is CLIENT error (400), not a
    # server crash (500). Without this the ValueError from chat_format's
    # renderer leaks a stack trace and returns a generic 500.
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty list")
    bad = sorted({m.role for m in req.messages if m.role not in ROLES})
    if bad:
        raise HTTPException(status_code=400, detail=f"unknown chat role(s) {bad}; need one of {list(ROLES)}")
    # Eyes organ: caption multimodal content into plain text BEFORE the tool
    # gates, memory retrieval, or either render path sees the messages.
    for m in req.messages:
        if isinstance(m.content, list):
            m.content = flatten_image_content(m.content, EYES.describe if EYES is not None else None)
    if INSTRUCT:
        return _chat_instruct(req)
    messages = list(req.messages)
    if MEMORY is not None:
        mem = MEMORY.render_context(_last_user_text(messages), tokenizer, max_ids=128)
        if mem:
            messages = [Msg(role="system", content=mem)] + messages
    prompt = _render_transcript(messages)
    created = int(time.time())
    cid = f"chatcmpl-{created}"
    stats: dict = {}
    gen = _generate_text(
        prompt, req.max_tokens, req.temperature, req.top_p, _STOP_TEXTS,
        min_p=req.min_p, top_k=req.top_k, repetition_penalty=req.repetition_penalty, stats=stats,
    )

    if req.stream:

        def _events_body():
            for delta in gen:
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "id": cid,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": MODEL_ID,
                            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                        }
                    )
                    + "\n\n"
                )
            yield (
                "data: "
                + json.dumps(
                    {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_ID,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": stats.get("finish", "stop")}],
                    }
                )
                + "\n\n"
            )
            yield "data: [DONE]\n\n"

        def events():
            try:
                yield from _events_body()
            except Exception as exc:
                yield from _sse_error_end(cid, created, "chat.completion.chunk", exc)

        return StreamingResponse(events(), media_type="text/event-stream")

    text = "".join(gen).strip()
    # Usage is ground truth from _generate_text: ids actually fed and ids
    # actually sampled. (Re-encoding the stripped text under-counted -- BPE
    # re-merge + strip() lost tokens the model really produced.)
    n_prompt = stats["prompt_tokens"]
    n_out = stats["completion_tokens"]
    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": stats["finish"]}],
        "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_out, "total_tokens": n_prompt + n_out},
    }


@app.post("/v1/completions")
def completions(req: CompletionReq):
    created = int(time.time())
    cid = f"cmpl-{created}"
    stats: dict = {}
    gen = _generate_text(
        req.prompt, req.max_tokens, req.temperature, req.top_p,
        min_p=req.min_p, top_k=req.top_k, repetition_penalty=req.repetition_penalty, stats=stats,
    )

    if req.stream:

        def _events_body():
            for delta in gen:
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "id": cid,
                            "object": "text_completion",
                            "created": created,
                            "model": MODEL_ID,
                            "choices": [{"index": 0, "text": delta, "finish_reason": None}],
                        }
                    )
                    + "\n\n"
                )
            yield (
                "data: "
                + json.dumps(
                    {
                        "id": cid,
                        "object": "text_completion",
                        "created": created,
                        "model": MODEL_ID,
                        "choices": [{"index": 0, "text": "", "finish_reason": stats.get("finish", "stop")}],
                    }
                )
                + "\n\n"
            )
            yield "data: [DONE]\n\n"

        def events():
            try:
                yield from _events_body()
            except Exception as exc:
                yield from _sse_error_end(cid, created, "text_completion", exc)

        return StreamingResponse(events(), media_type="text/event-stream")

    text = "".join(gen)
    # Usage is ground truth from _generate_text: ids actually fed and sampled.
    n_prompt = stats["prompt_tokens"]
    n_out = stats["completion_tokens"]
    return {
        "id": cid,
        "object": "text_completion",
        "created": created,
        "model": MODEL_ID,
        "choices": [{"index": 0, "text": text, "finish_reason": stats["finish"]}],
        "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_out, "total_tokens": n_prompt + n_out},
    }


class MemReq(BaseModel):
    text: str
    kind: str = "fact"


def _organ_off(what: str) -> HTTPException:
    """Disabled organ/store: a real 503, not a 200 with an error body -- an
    OpenAI-SDK client treats 200 as success and mis-handles the JSON as the
    payload it asked for (audit 2026-07-15). Organ FAILURES raise 500 the
    same way; only the detail text differs."""
    return HTTPException(status_code=503, detail=what)


@app.post("/v1/memory")
def memory_add(req: MemReq):
    if MEMORY is None:
        raise _organ_off("memory disabled -- start with --memory-dir")
    try:
        return {"ok": True, "memory": MEMORY.add(req.text, kind=req.kind)}
    except ValueError as exc:
        # client-input error (empty/whitespace text), not a server crash
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v1/memory")
def memory_list(q: str | None = None, k: int = 5):
    if MEMORY is None:
        raise _organ_off("memory disabled -- start with --memory-dir")
    recs = MEMORY.search(q, k=k) if q else MEMORY.all()[-k:]
    return {"count": len(MEMORY), "results": recs}


@app.delete("/v1/memory/{mem_id}")
def memory_delete(mem_id: int):
    """User control over her memory (the ChatGPT-memory-management parallel):
    a saved fact can always be inspected (GET) and removed."""
    if MEMORY is None:
        raise _organ_off("memory disabled -- start with --memory-dir")
    return {"ok": MEMORY.delete(mem_id), "count": len(MEMORY)}


@app.delete("/v1/memory")
def memory_clear():
    if MEMORY is None:
        raise _organ_off("memory disabled -- start with --memory-dir")
    return {"ok": True, "cleared": MEMORY.clear()}


class SpeechReq(BaseModel):
    """OpenAI audio.speech shape (subset): synthesize input, return WAV bytes.
    This is the organ's service face -- the avatar requests audio it plays
    itself (lip-sync later); the speak TOOL plays on this machine instead."""

    model: str = MODEL_ID
    input: str
    voice: str | None = None  # one system voice for now; reject others honestly


@app.post("/v1/audio/speech")
def audio_speech(req: SpeechReq):
    if SPEAKER is None:
        raise _organ_off("voice disabled -- start with --voice")
    if req.voice is not None:
        raise HTTPException(status_code=400, detail="voice selection not supported yet -- one system voice")
    if MUTED:
        return Response(status_code=204)
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        SPEAKER.save_wav(req.input, tmp)
        wav = Path(tmp).read_bytes()
    except TTSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        Path(tmp).unlink(missing_ok=True)
    return Response(content=wav, media_type="audio/wav")


@app.get("/v1/audio/voices")
def audio_voices():
    if SPEAKER is None:
        raise _organ_off("voice disabled -- start with --voice")
    return {"voices": SPEAKER.voices}


class MuteReq(BaseModel):
    muted: bool


@app.get("/v1/audio/mute")
def get_mute():
    return {"muted": MUTED}


@app.post("/v1/audio/mute")
def set_mute(req: MuteReq):
    """The mute switch (chat page button + tray icon). Gates the server-side
    speak TOOL and turns /v1/audio/speech into 204s; persisted so a restart
    cannot silently unmute."""
    global MUTED
    MUTED = bool(req.muted)
    try:
        _MUTE_STATE.parent.mkdir(parents=True, exist_ok=True)
        _MUTE_STATE.write_text(json.dumps({"muted": MUTED}), encoding="utf-8")
    except OSError:
        pass  # mute still works for this run; it just won't survive a restart
    return {"muted": MUTED}


@app.post("/v1/audio/transcriptions")
def audio_transcriptions(file: UploadFile = File(...)):
    """OpenAI audio.transcriptions shape (subset): upload audio, get text.
    The ears organ -- clients (push-to-talk, the avatar) send what they hear
    and feed the text back into chat."""
    if EARS is None:
        raise _organ_off("ears disabled -- start with --ears")
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        Path(tmp).write_bytes(file.file.read())
        return EARS.transcribe(tmp)
    except ASRError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        Path(tmp).unlink(missing_ok=True)


@app.post("/v1/images/describe")
def images_describe(file: UploadFile = File(...)):
    """The eyes organ's direct face: upload an image, get her caption.
    (In chat, OpenAI-style image messages are captioned automatically.)"""
    if EYES is None:
        raise _organ_off("eyes disabled -- start with --eyes")
    try:
        return {"description": EYES.describe(file.file.read())}
    except EyesError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class ImageGenReq(BaseModel):
    """OpenAI images.generations shape (subset)."""

    model: str = MODEL_ID
    prompt: str
    n: int = 1
    size: str = "512x512"


@app.post("/v1/images/generations")
def images_generations(req: ImageGenReq):
    if PAINTER is None:
        raise _organ_off("image generation disabled -- start with --image-gen")
    try:
        width, height = (int(x) for x in req.size.lower().split("x"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"bad size {req.size!r}; use WIDTHxHEIGHT like 512x512") from exc
    data = []
    for _ in range(max(1, min(int(req.n), 4))):  # bound n: VRAM is shared with the LLM
        out = IMAGES_DIR / f"gen_{uuid.uuid4().hex[:8]}.png"
        try:
            path = PAINTER.generate(req.prompt, out, width=width, height=height)
        except ImageGenError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        data.append({"b64_json": base64.b64encode(path.read_bytes()).decode("ascii")})
    return {"created": int(time.time()), "data": data}


def main() -> None:
    """Run the server. Console-script entry point (pyproject [project.scripts])
    and the __main__ path share this."""
    boot()
    print(f"Enigma OpenAI-compatible API -> http://{ARGS.host}:{ARGS.port}/v1", flush=True)
    print(f"In Odysseus:  /setup local http://{ARGS.host}:{ARGS.port}/v1", flush=True)
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="warning")


if __name__ == "__main__":
    main()
