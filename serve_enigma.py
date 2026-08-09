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
import hashlib
import json
import math
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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from enigma_engine.core.chat_format import (
    BUILTIN_NAMES,
    CHAT_FORMAT_NAME,
    ROLES,
    attach_chat_tokens,
    builtin_tool,
    chat_token_ids,
    parse_assistant_ids,
    render_chat,
    render_tools_system,
    search_token_ids,
    think_token_ids,
)
from enigma_engine.core.search import DEFAULT_K as SEARCH_DEFAULT_K
from enigma_engine.core.search import Searcher, SearchError, render_results
from enigma_engine.core.asr import ASRError, Ears
from enigma_engine.core.calculator import CalcError, evaluate, format_result
from enigma_engine.core.eyes import Eyes, EyesError, flatten_image_content
from enigma_engine.core.imagegen import ImageGenError, Painter
from enigma_engine.core.tts import Speaker, TTSError, list_output_devices
from enigma_engine.core.model import Enigma
from enigma_engine.core.model_presets import ForgeConfig
from enigma_engine.core.memory_store import renders_forget_pending as _renders_forget_pending
from enigma_engine.core.persona import Persona
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
    # The ADOPTED model -- same posture as every launcher. The old default was
    # the raw pretrain checkpoint, so a bare `enigma` console script served the
    # wrong brain (audit 2026-07-17). ADOPTED 2026-08-08 (Gate D): the v2
    # lineage's SFT-2 checkpoint (67/120 vs v8 56, paired p=0.0433 -- the first
    # candidate the sealed gate could distinguish). enigma_dpo (v8) stays on
    # disk as the byte-identical rollback.
    default=str(ROOT / "models" / "enigma_v2_sft2" / "model.pth"),
    help="Enigma checkpoint (.pth with model_state_dict + config); default = the adopted v2 model",
)
_p.add_argument("--host", default="127.0.0.1")
_p.add_argument("--port", type=int, default=8000)
_p.add_argument(
    "--max-context",
    type=int,
    # SUNSET at Gate D adoption 2026-08-08: the adopted v2 lineage trains at
    # block 2048, so the default follows it. Safe for any checkpoint -- boot()
    # caps this down to the model's own KV-cache capacity with a WARN when a
    # smaller model is served, so a 1024-trained checkpoint still serves at
    # 1024. The old 1024 default silently truncated the v2 context win.
    default=2048,
    help="prompt+generation token budget; the adopted v2 model trains at block 2048 "
    "(capped down to a smaller model's KV-cache capacity with a WARN)",
)
_p.add_argument(
    "--memory-dir",
    default=None,
    help="enable the local memory store (JSONL + BM25); relevant memories are injected into her system context",
)
_p.add_argument(
    "--memory-recall",
    type=int,
    default=5,
    help="how many remembered facts the injected memory block may carry; the token "
    "budget trims further, and 0 keeps the store readable while injecting nothing",
)
_p.add_argument(
    "--persona",
    default=None,
    help="serve a DIFFERENT AI: a persona pack (JSON) giving her name, data home and "
    "the meaning of that name. Omitted = Enigma, which is this repo's own identity "
    "and what every default reproduces exactly",
)
_p.add_argument(
    "--voice",
    action="store_true",
    help="enable the voice organ: the speak built-in tool + /v1/audio/speech (local Kokoro-82M)",
)
_p.add_argument(
    "--voice-name",
    default=None,
    help="override the voice with a single Kokoro preset, e.g. 'af_heart' (default: her saved blend)",
)
_p.add_argument(
    "--voice-device",
    default=None,
    help="which speaker she talks out of: a device name, part of one, or an index from "
    "--list-audio-outputs. 'default' follows the Windows default output. The choice "
    "persists, so this is only needed to CHANGE it (default: her saved choice)",
)
_p.add_argument(
    "--list-audio-outputs",
    action="store_true",
    help="print the output devices --voice-device accepts, then exit",
)
_p.add_argument(
    "--barge-in",
    action="store_true",
    help="stop speaking when the mic hears you talk (energy VAD; needs headphones or AEC -- tune live)",
)
_p.add_argument(
    "--barge-in-threshold",
    type=float,
    default=None,
    help="RMS loudness that counts as you talking (default 0.02); raise if she cuts herself off, lower if she ignores you",
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
    "--search",
    action="store_true",
    help="enable the search organ: a <search>query</search> span in her output runs a lookup "
    "through the machine's own SearXNG and the results return to her context. Needs a vocab "
    "that carves the tags (v2); on older vocabs the feature is honestly absent",
)
_p.add_argument(
    "--search-url",
    default=None,
    help="SearXNG base URL (default http://127.0.0.1:8888 -- the local WSL2 docker instance; "
    "queries leave this machine only through THAT service's user-configured engines)",
)
_p.add_argument(
    "--search-k",
    type=int,
    default=SEARCH_DEFAULT_K,
    help="results per lookup fed back into her context",
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
# Which weights answered, as bytes: two same-architecture checkpoints (v5 vs
# v8) have identical key sets, shapes, and often the same step field -- only
# the file hash distinguishes them, and the eval gate records it from
# /v1/models so a locked transcript proves WHAT was measured.
MODEL_PATH: str | None = None
MODEL_SHA256: str | None = None
META: dict = {}
INSTRUCT = False
MEMORY = None
MEMORY_RECALL = 5
SPEAKER = None
MUTED = False
EARS = None
EYES = None
PAINTER = None
SEARCHER = None
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

# WHO this server is serving. Identity lives in a persona pack so the trainer
# can mold a different AI instead of a second Enigma; with no pack this IS
# Enigma, and every string below is what the literals it replaced already said.
PERSONA = Persona.load()

# The runtime-editable voice recipe (Kokoro blend + speed). Lives in the
# engine's data home so set_voice edits survive restarts and are shared by any
# launcher, wherever it was started from. Absent = the shipped Cortana default.
_VOICE_STATE = PERSONA.home / "voice.json"

# Talk-mode: when ON, the chat window speaks EVERY reply out loud (conversation
# mode); when OFF, she stays quiet unless a reply used the speak tool. Distinct
# from mute (a hard silence). Server-owned + persisted like mute; defaults OFF
# so enabling the voice organ never surprises the user with narration.
_TALK_STATE = Path(__file__).resolve().parent / "data" / "talk_mode.json"
TALK_MODE = False

# Bumped by POST /v1/audio/stop. An open chat window polls it and hushes its own
# browser audio when it changes, so a desktop/tray Stop reaches the window too
# (server-side playback is aborted directly via SPEAKER.stop()).
_STOP_GEN = 0


def _write_state_atomic(path: Path, obj: dict) -> None:
    """Persist a small state file so a crash mid-write can never leave a corrupt
    file that loads as the wrong default on the next boot -- the exact case the
    mute-state comment promises against (a half-written mute_state.json must not
    silently unmute a muted gaming session). Write-temp + os.replace is atomic
    on the same volume; a failed write is swallowed (state still holds this run).
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(obj), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass

# Where the imagine tool and /v1/images/generations drop their PNGs: the
# engine's data home, not the repo checkout.
IMAGES_DIR = PERSONA.home / "images"


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
        eck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
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
    global ARGS, CONFIG, model, tokenizer, DEVICE, _BF16_GEN, STEP, META, MODEL_PATH, MODEL_SHA256
    global INSTRUCT, MEMORY, MEMORY_RECALL, SPEAKER, MUTED, TALK_MODE, EARS, EYES, PAINTER, SEARCHER, EOS_ID, BOS_ID
    global _BOOTED, PERSONA, _VOICE_STATE, IMAGES_DIR, _STOP_TEXTS

    _BOOTED = False  # a re-boot is unready until it completes

    # parse_known_args is deliberate (the server must start under runners
    # that carry their own argv) -- but a typo'd flag must not silently
    # disable an organ, so unknowns are named out loud.
    ARGS, unknown = _p.parse_known_args(argv)
    if unknown:
        print(f"  WARN: ignoring unrecognized args: {' '.join(unknown)}", flush=True)

    # WHO is being served. The module-level defaults are Enigma; a pack rebinds
    # the values derived from her identity. Rebinding here rather than reading
    # PERSONA at each use keeps those names plain constants for every other
    # reader, and a boot is the only moment identity can change.
    # UNCONDITIONAL. Rebinding only when the flag is present left a flagless
    # re-boot serving the PREVIOUS persona -- the same shape as the double-boot
    # environment hole fixed thirty lines below, and the reason a boot must be
    # the moment identity is decided rather than the moment it can change.
    PERSONA = Persona.load(Path(ARGS.persona) if ARGS.persona else None)
    _VOICE_STATE = PERSONA.home / "voice.json"
    IMAGES_DIR = PERSONA.home / "images"
    _STOP_TEXTS = ("\nUser:", PERSONA.transcript_label)
    if ARGS.persona:
        print(f"  persona: {PERSONA.name} (home {PERSONA.home})", flush=True)

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
    MODEL_PATH = str(Path(ARGS.model).resolve())
    _h = hashlib.sha256()
    with open(ARGS.model, "rb") as _f:
        for _chunk in iter(lambda: _f.read(1 << 22), b""):
            _h.update(_chunk)
    MODEL_SHA256 = _h.hexdigest()
    print(f"  checkpoint sha256 {MODEL_SHA256[:16]}...", flush=True)
    _ck = torch.load(ARGS.model, map_location="cpu", weights_only=True)  # our own checkpoint
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
        # Only the fallback can mismatch: the selected path is exact-or-padding
        # by construction, and padded sizes (table rounded up to 64) are not a
        # mismatch worth warning about.
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
    # The chat/tool specials sit in the first alignment-padding rows and are
    # trained there. Sampling masks everything past the live vocab, so without
    # this the model's own <|tool_call|> and <|im_end|> are -inf'd out of every
    # reply -- tools, built-ins and clean turn endings all die silently.
    _live_vocab = max(chat_token_ids(tokenizer).values()) + 1
    _head_width = model.output.weight.shape[0]
    if not INSTRUCT:
        # A BASE checkpoint never trained those rows -- they are random-init,
        # exactly what the pad-row guard exists to keep out of argmax, and the
        # base decode path renders specials literally. Only an instruct
        # lineage has earned the declaration. (T2/T3 produce this class.)
        pass
    elif _live_vocab <= _head_width:
        # Declare what the tokenizer can actually decode, ALWAYS -- including
        # when that is below config.vocab_size. Skipping the declaration there
        # looked harmless (the default mask "already keeps" the specials) but
        # the default masks at config.vocab_size, so on a checkpoint whose
        # vocab is a multiple of 64 it masks nothing at all and every
        # undecodable row becomes samplable.
        model.set_live_vocab_size(_live_vocab)
        if _live_vocab < CONFIG.vocab_size:
            # The chat ids are landing on rows this checkpoint calls REAL
            # vocab: the tokenizer table is smaller than the checkpoint
            # declares, so <|im_start|> and friends ALIAS learned tokens
            # (chat_format records the measured case: id 4718 decoded as
            # ' crashes' on a 5,996-row vocab). Turn boundaries and tool-call
            # parsing will fire on ordinary text. Say so loudly.
            print(
                f"WARN: tokenizer table ends at {_live_vocab} but the checkpoint declares "
                f"{CONFIG.vocab_size} rows; chat/tool tokens ALIAS trained vocab -- "
                "turn boundaries and tool parsing are unreliable on this pairing",
                flush=True,
            )
    else:
        # The chat ids sit outside the head entirely -- the model cannot emit
        # them at all. Say so rather than crashing the boot over it.
        print(
            f"WARN: chat tokens need {_live_vocab} rows but the model head has {_head_width}; "
            "tool calls and <|im_end|> are unavailable on this checkpoint",
            flush=True,
        )

    MEMORY = None
    MEMORY_RECALL = max(0, ARGS.memory_recall)
    if ARGS.memory_recall < 0:
        print(
            f"  WARN: --memory-recall {ARGS.memory_recall} is below zero; recalling nothing",
            flush=True,
        )
    if ARGS.memory_dir:
        from enigma_engine.core.memory_store import MemoryStore

        # Same contract as the organs below: a memory dir that cannot be opened
        # (locked by another process, unwritable, full) costs her memory, not
        # the whole server. MemoryStore mkdirs and reads the file in __init__,
        # so both raise here.
        try:
            MEMORY = MemoryStore(ARGS.memory_dir)
        except Exception as exc:
            print(f"  WARN: memory disabled -- {exc}", flush=True)

    # Organs: constructed eagerly so a broken backend surfaces at startup, not
    # mid-conversation. She still serves text if an organ fails to come up.
    SPEAKER = None
    if ARGS.voice:
        try:
            SPEAKER = Speaker(recipe_path=_VOICE_STATE, voice_name=ARGS.voice_name)
        except TTSError as exc:
            print(f"  WARN: voice disabled -- {exc}", flush=True)
    if SPEAKER is not None:
        # A --voice-device that no longer exists keeps the SAVED choice rather
        # than falling back silently: the flag was an instruction, and losing it
        # without a word is how she ends up talking to an empty room again.
        if ARGS.voice_device is not None:
            try:
                SPEAKER.set_output_device(ARGS.voice_device)
            except TTSError as exc:
                print(f"  WARN: --voice-device ignored -- {exc}", flush=True)
        _out = SPEAKER.get_output_device()
        print(f"  voice output: {_out if _out else 'system default'}", flush=True)

    # Barge-in: let the mic cut her off when the user talks over her. The mic
    # opens only while she speaks (via the speaking-state callback). Off unless
    # asked -- energy VAD self-triggers on speakers without echo cancellation.
    if SPEAKER is not None and ARGS.barge_in:
        try:
            from enigma_engine.core.barge_in import DEFAULT_THRESHOLD, MicBargeIn

            _threshold = ARGS.barge_in_threshold if ARGS.barge_in_threshold is not None else DEFAULT_THRESHOLD
            # A threshold <= 0 fires on pure silence (she cuts herself off 0.25s
            # into every utterance); NaN/inf never fires while claiming "on".
            if not math.isfinite(_threshold) or _threshold <= 0:
                print(f"  WARN: --barge-in-threshold {_threshold} is unusable (must be a positive number); using {DEFAULT_THRESHOLD}", flush=True)
                _threshold = DEFAULT_THRESHOLD
            _bargein = MicBargeIn(on_detect=SPEAKER.stop, threshold=_threshold)
            SPEAKER.set_on_speaking(_bargein.set_active)
            print(f"  barge-in: on (energy VAD, threshold {_threshold} -- retune with --barge-in-threshold)", flush=True)
        except Exception as exc:
            print(f"  WARN: barge-in disabled -- {exc}", flush=True)

    MUTED = False
    try:
        _state = json.loads(_MUTE_STATE.read_text(encoding="utf-8"))
        if isinstance(_state, dict):
            MUTED = bool(_state.get("muted", False))
    except (OSError, ValueError):
        pass  # best-effort: a missing or corrupt state file must never stop serve

    TALK_MODE = False  # OFF until the user turns it on -- enabling voice is silent
    try:
        _tstate = json.loads(_TALK_STATE.read_text(encoding="utf-8"))
        if isinstance(_tstate, dict):
            TALK_MODE = bool(_tstate.get("enabled", False))
    except (OSError, ValueError):
        pass

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
        except Exception as exc:
            # Not just EyesError: construction moves the encoder onto the
            # device, which raises torch.OutOfMemoryError (a RuntimeError) when
            # the GPU is busy -- the case this machine hits while gaming.
            print(f"  WARN: eyes disabled -- {exc}", flush=True)

    PAINTER = None
    if ARGS.image_gen:
        try:
            PAINTER = Painter()
        except ImageGenError as exc:
            print(f"  WARN: image-gen disabled -- {exc}", flush=True)

    SEARCHER = None
    if ARGS.search:
        _search_ids_ok = None not in search_token_ids(tokenizer)
        if not _search_ids_ok:
            # The v1 table predates the tags: the model cannot emit a span the
            # vocab does not carve, so the organ would sit unreachable. Say so
            # instead of pretending search is on.
            print("  WARN: search disabled -- this vocab carries no <search> tags (needs the v2 table)", flush=True)
        else:
            try:
                SEARCHER = Searcher(base_url=ARGS.search_url) if ARGS.search_url else Searcher()
                # Reachability is a per-query property, never a boot gate: the
                # backend lives in WSL, whose VM sleeps and wakes AFTER serve
                # starts. The probe only makes the boot log honest.
                _up = "reachable" if SEARCHER.probe() else "NOT reachable yet (checked per query)"
                print(f"  search: SearXNG at {SEARCHER.base_url} -- {_up}", flush=True)
            except SearchError as exc:
                print(f"  WARN: search disabled -- {exc}", flush=True)

    _n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Enigma loaded: {_n_params / 1e6:.1f}M params on {DEVICE}"
        + (f", checkpoint step {STEP:,}" if STEP is not None else "")
        + (f" | INSTRUCT ({META.get('chat_format')})" if INSTRUCT else " | base (transcript bridge)")
        + (f" | memory: {len(MEMORY)} entries" if MEMORY is not None else "")
        + (" | voice: on" if SPEAKER is not None else "")
        + (" | ears: on" if EARS is not None else "")
        + (" | eyes: on" if EYES is not None else "")
        + (" | image-gen: on" if PAINTER is not None else "")
        + (" | search: on" if SEARCHER is not None else ""),
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
_STOP_TEXTS = ("\nUser:", PERSONA.transcript_label)


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
            lines.append(f"{PERSONA.name}: {text}")
        else:
            lines.append(f"User: {text}")
    # The label MUST be the one _STOP_TEXTS cuts on. Hardcoding "Enigma:" here
    # while the stop text followed the persona meant a --persona run prompted
    # her as Enigma and then had no stop sequence that could fire on the turn
    # marker it had just taught -- the reply kept a whole fabricated assistant
    # turn and was only cut at the next "\nUser:".
    lines.append(f"{PERSONA.name}:")
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


def _sse_role_open(cid: str, created: int) -> str:
    """The opening chunk of a chat stream: the assistant role, no content.

    OpenAI clients read the role from the first frame to open the message,
    and this one carries no "content" key, so a consumer joining the content
    deltas sees the same bytes with or without it. It is yielded before the
    prompt is encoded, which is what puts something on the wire during the
    prefill and tool hops rather than leaving the connection silent."""
    return (
        "data: "
        + json.dumps(
            {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        )
        + "\n\n"
    )


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


def _recent_user_text(messages: list[Msg], n: int = 3) -> str:
    """The last N user turns joined, for MEMORY RETRIEVAL only. Keying recall on
    the single last message blanked the memory block on a follow-up ("and what
    did I say my dog was called?" shares no term with the stored fact). Widening
    the query to the recent turns keeps the referenced fact reachable. The
    tool-offer gates still key on the LAST message -- they are about the current
    ask, not the thread."""
    users = [m.content for m in messages if m.role == "user" and m.content]
    return " ".join(users[-n:])


# Built-in tools serve executes ITSELF (no client round-trip), in the same
# spec shape make_sft_data trains on (flat params). calculate: a from-scratch
# 182M model can't compute arithmetic in-weights (tokenizer splits numbers
# inconsistently). remember: the ChatGPT-bio-tool pattern -- she calls it when
# the user states a fact worth keeping, serve writes it to the MemoryStore,
# and render_context injects it back on every future relevant ask.
_CALC_TOOL = builtin_tool("calculate")
_REMEMBER_TOOL = builtin_tool("remember")
_SPEAK_TOOL = builtin_tool("speak")
_IMAGINE_TOOL = builtin_tool("imagine")
_FORGET_TOOL = builtin_tool("forget")
_BUILTIN_NAMES = BUILTIN_NAMES
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


# "Don't forget my birthday is in May" is a SAVE ask wearing the forget verb.
# Reading it as a forget ask suppressed the save and offered deletion instead
# -- the exact inversion of the intent -- so a negated forget disarms the
# forget gate and arms remember instead.
#
# The negation does NOT have to sit against the verb. Requiring adjacency read
# "Don't ever forget my birthday", "You must not forget the vet" and "We can't
# forget that mum visits" as DELETION asks -- the same inversion, one adverb
# further out.
#
# ANY negation earlier in the same sentence disarms, which is deliberately
# generous. A character budget was tried and fails in both directions at once:
# at 24 characters "Don't, and I really mean this, forget my anniversary" reads
# as a DELETE, while widening it makes "I can't remember much, forget my old
# address" read as a SAVE. The two errors are not equal -- reading a save as a
# delete destroys a fact, reading a delete as a save leaves one standing and
# the user simply asks again -- so the tie goes to the non-destructive reading
# and the sentence, not a byte count, is the boundary.
#
# Both gates read THIS spelling. They decide opposite things about the same
# family, and a cue only one of them recognizes is the worst outcome: the
# forget gate disarms, remember never matches, and the message arms nothing at
# all -- the fact silently unsaved.
_NEGATED_FORGET_SRC = (
    r"(do\s*n'?o?t|does\s*n'?o?t|did\s*n'?o?t|ca\s*n'?o?t|cannot|could\s*n'?o?t|"
    r"wo\s*n'?o?t|would\s*n'?o?t|should\s*n'?o?t|must\s*n'?o?t|never|no need to)"
    r"[^.?!]{0,80}?\s+forget\b"
)
_FORGET_NEGATED = re.compile(r"\b" + _NEGATED_FORGET_SRC, re.IGNORECASE)

# Talking ABOUT forgetting is not asking her to forget. "I forget where I put
# my keys" and "Did you forget my name?" armed the deletion tool, and the store
# then deleted on a single coincidental match -- the widened verb's premise is
# that a false OFFER is cheap, which only holds while the tool cannot destroy
# something on its own.
#
# Same safe direction as the negation: these SUPPRESS the offer, so a miss
# costs one un-offered tool and never a fact. First person and second person
# only -- an imperative "forget my address" has no subject and stays armed.
# The AUXILIARY is what separates the two: "did/do/does you forget" asks about
# the act of forgetting; "can/could/will/would you forget" is an imperative
# wearing a question mark. A bare "(you|they) forget" branch was tried and
# matched inside "can you forget my address", which disarmed a real delete ask
# -- and because remember is gated on `not forgettable`, "could you forget that
# I like tea" then offered to SAVE it. That is the save-instead-of-delete
# inversion the negation guard above exists to prevent, arriving through the
# suppressor instead. Statements like "you forget how cold it gets" are left
# armed: a false offer costs one declined tool, and the store's one-candidate
# rule is what actually stands between an offer and a deletion.
_FORGET_NOT_A_REQUEST = re.compile(
    r"\b("
    r"i (always |often |sometimes |usually |constantly |never |keep |kept )*forget|"
    r"i'?ve forgotten|i forgot|"
    r"(did|do|does|have|has) (you|he|she|they|we) forget"
    r")\b",
    re.IGNORECASE,
)


# remember is offered when the message states something save-worthy: an
# explicit remember ask, a first-person fact/preference, or a factual
# correction. Same rationale as the calculate gate -- an ever-present tool
# prompt degrades normal chat, so this stays intent-gated; she still decides
# whether to call. (At the v2 regen the gate retires for an always-offered
# built-in block -- ruled 2026-07-24 -- but the live v8 lineage keeps it.)
_MEMORABLE = re.compile(
    # The negated-forget family is a SAVE cue, spelled ONCE above.
    r"\b(remember|" + _NEGATED_FORGET_SRC + r"|note (that|this)|keep in mind|save (this|that)|"
    r"call me|my name('s| is)|"
    # "my <up to 3 words> is/are": covers "my dog's name is", "my favorite
    # season is" (two attribute words -- a single-\w+ pattern missed it,
    # measured 2026-07-06). Offering is cheap; she decides whether to call.
    r"my (\w+('s)? ){1,3}(is|are)\b|"
    r"i (like|love|hate|prefer|live|work|drive|play|own|always|never|usually)\b|"
    # "i have" minus its idioms. Excluding the whole "no" family took real
    # facts with it ("I have no siblings", "I have no allergies" are worth
    # saving) -- only the specific non-possession phrasings are out.
    r"i have (?!no (?:idea|clue)\b|to \w)\w+|"
    # first-person identity/state: "I'm a nurse", "I am from Denver", "I was
    # born in 1990" -- profession and origin were unreachable before.
    r"i'?m an? \w+|i am an? \w+|i'?m (allergic|from|married|working|called)|"
    r"i (was|am) (born|from|based)|"
    r"we (renamed|changed|moved|got|now)\b|"
    # factual corrections: a correction cue with a copula/naming nearby, so the
    # supersede path fires from natural chat ("Actually, my dog is Bruno now").
    # ACCEPTED FALSE POSITIVE: agreement smalltalk of the exact shape
    # "Actually, it is a great question." also arms remember. Two attempts to
    # exclude it by pattern each took real corrections with them ("Actually,
    # it's a Toyota, not a Honda."), and the trade is lopsided -- a false
    # positive costs one unnecessary tool offer, a false negative loses the
    # user's correction. Offering is cheap; she still decides whether to call.
    r"(actually|no,? it'?s|i meant|correction[:,]|that'?s not right)\b"
    r"[^.?!]{0,40}?\b(is|are|not|named|called|no longer)\b)",
    re.IGNORECASE,
)

# forget is offered when the user asks to drop a fact or says one no longer
# holds. It must SUPPRESS remember: "forget that I like tea" matches "i like"
# too, and offering remember there is the wrong direction (it would re-save the
# thing she was told to drop).
#
# The verb is matched BARE rather than through a list of following words. The
# list version ("forget (that|about|my|the|it|this)") silently dropped the very
# asks the store's one-candidate rule was built for -- "forget I'm tall",
# "forget everything about my dog", "forget where I live" -- so the store did
# the right thing and was never reached. A false OFFER costs one tool she may
# decline; a miss means no offer, no gradient, and a capability she can never
# learn to use.
_FORGETTABLE = re.compile(
    r"\b(forget\b|don'?t remember|stop remembering|"
    r"no longer (true|the case|like|have|live|work)|not true anymore|"
    r"scratch that|delete (that|this|the|my)|remove (that|this|the|my))",
    re.IGNORECASE,
)


def _looks_forgettable(text: str) -> bool:
    return (
        bool(text)
        and MEMORY is not None
        and not _FORGET_NEGATED.search(text)
        and not _FORGET_NOT_A_REQUEST.search(text)
        and bool(_FORGETTABLE.search(text))
    )


def _looks_memorable(text: str) -> bool:
    # A forget ask outranks a memorable shape -- offer forget, not remember.
    return (
        bool(text)
        and MEMORY is not None
        and not _looks_forgettable(text)
        and bool(_MEMORABLE.search(text))
    )


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


def _answering_a_forget_question(messages: list[Msg]) -> bool:
    """True when her last turn ASKED which memory to forget.

    The gate decides per request from the user's wording, so after she reported
    "2 memories match that -- say the one you mean word for word, or give its
    id", the natural answers ("#2", "the second one", quoting the memory back)
    armed nothing at all: only a reply that happened to contain the word
    "forget" reached the tool again. The question was unanswerable, which is
    not a question.

    No conversation state is needed -- the client sends the history, so the
    pending question is right there in the request."""
    for m in reversed(messages):
        if m.role == "assistant" and m.content:
            # The exact refusal rendering at a line start -- NOT a substring
            # test on the marker phrase. Her turn can QUOTE the phrase without
            # having asked anything (a surfaced "forgot: ..." success naming a
            # memory that contains it, or her own paraphrase mid-sentence),
            # and a substring check armed answering-mode on exactly those
            # (2026-07-25 fix-arc audit): the next ordinary user turn was then
            # read as naming a memory to delete.
            return _renders_forget_pending(m.content)
        if m.role == "user":
            continue
    return False


def _builtin_tools(user_text: str, client_mode: bool,
                   messages: list[Msg] | None = None) -> list[dict]:
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
    # ...or she just asked WHICH memory to forget, in which case whatever the
    # user says next is the answer to that question.
    if MEMORY is not None and (
        _looks_forgettable(user_text)
        or (messages is not None and _answering_a_forget_question(messages))
    ):
        tools.append(_FORGET_TOOL)
    if _looks_speakable(user_text):  # checks SPEAKER is enabled too
        tools.append(_SPEAK_TOOL)
    if _looks_imaginable(user_text):  # checks PAINTER is enabled too
        tools.append(_IMAGINE_TOOL)
    return tools


def _memory_id(raw) -> int | None:
    """A memory id from a tool argument, or None if it is not plainly one.

    `int()` is not a validator: it accepts other scripts' digits, underscore
    separators and surrounding space, so "3_0" quietly became 30 and deleted a
    different memory than the one meant."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    text = str(raw).strip().lstrip("#").strip()
    return int(text) if re.fullmatch(r"[0-9]{1,9}", text) else None


def _surfaced_forget_refusal(name: str, result: str) -> str | None:
    """The forget refusal the CLIENT must see verbatim, or None.

    The "N memories match that" refusal is a QUESTION: serve re-arms the
    forget tool on the next request by finding its marker in her last
    assistant turn -- which lives in the client's own history. Feeding the
    refusal to the model as a tool result was not enough: a 182M paraphrase
    loses the ids and exact wordings the user must answer with, and the
    marker never reached the client, so the handshake armed in tests and
    never once on the live path (round-7 audit, 2026-07-25). The refusal is
    surfaced verbatim in her visible reply, alongside whatever she says
    about it, on BOTH response paths (stream parity is byte-exact).

    Matched by its exact RENDERING, not by marker substring: three other
    forget results embed store or model text verbatim ("forgot: ...",
    "no matching memory to forget for: ...", "error: memory #N is not one
    of the ones that match ..."), so a memory that merely CONTAINS the
    marker phrase rode a SUCCESS line into her visible reply and armed the
    handshake with no question pending (2026-07-25 fix-arc audit)."""
    if name != "forget":
        return None
    body = result.removeprefix("error: ")
    return body if _renders_forget_pending(body) else None


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
        try:
            rec = MEMORY.remember(text, source="chat")
        except OSError as exc:
            return f"error: could not update the memory file ({exc})"
        return f"updated: {rec['text']}" if rec.get("superseded") else f"saved: {rec['text']}"
    if name == "forget":
        if MEMORY is None:
            return "error: memory disabled (start serve with --memory-dir)"
        # An id answers the ambiguity the store reports. It SELECTS among the
        # records the text already matched -- it is not a second way in. As its
        # own door it deleted whatever record happened to hold that id,
        # overriding a perfectly good text, and she has no honest source for an
        # id outside a refusal she just received, so every other id is invented.
        raw_id = arguments.get("id")
        mem_id = None
        if raw_id is not None and str(raw_id).strip() != "":
            mem_id = _memory_id(raw_id)
            if mem_id is None:
                return f"error: memory id must be a whole number, got {raw_id!r}"
        text = arguments.get("text")
        # Whitespace-NORMALIZED, not just stripped: this argument is echoed
        # verbatim into the "no matching memory to forget for: ..." result,
        # and a crafted multi-line argument FORGED the TooBroad rendering at
        # a line start there -- surfacing a fake "which memory?" question and
        # arming answering-mode (round-B audit, 2026-07-25). The OTHER echo,
        # "forgot: <record text>", is closed at the store: add(), remember()
        # AND the JSONL loader all normalize record text the same way
        # (round-C found the hand-edited-file route). Matching is unaffected.
        text = " ".join(text.split()) if isinstance(text, str) else ""
        if not text:
            return ("error: say which memory to forget in words"
                    if mem_id is None else
                    "error: give the wording too, not just an id -- an id on its own "
                    "could name any memory at all")
        try:
            removed = MEMORY.forget(text, only_id=mem_id)
        except MEMORY.TooBroad as exc:
            # Honest refusal beats a silent mass delete: memories have no .bak.
            return f"error: {exc}"
        except OSError as exc:
            # The store deletes in memory and then rewrites the file. A failed
            # rewrite used to escape as a 500 mid-conversation and leave memory
            # and disk disagreeing until the next boot.
            return f"error: could not update the memory file ({exc})"
        if not removed:
            if mem_id is not None:
                return (f"error: memory #{mem_id} is not one of the ones that match "
                        f"{text!r} -- name it in words instead")
            return f"no matching memory to forget for: {text}"
        return "forgot: " + "; ".join(r["text"] for r in removed)
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
        mem = MEMORY.render_context(
            _recent_user_text(req.messages), tokenizer, max_ids=128,
            k=MEMORY_RECALL, focus_query=_last_user_text(req.messages),
        )
        if mem:
            extra.append(mem)
    # Built-ins are gated on intent (see _builtin_tools); client tools are
    # always honored.
    client_tools = list(req.tools or [])
    all_tools = _builtin_tools(
        _last_user_text(req.messages), bool(client_tools), req.messages
    ) + client_tools
    if all_tools:
        tools_block = render_tools_system(all_tools)
        if not (msgs and msgs[0].get("role") == "system"):
            # Training's tool examples ALWAYS lead with this exact preamble
            # (make_sft_data._system, single \n before "Available tools:");
            # a system message that OPENS with "Available tools:" is a shape
            # the model never saw.
            tools_block = PERSONA.tools_preamble + "\n" + tools_block
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

    def _loop_on_search(out: dict, parsed: list[dict], hop: int) -> bool:
        # A search hop runs only when the turn is PURELY a search: a turn
        # that also names tool calls takes the tool path above (mixed intents
        # follow the older, surfaced route rather than guessing an order).
        # It runs even with the organ absent -- the error text below is how
        # a search-trained model serving without --search learns, in-context,
        # that the ability is off, instead of the span silently vanishing.
        return bool(out.get("search")) and not parsed and hop < _MAX_TOOL_HOPS

    def _apply_search(cur_msgs: list[dict], out: dict, results: list | None = None) -> list[dict]:
        # Append the assistant turn that asked (the literal span -- assistant
        # content keeps native tag ids, so history re-renders exactly as
        # generated) and the lookup's answer as a tool turn: the same trace
        # shape the SFT data teaches (gen_search_examples).
        q = out["search"]
        if SEARCHER is None:
            result = "error: search disabled (start serve with --search)"
        else:
            try:
                result = render_results(q, SEARCHER.query(q, k=ARGS.search_k))
            except SearchError as exc:
                result = f"error: {exc}"
        if results is not None:
            results.append(("search", result))
        # Any content she spoke before the span stays in the history turn,
        # matching _apply_builtins keeping out["content"] beside its calls.
        return cur_msgs + [
            {"role": "assistant", "content": f"{out.get('content') or ''}<search>{q}</search>"},
            {"role": "tool", "content": result},
        ]

    if req.stream:

        def _events_body():
            yield _sse_role_open(cid, created)
            # Span ids from the ATTACHED tokenizer (v1: identical to the
            # module constants; bigger vocab: derived rows -- HIGH-2).
            _ct = chat_token_ids(tokenizer)
            THINK, THINK_END = think_token_ids(tokenizer)
            TOOL_CALL, TOOL_CALL_END = _ct["<|tool_call|>"], _ct["<|/tool_call|>"]
            # None on vocabs without the tags -- the comparisons below skip.
            SEARCH, SEARCH_END = search_token_ids(tokenizer)
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
                    if tid in (THINK, TOOL_CALL) or (SEARCH is not None and tid == SEARCH):
                        depth += 1
                        continue
                    if tid in (THINK_END, TOOL_CALL_END) or (SEARCH_END is not None and tid == SEARCH_END):
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
                # never sees it. Anything else is surfaced whole. Exception:
                # a forget refusal that asks WHICH memory is surfaced as
                # content, or the handshake's marker (and the ids the user
                # must answer with) dies in the server-side tool trace.
                if _loop_on_search(out, parsed, hop):
                    # The query never streams (the span is depth-suppressed
                    # above); the next hop answers from the spliced results.
                    cur_msgs = _apply_search(cur_msgs, out)
                    continue
                if _loop_on_builtins(parsed, hop):
                    tool_results: list = []
                    cur_msgs = _apply_builtins(cur_msgs, out, parsed, tool_results)
                    for _name, _result in tool_results:
                        surfaced = _surfaced_forget_refusal(_name, _result)
                        if not surfaced:
                            continue
                        # Same separator semantics as the hop-content deltas
                        # above, so stream and non-stream stay byte-identical
                        # (non-stream appends this text to its "\n" join).
                        body = ("\n" + surfaced) if emitted_any else surfaced
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
    # Built-ins actually EXECUTED, in order. A looped built-in is consumed by
    # the hop that runs it, so the surfaced message carries no `tool_calls` for
    # it -- which left every server-side action unobservable from outside. An
    # eval could not tell a `speak` that fired from one that never happened,
    # and a restraint probe expecting NO call passed even when she called one.
    tools_run: list[str] = []
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
        if _loop_on_search(out, parsed, hop):
            if out.get("content"):
                hop_texts.append(out["content"])
            search_results: list = []
            cur_msgs = _apply_search(cur_msgs, out, search_results)
            tools_run += [name for name, _ in search_results]
            continue
        if _loop_on_builtins(parsed, hop):
            if out.get("content"):
                hop_texts.append(out["content"])
            tool_results: list = []
            cur_msgs = _apply_builtins(cur_msgs, out, parsed, tool_results)
            tools_run += [name for name, _ in tool_results]
            # Flag on the EXECUTED result, not the intent -- "error: nothing
            # to say" must not silence the page's own TTS (2026-07-17 audit).
            if any(name == "speak" and result == "speaking" for name, result in tool_results):
                spoke_server_side = True
            # A forget refusal that asks WHICH memory joins the visible
            # content (hop_texts feeds the final "\n" join), or its marker
            # and ids never reach the client and the handshake stays dead.
            hop_texts += [
                s for s in (_surfaced_forget_refusal(n, r) for n, r in tool_results) if s
            ]
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
    if spoke_server_side or tools_run:
        # Non-standard extension. `spoke` is what the chat page reads so it
        # never double-voices a reply the speak tool already said on the
        # server's speakers; `tools_run` is the execution trace that makes a
        # server-side action observable at all.
        resp["enigma"] = {"spoke": spoke_server_side, "tools_run": tools_run}
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
  #mute, #talk, #stop { background:var(--accent); color:#08121c; border:0; border-radius:8px;
          padding:8px 15px; font-size:14px; font-weight:700; cursor:pointer; }
  #mute.muted { background:var(--warn); }
  #talk.on { background:#6fca6f; }
  #stop { background:var(--warn); }
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
  #mic, #pic { background:#16222e; color:var(--text); border:1px solid #24384a;
               border-radius:8px; padding:0 16px; font-size:15px; cursor:pointer; }
  #mic.rec { background:#7a2020; border-color:#a83232; }
  #pic.on { background:#1f4a3a; border-color:#2f7a5f; }
  img.shot { display:block; max-width:min(420px, 100%); border-radius:10px;
             margin:6px 0 10px; border:1px solid #24384a; }
  button:disabled { opacity:.5; cursor:default; }
</style></head><body>
<header>
  <h1>Enigma</h1>
  <span id="voice-state">voice: checking...</span>
  <button id="talk" type="button" title="Speak every reply out loud">Talk: off</button>
  <button id="stop" type="button" title="Stop talking (Esc)">Stop</button>
  <button id="mute" type="button">Mute</button>
</header>
<div id="log"></div>
<form id="f"><button id="mic" type="button" title="Hold to talk (needs --ears)" hidden>Mic</button>
<button id="pic" type="button" title="Show her a picture (needs --eyes)" hidden>Img</button>
<input id="file" type="file" accept="image/*" hidden>
<input id="box" autocomplete="off" placeholder="Say something to her..." autofocus>
<button id="send" type="submit">Send</button></form>
<script>
"use strict";
var history_ = [];
var muted = false;       // the SERVER owns mute; syncStatus adopts within 3s
var talkMode = false;    // conversation mode: speak every reply (server-owned)
var voiceReady = false;
var currentAudio = null;
var currentUrl = null;   // blob URL of the playing reply, revoked in stopAudio
var muteEpoch = 0;       // clicks invalidate in-flight polls (no stale revert)
var talkEpoch = 0;
var stopGen = null;      // server's stop counter; a change means "hush now"
var log = document.getElementById("log");
var box = document.getElementById("box");
var send = document.getElementById("send");
var muteBtn = document.getElementById("mute");
var talkBtn = document.getElementById("talk");
var stopBtn = document.getElementById("stop");
var voiceState = document.getElementById("voice-state");

function add(cls, text) {
  var d = document.createElement("div");
  d.className = "msg " + cls;
  d.textContent = text;
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
  return d;
}
// "image saved to <dir>/imagine_ab12cd34.png" -- the imagine tool answers with
// a filesystem path, so a picture she made could only be read as a sentence.
// Only the bare NAME is taken, and it is put in an img src, never in the DOM as
// markup: the reply text itself is always rendered with textContent.
var IMG_NAME = /image saved to\\s+\\S*?([A-Za-z0-9_-]+\\.png)/i;
function addImage(text) {
  var m = IMG_NAME.exec(text);
  if (!m) return;
  var img = document.createElement("img");
  img.className = "shot";
  img.alt = "image she generated";
  img.src = "/v1/images/file/" + encodeURIComponent(m[1]);
  img.onerror = function () { img.remove(); };   // organ off or file gone: no broken icon
  log.appendChild(img);
  log.scrollTop = log.scrollHeight;
}
var speakSeq = 0;        // the newest speak() call owns playback; a hush outbids them all
function clearClip() {   // cleanup only -- NEVER a cancel signal (a clip ending naturally
  if (currentAudio) { currentAudio.pause(); currentAudio = null; }   // must not silence a
  if (currentUrl) { URL.revokeObjectURL(currentUrl); currentUrl = null; }  // pending reply)
}
function stopAudio() {   // hush: cancel every reply still synthesizing, then clean up
  speakSeq += 1;
  clearClip();
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
function paintTalk() {
  talkBtn.textContent = talkMode ? "Talk: on" : "Talk: off";
  talkBtn.className = talkMode ? "on" : "";
}
function pushTalk() {
  fetch("/v1/audio/talk-mode", { method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled: talkMode }) }).catch(function () {});
}
function stopTalking() {
  stopAudio();                                                     // hush this window now
  fetch("/v1/audio/stop", { method: "POST" }).catch(function () {});  // and the server's speakers
}
function syncStatus() {
  var me = muteEpoch, te = talkEpoch;
  fetch("/v1/audio/status")
    .then(function (r) { if (!r.ok) throw new Error(); return r.json(); })
    .then(function (s) {
      if (me === muteEpoch && s.muted !== muted) {  // no click since this poll left
        muted = s.muted;
        if (muted) stopAudio();
        paintMute();
      }
      if (te === talkEpoch && s.talk_mode !== talkMode) {
        talkMode = s.talk_mode;
        if (!talkMode) stopAudio();  // tray turned narration off -- hush here too
        paintTalk();
      }
      if (stopGen === null) stopGen = s.stop_gen;  // first poll: adopt baseline, do not hush
      else if (s.stop_gen !== stopGen) { stopGen = s.stop_gen; stopAudio(); }  // a Stop fired elsewhere
    }).catch(function () {});
}
muteBtn.onclick = function () {
  muted = !muted;
  muteEpoch += 1;
  if (muted) stopAudio();
  paintMute();
  pushMute();
};
talkBtn.onclick = function () {
  talkMode = !talkMode;
  talkEpoch += 1;
  if (!talkMode) stopAudio();  // narration off means NOW -- cancel the reply
  paintTalk();                 // playing or still synthesizing (mute parity)
  pushTalk();
};
stopBtn.onclick = stopTalking;
document.addEventListener("keydown", function (ev) {
  if (ev.key === "Escape") stopTalking();
});
function speak(text) {
  if (!voiceReady || muted || !text) return;
  var seq = ++speakSeq;   // this speak is now the newest; a hush OR a newer reply outbids it
  fetch("/v1/audio/speech", { method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input: text }) })
    .then(function (r) {
      if (r.status === 204) return null;
      if (!r.ok) throw new Error();
      return r.blob(); })
    .then(function (b) {
      if (!b || muted || seq !== speakSeq) return;  // hushed or superseded while synthesizing
      clearClip();
      currentUrl = URL.createObjectURL(b);
      var a = new Audio(currentUrl);
      currentAudio = a;
      a.addEventListener("ended", function () { if (currentAudio === a) clearClip(); });
      a.play().catch(function () {});
    }).catch(function () {});
}
// A picture waiting to go with the next message, as a data: URL. The server
// captions image content into "[image: ...]" text before anything else sees
// it, so eyes were reachable from an API client and NOT from her own window --
// she could not be shown anything by the person using her.
var pendingImage = null;
var picBtn = document.getElementById("pic");
var fileInput = document.getElementById("file");
picBtn.addEventListener("click", function () { fileInput.click(); });
fileInput.addEventListener("change", function () {
  var f = fileInput.files && fileInput.files[0];
  fileInput.value = "";                       // re-picking the same file re-fires
  if (!f) return;
  if (f.size > 8 * 1024 * 1024) { add("sys", "that image is too large (8 MB max)"); return; }
  var fr = new FileReader();
  fr.onload = function () {
    pendingImage = String(fr.result);
    picBtn.className = "on";
    picBtn.textContent = "Img*";
    box.placeholder = "Describe or ask about the picture...";
  };
  fr.onerror = function () { add("sys", "could not read that image"); };
  fr.readAsDataURL(f);                        // data: URL -- the only form serve accepts
});
function clearPending() {
  pendingImage = null;
  picBtn.className = "";
  picBtn.textContent = "Img";
  box.placeholder = "Say something to her...";
}
document.getElementById("f").onsubmit = function (ev) {
  ev.preventDefault();
  var text = box.value.trim();
  if (!text || send.disabled) return;
  box.value = "";
  add("me", text);
  if (pendingImage) {
    var shown = document.createElement("img");
    shown.className = "shot";
    shown.alt = "picture you showed her";
    shown.src = pendingImage;
    log.appendChild(shown);
    history_.push({ role: "user", content: [
      { type: "text", text: text },
      { type: "image_url", image_url: { url: pendingImage } },
    ] });
    clearPending();
  } else {
    history_.push({ role: "user", content: text });
  }
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
      addImage(reply);
      history_.push({ role: "assistant", content: reply });
      if (talkMode && !(data.enigma && data.enigma.spoke)) speak(reply);
    })
    .catch(function (e) {
      thinking.remove();
      add("sys", "error: " + e.message);
    })
    .then(function () { send.disabled = false; box.focus(); });
};
// Mic: hold to record, release to transcribe into the box. Shown only when the
// server actually booted --ears; an always-visible control that 404s teaches
// the user the feature is broken rather than absent.
var micBtn = document.getElementById("mic");
var recorder = null, chunks = [], micBusy = false;
function micLabel(t) {
  micBtn.textContent = t;
  micBtn.className = (t === "Rec") ? "rec" : "";
}
function stopRecording() {
  if (recorder && recorder.state === "recording") recorder.stop();
}
function startRecording() {
  if (micBusy || recorder) return;
  if (!navigator.mediaDevices || !window.MediaRecorder) {
    add("sys", "this browser cannot record audio");
    return;
  }
  navigator.mediaDevices.getUserMedia({ audio: true }).then(function (stream) {
    chunks = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = function (ev) { if (ev.data.size) chunks.push(ev.data); };
    recorder.onstop = function () {
      stream.getTracks().forEach(function (t) { t.stop(); });   // release the device
      recorder = null;
      micLabel("Mic");
      if (!chunks.length) return;
      micBusy = true;
      micLabel("...");
      var fd = new FormData();
      fd.append("file", new Blob(chunks, { type: "audio/webm" }), "clip.webm");
      fetch("/v1/audio/transcriptions", { method: "POST", body: fd })
        .then(function (r) { if (!r.ok) throw new Error("transcription failed"); return r.json(); })
        .then(function (d) {
          var said = (d && typeof d.text === "string") ? d.text.trim() : "";
          if (said) { box.value = box.value ? box.value + " " + said : said; box.focus(); }
          else add("sys", "[heard nothing]");
        })
        .catch(function (e) { add("sys", "error: " + e.message); })
        .then(function () { micBusy = false; micLabel("Mic"); });
    };
    recorder.start();
    micLabel("Rec");
  }).catch(function () { add("sys", "microphone unavailable (permission denied?)"); });
}
micBtn.addEventListener("mousedown", startRecording);
micBtn.addEventListener("mouseup", stopRecording);
micBtn.addEventListener("mouseleave", stopRecording);
micBtn.addEventListener("touchstart", function (ev) { ev.preventDefault(); startRecording(); });
micBtn.addEventListener("touchend", function (ev) { ev.preventDefault(); stopRecording(); });

fetch("/v1/capabilities")
  .then(function (r) { if (!r.ok) throw new Error(); return r.json(); })
  .then(function (c) {
    if (c && c.ears) micBtn.hidden = false;
    if (c && c.eyes) picBtn.hidden = false;
  })
  .catch(function () {});

fetch("/v1/audio/voices")
  .then(function (r) { if (!r.ok) throw new Error(); voiceReady = true; })
  .catch(function () { voiceReady = false; })
  .then(function () {
    voiceState.textContent = voiceReady
      ? (muted ? "voice: muted" : "voice: on")
      : "voice: off (start with --voice)";
    if (voiceReady) syncStatus();
  });
paintMute();
paintTalk();
setInterval(syncStatus, 3000);
</script></body></html>
"""


@app.get("/", response_class=HTMLResponse)
def chat_page():
    return _CHAT_PAGE


@app.get("/v1/models")
def list_models():
    # The static id string cannot distinguish two same-arch checkpoints; the
    # checkpoint block is what lets an eval transcript prove WHICH weights it
    # measured. Absent only when tests set the model globals directly.
    entry: dict = {"id": MODEL_ID, "object": "model", "owned_by": "enigma"}
    if MODEL_SHA256:
        entry["checkpoint"] = {"path": MODEL_PATH, "sha256": MODEL_SHA256, "step": STEP}
    return {"object": "list", "data": [entry]}


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
        mem = MEMORY.render_context(
            _recent_user_text(messages), tokenizer, max_ids=128,
            k=MEMORY_RECALL, focus_query=_last_user_text(messages),
        )
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
            yield _sse_role_open(cid, created)
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
        # remember(), not add(): the HTTP door gets the same dedup/supersede/
        # date semantics as the tool door, so posting a fact twice does not
        # mint a second record competing for the same retrieval slots.
        return {"ok": True, "memory": MEMORY.remember(req.text, kind=req.kind)}
    except ValueError as exc:
        # client-input error (empty/whitespace text), not a server crash
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v1/memory")
def memory_list(q: str | None = None, k: int = 5):
    if MEMORY is None:
        raise _organ_off("memory disabled -- start with --memory-dir")
    if k < 0:
        raise HTTPException(status_code=400, detail="k must be 0 or more")
    # The tail slice reads as "everything but the oldest k" on a k of 0, so
    # the no-query door answers an empty ask explicitly instead of slicing.
    recs = MEMORY.search(q, k=k) if q else (MEMORY.all()[-k:] if k else [])
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
    voice: str | None = None  # the active voice is server state; reject per-request picks honestly


@app.post("/v1/audio/speech")
def audio_speech(req: SpeechReq):
    if SPEAKER is None:
        raise _organ_off("voice disabled -- start with --voice")
    if req.voice is not None:
        raise HTTPException(
            status_code=400,
            detail="per-request voice selection is not supported; set the active voice with "
            "POST /v1/audio/voice, and list the choices at GET /v1/audio/voices",
        )
    text = (req.input or "").strip()
    if not text:  # a client input error is a 400, not a server 500
        raise HTTPException(status_code=400, detail="nothing to say -- 'input' is empty")
    if MUTED:
        return Response(status_code=204)
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        SPEAKER.save_wav(text, tmp)
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
    _write_state_atomic(_MUTE_STATE, {"muted": MUTED})
    return {"muted": MUTED}


@app.post("/v1/audio/stop")
def audio_stop():
    """Silence her NOW: abort the utterance playing on the server's speakers
    and cancel anything queued. Bumping the stop generation also tells an open
    chat window (which polls it) to hush its own browser audio. Never 503s --
    a desktop/tray Stop must be safe even when the voice organ is off."""
    global _STOP_GEN
    _STOP_GEN += 1
    stopped = False
    if SPEAKER is not None:
        SPEAKER.stop()
        stopped = True
    return {"stopped": stopped, "stop_gen": _STOP_GEN}


class TalkReq(BaseModel):
    enabled: bool


@app.get("/v1/audio/talk-mode")
def get_talk_mode():
    return {"enabled": TALK_MODE}


@app.post("/v1/audio/talk-mode")
def set_talk_mode(req: TalkReq):
    """Toggle conversation mode: when ON, the window speaks every reply out
    loud. Server-owned + persisted so it survives a restart and any open
    window adopts it within the poll interval."""
    global TALK_MODE
    TALK_MODE = bool(req.enabled)
    _write_state_atomic(_TALK_STATE, {"enabled": TALK_MODE})
    return {"enabled": TALK_MODE}


@app.get("/v1/audio/status")
def audio_status():
    """One poll the chat page reads every few seconds: the mute + talk-mode
    truth to adopt, the stop generation to hush on when it changes, and the
    voice health so a broken audio device (play jobs failing in the worker with
    only a console WARN) is visible to the page/tray instead of silent."""
    if SPEAKER is None:
        voice = "off"
    elif SPEAKER.last_error is not None:
        voice = "error"
    else:
        voice = "ok"
    status = {"muted": MUTED, "talk_mode": TALK_MODE, "stop_gen": _STOP_GEN, "voice": voice}
    if voice == "error":
        status["voice_error"] = str(SPEAKER.last_error)[:200]
    return status


@app.get("/v1/images/file/{name}")
def image_file(name: str):
    """Serve ONE generated PNG out of the images dir, by bare filename.

    The imagine tool answers with a filesystem path, which the chat page could
    only show as literal text -- she could make a picture nobody could see
    without opening a file manager. Rendering it needs a way to fetch it.

    Bare name only, matched against a strict pattern and then re-checked
    against the resolved directory: a name is never joined to a path it could
    escape, so `..`, absolute paths, alternate separators and symlinked
    lookalikes all fail before any read happens."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}\.png", name):
        raise HTTPException(status_code=404, detail="no such image")
    target = (IMAGES_DIR / name).resolve()
    try:
        root = IMAGES_DIR.resolve()
    except OSError:
        raise HTTPException(status_code=404, detail="no such image") from None
    if target.parent != root or not target.is_file():
        raise HTTPException(status_code=404, detail="no such image")
    return FileResponse(str(target), media_type="image/png")


@app.get("/v1/capabilities")
def capabilities():
    """Which organs this server actually booted with.

    Organs are flag-gated, so an ability can be absent for two very different
    reasons: she cannot do it, or it was never started. Nothing exposed that
    difference, which left two consumers guessing -- an eval scored a silent
    0/N for a tool the server never offered, and the chat page had to decide
    whether to show a mic or an image control with no way to ask.

    Read-only and cheap: no organ is touched, only whether one was built."""
    return {
        "memory": MEMORY is not None,
        "voice": SPEAKER is not None,
        "ears": EARS is not None,
        "eyes": EYES is not None,
        "image_gen": PAINTER is not None,
        "search": SEARCHER is not None,
        "instruct": INSTRUCT,
        # The serving context budget. An eval transcript that omits this
        # cannot prove two runs shared conditions (audit 2026-08-08: the v2
        # gate run served at 2048 against a baseline presumed 1024, and the
        # transcripts could not say). ARGS is None until boot() parses argv --
        # the bare-import test harness reaches this endpoint with no ARGS.
        "max_context": ARGS.max_context if ARGS is not None else None,
        # The built-ins that can actually run right now. The model is offered a
        # subset of these per request, by intent.
        "builtins": sorted(
            n for n in _BUILTIN_NAMES
            if not (n in ("remember", "forget") and MEMORY is None)
            and not (n == "speak" and SPEAKER is None)
            and not (n == "imagine" and PAINTER is None)
        ),
    }


class VoiceReq(BaseModel):
    """Any subset of the recipe; omitted fields keep their current value."""

    blend: list | None = None
    speed: float | None = None
    lang_code: str | None = None


@app.get("/v1/audio/voice")
def get_voice():
    if SPEAKER is None:
        raise _organ_off("voice disabled -- start with --voice")
    return SPEAKER.get_voice()


@app.post("/v1/audio/voice")
def set_voice(req: VoiceReq):
    """Retune her voice at runtime (blend, speed, or language) and persist it.
    An invalid recipe is a 400, not a 500 -- the current voice is unchanged."""
    if SPEAKER is None:
        raise _organ_off("voice disabled -- start with --voice")
    try:
        return SPEAKER.set_voice(blend=req.blend, speed=req.speed, lang_code=req.lang_code)
    except TTSError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class OutputReq(BaseModel):
    """Which speaker she plays out of; null (or "default") follows the system."""

    device: str | int | None = None


@app.get("/v1/audio/outputs")
def audio_outputs():
    """The output endpoints on the SERVER machine, plus the one in use.

    This governs server-side playback only -- the speak built-in and talk mode.
    Audio a browser fetches from /v1/audio/speech is routed by that browser's
    own OS, so a remote client picks its speaker in its own sound settings.
    """
    if SPEAKER is None:
        raise _organ_off("voice disabled -- start with --voice")
    try:
        outputs = SPEAKER.output_devices
    except TTSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"outputs": outputs, "current": SPEAKER.get_output_device()}


@app.post("/v1/audio/output")
def set_audio_output(req: OutputReq):
    """Point her at a speaker by name or index; null means the system default.
    Persisted, and live on her next utterance. A device that does not exist is
    a 400 naming the ones that do -- the current routing is left alone."""
    if SPEAKER is None:
        raise _organ_off("voice disabled -- start with --voice")
    try:
        return SPEAKER.set_output_device(req.device)
    except TTSError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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


def print_audio_outputs() -> None:
    """Print the output devices --voice-device accepts, marking the default."""
    try:
        outputs = list_output_devices()
    except TTSError as exc:
        print(f"cannot list audio outputs: {exc}", flush=True)
        return
    if not outputs:
        print("no audio output devices found on this machine", flush=True)
        return
    print("Output devices -- pass the name or the index to --voice-device:", flush=True)
    for d in outputs:
        mark = "   <== system default" if d["default"] else ""
        print(f"  [{d['index']:2}] {d['name']}  ({d['channels']} ch){mark}", flush=True)
    print("  --voice-device default follows whatever Windows calls the default", flush=True)


def main() -> None:
    """Run the server. Console-script entry point (pyproject [project.scripts])
    and the __main__ path share this."""
    # Listing speakers must not cost a checkpoint load, so this is answered
    # before boot() -- the flag is a question about the machine, not about her.
    _early, _ = _p.parse_known_args()
    if _early.list_audio_outputs:
        print_audio_outputs()
        return
    boot()
    print(f"Enigma OpenAI-compatible API -> http://{ARGS.host}:{ARGS.port}/v1", flush=True)
    print(f"In Odysseus:  /setup local http://{ARGS.host}:{ARGS.port}/v1", flush=True)
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="warning")


if __name__ == "__main__":
    main()
