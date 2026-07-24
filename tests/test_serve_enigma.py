"""serve_enigma.py behavior locks (test-suite audit 2026-07-17).

The serve module was a zero-coverage island: argparse + torch.load ran at
import time, so no test could touch the tool loop, the sampling defaults, the
train/serve system-prompt contract, stream parity, or the mute switch. boot()
now owns startup and importing the module is free -- these tests are the
coverage that refactor exists to enable.

Heavyweight pieces (the 182M model, organs) are NOT loaded here: generation
is monkeypatched at the _gen_ids/_stream_ids_locked seam, everything else
(tokenizer, chat rendering, memory store, FastAPI handlers) is real.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
import torch

import make_sft_data
import serve_enigma as serve
from enigma_engine.core.chat_format import CHAT_FORMAT_NAME, attach_chat_tokens, render_tools_system
from enigma_engine.core.eyes import EyesError
from enigma_engine.core.memory_store import MemoryStore
from enigma_engine.core.model import Enigma
from enigma_engine.core.model_presets import ForgeConfig
from enigma_engine.core.tokenizer import get_tokenizer

# ---------------------------------------------------------------------------
# import safety -- the regression gate for the whole refactor
# ---------------------------------------------------------------------------


def test_import_ran_no_startup():
    """Importing serve_enigma must not parse argv, load a checkpoint, or
    construct organs. (This ran at import time until 2026-07-17; a bare
    import needed a real model file.) Runs first in this module: boot() is
    only exercised by the last test, which restores every global."""
    assert serve.ARGS is None
    assert serve.model is None
    assert serve.tokenizer is None
    assert serve.MEMORY is None and serve.SPEAKER is None
    assert serve.EARS is None and serve.EYES is None and serve.PAINTER is None
    assert serve._BOOTED is False


def test_unbooted_app_refuses_with_503(monkeypatch):
    """`uvicorn serve_enigma:app` without boot() WORKED before the refactor
    (startup ran at import); afterwards it served a deceptive half-up app --
    health endpoints 200 while every generation request 500'd with an opaque
    NoneType error (re-audit 2026-07-18). The middleware must refuse
    honestly, gate on boot COMPLETION (a boot that dies after assigning
    `model` is still unready -- round-2 re-audit), and pass requests through
    untouched once booted."""
    # the middleware must actually be REGISTERED -- deleting the decorator
    # kept every test green until this pin (round-2 re-audit)
    assert any(
        getattr(m, "kwargs", {}).get("dispatch") is serve._require_boot
        for m in serve.app.user_middleware
    ), "boot-guard middleware is not registered on the app"

    monkeypatch.setattr(serve, "_BOOTED", False)
    monkeypatch.setattr(serve, "model", object())  # half-booted: model set...
    reached = []

    async def call_next(request):
        reached.append(request)
        return "downstream-response"

    resp = asyncio.run(serve._require_boot(object(), call_next))
    assert resp.status_code == 503  # ...but completion is what counts
    assert b"not booted" in resp.body
    assert not reached  # the request never hit a handler

    monkeypatch.setattr(serve, "_BOOTED", True)
    assert asyncio.run(serve._require_boot(object(), call_next)) == "downstream-response"
    assert len(reached) == 1


@pytest.fixture(scope="module")
def tok():
    t = get_tokenizer("bpe")
    attach_chat_tokens(t)
    return t


# ---------------------------------------------------------------------------
# API sampling defaults (ultrareview #9 scope: these are LIVE serving knobs)
# ---------------------------------------------------------------------------


def test_stop_ids_derive_from_the_attached_tokenizer(monkeypatch, tok):
    """Generation stop ids must come from the ATTACHED tokenizer.

    They were module constants (IM_END=4719), which alias a real learned
    token on any vocab bigger than 4718 -- generation would then either
    never stop or stop on a content token (chat_format HIGH-2). An earlier
    version of this fix rebound a module GLOBAL inside boot(); that leaked
    across boots because serve's _RUNTIME_GLOBALS snapshot cannot restore a
    name only boot() writes, so the value is derived per call instead.
    """
    from enigma_engine.core.chat_format import IM_END, chat_token_ids

    monkeypatch.setattr(serve, "tokenizer", tok)
    monkeypatch.setattr(serve, "EOS_ID", 2)
    assert serve._stop_ids() == (2, IM_END)  # live v1 vocab: derived == constant

    class _BigVocabTok:
        """A vocab whose chat rows sit far past the v1 constants."""

        token_to_id = {f"<t{i}>": i for i in range(9000)}
        id_to_token = {i: f"<t{i}>" for i in range(9000)}
        special_tokens = {}

    big = _BigVocabTok()
    attach_chat_tokens(big)
    monkeypatch.setattr(serve, "tokenizer", big)
    eos, im_end = serve._stop_ids()
    assert im_end == chat_token_ids(big)["<|im_end|>"] == 9001
    assert im_end != IM_END, "stop id fell back to the hardcoded v1 constant"


def test_chat_defaults_are_the_clarity_settings():
    req = serve.ChatReq(messages=[])
    assert req.temperature == 0.3
    assert req.min_p == 0.05
    assert req.top_p == 0.9
    assert req.top_k == 50
    assert req.repetition_penalty == 1.1
    assert req.max_tokens == 256


def test_completion_defaults_keep_the_exploratory_temperature():
    req = serve.CompletionReq(prompt="x")
    assert req.temperature == 0.8
    assert req.min_p == 0.0


# ---------------------------------------------------------------------------
# intent gates + built-in tool offering (tool-stealing history, 2026-07-06)
# ---------------------------------------------------------------------------


def test_arithmetic_gate():
    assert serve._looks_arithmetic("What is 372 + 519?")
    assert serve._looks_arithmetic("what's 15 percent of 80")
    assert not serve._looks_arithmetic("Tell me about the seven seas.")
    assert not serve._looks_arithmetic("sum up your day for me")  # no digits


def test_memorable_gate_requires_memory_enabled(monkeypatch):
    text = "Remember that my dog is named Rex."
    monkeypatch.setattr(serve, "MEMORY", None)
    assert not serve._looks_memorable(text)
    monkeypatch.setattr(serve, "MEMORY", object())
    assert serve._looks_memorable(text)
    assert serve._looks_memorable("my favorite season is autumn")
    assert not serve._looks_memorable("What's the weather like?")


def test_memorable_gate_catches_preferences_facts_and_corrections(monkeypatch):
    monkeypatch.setattr(serve, "MEMORY", object())
    mem = serve._looks_memorable
    # preferences and first-person facts that were unreachable before
    assert mem("I'm a nurse.")                       # profession
    assert mem("I have two cats.")                   # possession
    assert mem("I was born in 1990.")                # origin fact
    assert mem("I prefer tea over coffee.")
    # on-the-spot factual corrections must arm remember (-> supersede path)
    assert mem("Actually, my dog is named Bruno now.")
    assert mem("No, it's Samantha, not Sam.")
    assert mem("We renamed the project to Orion.")
    # conversational cues that are NOT facts must stay quiet
    assert not mem("Actually, that's a great question.")
    assert not mem("No, thanks.")
    assert not mem("What's the weather like?")


def test_forget_gate_suppresses_remember(monkeypatch):
    monkeypatch.setattr(serve, "MEMORY", object())
    # "forget that I like tea" matches the memorable "i like" shape too --
    # forget must win, or she re-saves the thing she was told to drop.
    assert serve._looks_forgettable("Forget that I like tea.")
    assert not serve._looks_memorable("Forget that I like tea.")
    assert serve._looks_forgettable("I no longer live in Denver.")
    assert serve._looks_forgettable("Scratch that.")
    assert not serve._looks_forgettable("I like tea.")  # a plain fact is not a forget


def test_recent_user_text_spans_the_thread():
    Msg = serve.Msg
    msgs = [
        Msg(role="user", content="My dog is named Rex."),
        Msg(role="assistant", content="Got it."),
        Msg(role="user", content="And what did I say he was called?"),
    ]
    # last-message-only recall would drop "Rex" from the query on the follow-up
    q = serve._recent_user_text(msgs)
    assert "Rex" in q and "called" in q
    assert serve._last_user_text(msgs) == "And what did I say he was called?"


def test_speak_and_imagine_gates(monkeypatch):
    monkeypatch.setattr(serve, "SPEAKER", object())
    monkeypatch.setattr(serve, "PAINTER", object())
    assert serve._looks_speakable("Say hello out loud.")
    assert not serve._looks_speakable("What did he say to you?")
    assert serve._looks_imaginable("Draw me a picture of a lighthouse at sunset.")
    assert not serve._looks_imaginable("The picture was hanging crooked.")


def test_builtin_tools_offering(monkeypatch):
    monkeypatch.setattr(serve, "MEMORY", None)
    monkeypatch.setattr(serve, "SPEAKER", None)
    monkeypatch.setattr(serve, "PAINTER", None)
    # calculate rides along whenever client tools exist (client_mode=True)
    assert serve._builtin_tools("Check the weather in Toronto", True) == [serve._CALC_TOOL]
    # ...but intent-gated otherwise
    assert serve._builtin_tools("Check the weather in Toronto", False) == []
    assert serve._builtin_tools("What is 7 * 8?", False) == [serve._CALC_TOOL]
    # remember is intent-gated ALWAYS (it stole tool calls when ever-present)
    monkeypatch.setattr(serve, "MEMORY", object())
    offered = serve._builtin_tools("Check the weather in Toronto", True)
    assert serve._REMEMBER_TOOL not in offered


# ---------------------------------------------------------------------------
# _generate_text: the MIN_GEN_TOKENS reserve arithmetic (prompt is never
# squeezed below max_context - reserve; generation takes what's left)
# ---------------------------------------------------------------------------


def test_generate_text_reserves_prompt_budget(monkeypatch, tok):
    monkeypatch.setattr(serve, "tokenizer", tok)
    monkeypatch.setattr(serve, "EOS_ID", tok.eos_token_id)
    monkeypatch.setattr(serve, "BOS_ID", tok.bos_token_id)
    monkeypatch.setattr(serve, "ARGS", SimpleNamespace(max_context=128))
    seen = {}

    def fake_stream(x, max_tokens, *a, **k):
        seen["prompt_len"] = x.shape[1]
        seen["gen_budget"] = max_tokens
        seen["first_id"] = int(x[0, 0])
        yield from [5, 6, tok.eos_token_id]

    monkeypatch.setattr(serve, "_stream_ids_locked", fake_stream)
    stats: dict = {}
    long_prompt = "many words ride along here " * 200
    out = "".join(serve._generate_text(long_prompt, 256, 0.0, 0.9, stats=stats))
    assert isinstance(out, str)
    # prompt capped at max_context - MIN_GEN_TOKENS (=64), BOS included
    assert seen["prompt_len"] == 128 - serve.MIN_GEN_TOKENS
    assert seen["first_id"] == tok.bos_token_id
    # generation gets exactly the room the prompt left, not the client's 256
    assert seen["gen_budget"] == 128 - seen["prompt_len"]
    assert stats["prompt_tokens"] == seen["prompt_len"]
    assert stats["completion_tokens"] == 2
    assert stats["finish"] == "stop"


# ---------------------------------------------------------------------------
# _with_context: the train/serve system-shape contract (ultrareview #6).
# BOTH sides are real -- serve's join vs make_sft_data's generator constants.
# The old test_memory_tools_data pinned training against a hardcoded string
# copy; this is the missing serve half.
# ---------------------------------------------------------------------------


def test_with_context_matches_training_system_shape(monkeypatch, tok, tmp_path):
    store = MemoryStore(str(tmp_path / "mem"))
    store.add("User's dog is named Rex.")
    monkeypatch.setattr(serve, "MEMORY", store)
    monkeypatch.setattr(serve, "tokenizer", tok)
    monkeypatch.setattr(serve, "SPEAKER", None)
    monkeypatch.setattr(serve, "PAINTER", None)

    user = "What is my dog called? Also compute 372 + 519."
    # sanity: this text trips exactly the calculate gate
    assert serve._builtin_tools(user, False) == [serve._CALC_TOOL]

    req = serve.ChatReq(messages=[serve.Msg(role="user", content=user)])
    out = serve._with_context([m.model_dump(exclude_none=True) for m in req.messages], req)

    mem_block = store.render_context(user, tok, max_ids=128)
    assert mem_block.startswith("Things you remember:")  # retrieval really hit
    calc = next(t for t in make_sft_data.TOOLS if t[0] == "calculate")
    expected = mem_block + "\n\n" + make_sft_data._system([calc])
    assert out[0]["role"] == "system"
    assert out[0]["content"] == expected  # byte-identical to the trained shape


def test_with_context_client_system_message_is_appended_not_preambled(monkeypatch, tok, tmp_path):
    """A client-supplied system message keeps its own opener: memories and the
    bare tools block are appended with the blank-line join, and serve's
    'You are Enigma' preamble is NOT injected on top of it."""
    store = MemoryStore(str(tmp_path / "mem"))
    store.add("User's dog is named Rex.")
    monkeypatch.setattr(serve, "MEMORY", store)
    monkeypatch.setattr(serve, "tokenizer", tok)
    monkeypatch.setattr(serve, "SPEAKER", None)
    monkeypatch.setattr(serve, "PAINTER", None)

    user = "What is my dog called? Also compute 372 + 519."
    req = serve.ChatReq(messages=[serve.Msg(role="user", content=user)])
    msgs = [{"role": "system", "content": "Client system."}, {"role": "user", "content": user}]
    out = serve._with_context(msgs, req)

    mem_block = store.render_context(user, tok, max_ids=128)
    tools_block = render_tools_system([serve._CALC_TOOL])
    assert out[0]["content"] == "Client system." + "\n\n" + mem_block + "\n\n" + tools_block
    assert "You are Enigma" not in out[0]["content"]


# ---------------------------------------------------------------------------
# stream vs non-stream byte parity (ultrareview #31 + re-audit 2026-07-17).
# The whitespace shapes are exactly the class that broke the first fix.
# ---------------------------------------------------------------------------


def _drain_stream(resp) -> str:
    async def _drain():
        frames = []
        async for chunk in resp.body_iterator:
            frames.append(chunk if isinstance(chunk, str) else chunk.decode("utf-8"))
        return frames

    parts = []
    for frame in asyncio.run(_drain()):
        for line in frame.splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0]["delta"]
            if delta.get("content"):
                parts.append(delta["content"])
    return "".join(parts)


@pytest.mark.parametrize(
    "gen_text",
    [
        "Hello world.",
        " Hello world.\n",
        "Line one.\n\nLine two.   ",
        "   ",  # whitespace-only hop
        "Hi.\nBye. ",
    ],
)
def test_stream_and_nonstream_content_byte_identical(monkeypatch, tok, gen_text):
    monkeypatch.setattr(serve, "tokenizer", tok)
    monkeypatch.setattr(serve, "EOS_ID", tok.eos_token_id)
    monkeypatch.setattr(serve, "BOS_ID", tok.bos_token_id)
    monkeypatch.setattr(serve, "ARGS", SimpleNamespace(max_context=512))
    monkeypatch.setattr(serve, "MEMORY", None)
    monkeypatch.setattr(serve, "SPEAKER", None)
    monkeypatch.setattr(serve, "PAINTER", None)
    scripted = tok.encode(gen_text, add_special_tokens=False)

    def fake_gen(ids, max_tokens, *a, **k):
        yield from scripted

    monkeypatch.setattr(serve, "_gen_ids", fake_gen)

    def _req(stream: bool) -> serve.ChatReq:
        return serve.ChatReq(
            messages=[serve.Msg(role="user", content="Tell me a story.")],
            stream=stream,
            max_tokens=64,
        )

    resp = serve._chat_instruct(_req(stream=False))
    nonstream = resp["choices"][0]["message"]["content"] or ""
    stream = _drain_stream(serve._chat_instruct(_req(stream=True)))
    assert stream == nonstream


# ---------------------------------------------------------------------------
# mute: roundtrip, persistence, 204, and the speak-tool gate
# ---------------------------------------------------------------------------


def test_mute_roundtrip_persists_and_gates(monkeypatch, tmp_path):
    state = tmp_path / "mute_state.json"
    monkeypatch.setattr(serve, "_MUTE_STATE", state)
    monkeypatch.setattr(serve, "MUTED", False)
    monkeypatch.setattr(serve, "SPEAKER", object())

    assert serve.set_mute(serve.MuteReq(muted=True)) == {"muted": True}
    assert serve.MUTED is True
    assert json.loads(state.read_text(encoding="utf-8")) == {"muted": True}
    assert serve.get_mute() == {"muted": True}

    # muted speech endpoint: 204, no audio, no synthesis attempted
    resp = serve.audio_speech(serve.SpeechReq(input="hello"))
    assert resp.status_code == 204
    # muted speak TOOL: honest "muted:" result string, not an exception
    assert serve._execute_builtin("speak", {"text": "hi"}).startswith("muted:")

    assert serve.set_mute(serve.MuteReq(muted=False)) == {"muted": False}
    assert json.loads(state.read_text(encoding="utf-8")) == {"muted": False}


# ---------------------------------------------------------------------------
# talk-mode toggle, stop endpoint, status poll, and runtime voice endpoints
# ---------------------------------------------------------------------------


class _FakeSpeaker:
    def __init__(self):
        self.stopped = 0
        self.last_error = None
        self._recipe = {"engine": "kokoro", "lang_code": "a",
                        "blend": [["af_heart", 1.0]], "speed": 1.0}

    def stop(self):
        self.stopped += 1

    def get_voice(self):
        return dict(self._recipe)

    def set_voice(self, blend=None, speed=None, lang_code=None):
        if speed is not None and not 0.5 <= speed <= 2.0:
            raise serve.TTSError("speed must be between 0.5 and 2.0")
        if speed is not None:
            self._recipe["speed"] = speed
        if blend is not None:
            self._recipe["blend"] = blend
        return dict(self._recipe)


def test_talk_mode_roundtrip_persists(monkeypatch, tmp_path):
    state = tmp_path / "talk_mode.json"
    monkeypatch.setattr(serve, "_TALK_STATE", state)
    monkeypatch.setattr(serve, "TALK_MODE", False)
    assert serve.get_talk_mode() == {"enabled": False}
    assert serve.set_talk_mode(serve.TalkReq(enabled=True)) == {"enabled": True}
    assert serve.TALK_MODE is True
    assert json.loads(state.read_text(encoding="utf-8")) == {"enabled": True}
    assert serve.get_talk_mode() == {"enabled": True}


def test_stop_bumps_generation_and_aborts_speaker(monkeypatch):
    spk = _FakeSpeaker()
    monkeypatch.setattr(serve, "SPEAKER", spk)
    monkeypatch.setattr(serve, "_STOP_GEN", 0)
    assert serve.audio_stop() == {"stopped": True, "stop_gen": 1}
    assert spk.stopped == 1
    assert serve.audio_stop()["stop_gen"] == 2 and spk.stopped == 2


def test_stop_is_safe_when_voice_off(monkeypatch):
    monkeypatch.setattr(serve, "SPEAKER", None)
    monkeypatch.setattr(serve, "_STOP_GEN", 5)
    assert serve.audio_stop() == {"stopped": False, "stop_gen": 6}  # never 503s


def test_status_reports_mute_talk_and_stop_gen(monkeypatch):
    spk = _FakeSpeaker()
    monkeypatch.setattr(serve, "SPEAKER", spk)
    monkeypatch.setattr(serve, "MUTED", True)
    monkeypatch.setattr(serve, "TALK_MODE", True)
    monkeypatch.setattr(serve, "_STOP_GEN", 3)
    assert serve.audio_status() == {
        "muted": True, "talk_mode": True, "stop_gen": 3, "voice": "ok"}


def test_status_reports_voice_health(monkeypatch):
    """A broken audio device (play jobs failing in the worker with only a
    console WARN) must be visible on the poll, not silent (audit 2026-07-23 M3)."""
    monkeypatch.setattr(serve, "MUTED", False)
    monkeypatch.setattr(serve, "TALK_MODE", False)
    # voice off
    monkeypatch.setattr(serve, "SPEAKER", None)
    assert serve.audio_status()["voice"] == "off"
    # voice healthy
    spk = _FakeSpeaker()
    monkeypatch.setattr(serve, "SPEAKER", spk)
    assert serve.audio_status()["voice"] == "ok"
    assert "voice_error" not in serve.audio_status()
    # voice erroring -- the error surfaces
    spk.last_error = RuntimeError("no audio device")
    s = serve.audio_status()
    assert s["voice"] == "error" and "no audio device" in s["voice_error"]


def test_voice_get_set_and_validation(monkeypatch):
    spk = _FakeSpeaker()
    monkeypatch.setattr(serve, "SPEAKER", spk)
    assert serve.get_voice()["blend"] == [["af_heart", 1.0]]
    assert serve.set_voice(serve.VoiceReq(speed=1.2))["speed"] == 1.2
    with pytest.raises(serve.HTTPException) as ei:  # bad recipe = 400, not 500
        serve.set_voice(serve.VoiceReq(speed=9.0))
    assert ei.value.status_code == 400


def test_voice_endpoints_organ_off_when_disabled(monkeypatch):
    monkeypatch.setattr(serve, "SPEAKER", None)
    with pytest.raises(serve.HTTPException):
        serve.get_voice()
    with pytest.raises(serve.HTTPException):
        serve.set_voice(serve.VoiceReq(speed=1.0))


def test_set_voice_endpoint_maps_real_validation_to_400(monkeypatch):
    """The 400-not-500 contract, exercised through the REAL Speaker/normalizer
    -- not a fake that reimplements validation (audit 2026-07-23 T2). A malformed
    blend item raises a raw ValueError/TypeError inside _normalize_recipe unless
    it is caught as TTSError; this pins that it surfaces as a 400."""
    from enigma_engine.core.tts import Speaker

    class _Backend:
        sample_rate = 24000
        def synth(self, text):
            import numpy as np
            return np.zeros(8, dtype=np.float32)
        def set_recipe(self, recipe):
            pass

    spk = Speaker(synth_factory=lambda: _Backend(), player_factory=lambda: None)
    monkeypatch.setattr(serve, "SPEAKER", spk)
    try:
        for bad in (serve.VoiceReq(speed=9.0),                 # out of range
                    serve.VoiceReq(blend=[["af_heart", "loud"]]),  # non-numeric weight
                    serve.VoiceReq(blend=[["nope", 1.0]])):    # unknown voice
            with pytest.raises(serve.HTTPException) as ei:
                serve.set_voice(bad)
            assert ei.value.status_code == 400
        # a valid change still succeeds through the real path
        assert serve.set_voice(serve.VoiceReq(speed=1.3))["speed"] == 1.3
    finally:
        spk.close()


# ---------------------------------------------------------------------------
# --eyes graft guards (_load_eyes): every malformed align checkpoint must
# raise EyesError so boot() degrades to text-only instead of dying
# ---------------------------------------------------------------------------


def test_load_eyes_prefers_stored_encoder_config(tmp_path):
    """An align checkpoint carrying vision_encoder_config must rebuild THAT
    architecture even when the caller passes a mismatched preset name --
    the preset-coupling silent-degrade risk (2026-07-20 eyes polish). Older
    checkpoints without the key keep the preset path (covered by the
    guards/happy-path test)."""
    from enigma_engine.core.vision_encoder import VISION_PRESETS, VisionEncoder

    enc = VisionEncoder(VISION_PRESETS["small"])
    ck = tmp_path / "stored_cfg.pt"
    torch.save(
        {
            "vision_encoder_state_dict": enc.state_dict(),
            "vision_encoder_config": VISION_PRESETS["small"].to_dict(),
            "model_state_dict": {"vision_projection.0.weight": torch.zeros(4, 4)},
        },
        ck,
    )
    # preset says "medium" (512d) but the checkpoint is "small" -- stored
    # config must win or the strict load would raise RuntimeError
    venc, proj_sd, dim = serve._load_eyes(ck, "medium")
    assert dim == VISION_PRESETS["small"].dim
    assert "0.weight" in proj_sd


@pytest.mark.parametrize(
    "field, value",
    [
        ("field_from_the_future", 7),  # unknown key: a newer writer's config
        ("patch_size", 0),  # the ONE field the dataclass itself validates
        ("image_size", 225),  # not divisible by patch_size -- validated lazily
        ("dropout", 1.7),  # out of range; raises inside torch, not the config
        ("n_heads", 0),  # ZeroDivisionError while building the encoder
        ("n_layers", 2.5),  # TypeError deep in construction
        ("image_size", None),  # None where an int is required
        ("dim", 0),  # ZeroDivisionError on head_dim
    ],
)
def test_load_eyes_rejects_unusable_stored_config(tmp_path, field, value):
    """EVERY way a stored vision_encoder_config can fail to rebuild must surface
    as EyesError. The dataclass validates only patch_size; image_size
    divisibility, head/layer counts and dropout range are not checked until the
    encoder is constructed, so a guard around the dataclass call alone leaves
    ValueError/TypeError/ZeroDivisionError escaping boot()'s degrade catch and
    killing text serving with the eyes."""
    from enigma_engine.core.vision_encoder import VISION_PRESETS, VisionEncoder

    enc = VisionEncoder(VISION_PRESETS["small"])
    cfg = VISION_PRESETS["small"].to_dict()
    cfg[field] = value
    ckpt = tmp_path / "unusable.pt"
    torch.save(
        {
            "vision_encoder_state_dict": enc.state_dict(),
            "vision_encoder_config": cfg,
            "model_state_dict": {"vision_projection.0.weight": torch.zeros(4, 4)},
        },
        ckpt,
    )
    with pytest.raises(EyesError, match="will not rebuild"):
        serve._load_eyes(ckpt, "small")


@pytest.mark.parametrize("preset", ["tiny", "medium"])
def test_load_eyes_empty_stored_config_falls_back_to_preset(tmp_path, preset):
    """An empty stored config must NOT be treated as a config: building from it
    would produce an all-defaults encoder and ignore --eyes-preset.

    The presets here must DIFFER from VisionEncoderConfig()'s defaults -- those
    defaults equal the "small" preset, so a "small" case would pass whether or
    not the fallback works."""
    from enigma_engine.core.vision_encoder import VISION_PRESETS, VisionEncoder

    enc = VisionEncoder(VISION_PRESETS[preset])
    ckpt = tmp_path / "empty_cfg.pt"
    torch.save(
        {
            "vision_encoder_state_dict": enc.state_dict(),
            "vision_encoder_config": {},
            "model_state_dict": {"vision_projection.0.weight": torch.zeros(4, 4)},
        },
        ckpt,
    )
    _venc, _proj, dim = serve._load_eyes(ckpt, preset)
    assert dim == VISION_PRESETS[preset].dim


def test_load_eyes_rejects_unreadable_checkpoint(tmp_path):
    """A truncated or non-checkpoint file must degrade the eyes, not kill boot:
    torch.load raises EOFError/UnpicklingError, which boot's catch does not cover."""
    empty = tmp_path / "empty.pt"
    empty.write_bytes(b"")
    with pytest.raises(EyesError, match="could not be read"):
        serve._load_eyes(empty, "small")

    garbage = tmp_path / "garbage.pt"
    garbage.write_text("this is not a checkpoint", encoding="ascii")
    with pytest.raises(EyesError, match="could not be read"):
        serve._load_eyes(garbage, "small")

    not_a_dict = tmp_path / "not_a_dict.pt"
    torch.save([1, 2, 3], not_a_dict)
    with pytest.raises(EyesError, match="vision_encoder_state_dict"):
        serve._load_eyes(not_a_dict, "small")


def test_load_eyes_guards_and_happy_path(tmp_path):
    from enigma_engine.core.vision_encoder import VISION_PRESETS, VisionEncoder

    with pytest.raises(EyesError, match="not found"):
        serve._load_eyes(tmp_path / "missing.pt", "small")

    no_enc = tmp_path / "no_enc.pt"
    torch.save({"model_state_dict": {}}, no_enc)
    with pytest.raises(EyesError, match="vision_encoder_state_dict"):
        serve._load_eyes(no_enc, "small")

    enc = VisionEncoder(VISION_PRESETS["small"])
    no_proj = tmp_path / "no_proj.pt"
    torch.save(
        {"vision_encoder_state_dict": enc.state_dict(), "model_state_dict": {"tok.weight": torch.zeros(1)}},
        no_proj,
    )
    with pytest.raises(EyesError, match="vision_projection"):
        serve._load_eyes(no_proj, "small")

    good = tmp_path / "good.pt"
    torch.save(
        {
            "vision_encoder_state_dict": enc.state_dict(),
            "model_state_dict": {
                "vision_projection.0.weight": torch.zeros(4, 4),
                "vision_projection.0.bias": torch.zeros(4),
                "tok_embeddings.weight": torch.zeros(2, 2),  # text keys are not projection keys
            },
        },
        good,
    )
    venc, proj_sd, dim = serve._load_eyes(good, "small")
    assert set(proj_sd) == {"0.weight", "0.bias"}  # prefix stripped, text keys excluded
    assert dim == VISION_PRESETS["small"].dim

    # the non-EyesError degrade paths boot() also catches (re-audit
    # 2026-07-18: these two classes were untested):
    with pytest.raises(KeyError):  # unknown preset
        serve._load_eyes(good, "no_such_preset")
    bad_sd = tmp_path / "bad_sd.pt"
    torch.save(
        {"vision_encoder_state_dict": {"nope.weight": torch.zeros(1)}, "model_state_dict": {}},
        bad_sd,
    )
    with pytest.raises(RuntimeError):  # strict load on a mismatched encoder
        serve._load_eyes(bad_sd, "small")


# ---------------------------------------------------------------------------
# boot() end to end on a tiny checkpoint (CPU-forced; every global restored)
# ---------------------------------------------------------------------------

_RUNTIME_GLOBALS = [
    "ARGS", "CONFIG", "model", "tokenizer", "DEVICE", "_BF16_GEN", "STEP", "META",
    "INSTRUCT", "MEMORY", "SPEAKER", "MUTED", "EARS", "EYES", "PAINTER", "EOS_ID", "BOS_ID",
    "_BOOTED",
]


_HF_ENV_KEYS = (
    "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
    "HF_HUB_DISABLE_TELEMETRY", "HF_HUB_DISABLE_IMPLICIT_TOKEN",
)


def test_boot_tiny_checkpoint(monkeypatch, tmp_path):
    """The full startup path on a 2-layer toy model, SIX boots: the first
    exercises the --allow-downloads env branch AND the KV-cache clamp
    (--max-context 4096 vs max_seq_len 256 -- the 2026-07-17 version never
    entered either branch); the second, flagless boot must RESTORE the
    offline default despite the first boot's leftover "0" (the double-boot
    hole, re-audit 2026-07-18); legs C/D pin the operator-export semantics.
    CUDA is masked off so this never touches the GPU; mute state and env are
    patched hermetic and restored."""
    import os

    snapshot = {name: getattr(serve, name) for name in _RUNTIME_GLOBALS}
    monkeypatch.setattr(serve.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(serve, "_MUTE_STATE", tmp_path / "mute_state.json")
    monkeypatch.setattr(serve, "_BOOT_ENV_WRITES", {})
    for key in _HF_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)  # no operator values in play
    cfg = ForgeConfig(
        vocab_size=64, dim=32, n_layers=2, n_heads=2,
        max_seq_len=256, dropout=0.0, use_gradient_checkpointing=False,
    )
    torch.manual_seed(0)
    ckpt = tmp_path / "tiny.pth"
    torch.save(
        {
            "model_state_dict": Enigma(cfg).state_dict(),
            "config": cfg.to_dict(),
            "step": 7,
            "meta": {"chat_format": CHAT_FORMAT_NAME},
        },
        ckpt,
    )
    try:
        serve.boot(argv=["--model", str(ckpt), "--max-context", "4096", "--allow-downloads"])
        assert serve._BOOTED is True  # readiness = boot ran to completion
        assert serve.model is not None
        assert serve.DEVICE == "cpu"
        assert serve.INSTRUCT is True  # meta.chat_format detected
        assert serve.STEP == 7
        # oversize budget clamps to the model's real cache capacity
        assert serve.ARGS.max_context == 256
        assert os.environ["HF_HUB_OFFLINE"] == "0"  # the flag won, out loud

        serve.boot(argv=["--model", str(ckpt), "--max-context", "128"])
        assert serve.ARGS.max_context == 128  # in-budget value passes through
        assert serve.tokenizer is not None
        assert serve.EOS_ID == serve.tokenizer.eos_token_id
        # the flagless boot restored the offline default -- before the
        # ownership fix, the first boot's "0" survived setdefault and the
        # second server would have phoned home
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"

        # OPERATOR-EXPORT legs (round-2 re-audit: the delete-only ownership
        # scheme destroyed a genuine export across these exact sequences).
        # (C) an export that AGREES with the flag is never claimed and
        # survives an allow-downloads -> flagless boot pair untouched:
        os.environ["HF_HUB_OFFLINE"] = "0"  # operator: downloads always ok
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        serve.boot(argv=["--model", str(ckpt), "--max-context", "128", "--allow-downloads"])
        # "never claimed" is a pinned claim, not prose (round-4 re-audit):
        assert "HF_HUB_OFFLINE" not in serve._BOOT_ENV_WRITES
        serve.boot(argv=["--model", str(ckpt), "--max-context", "128"])
        assert os.environ["HF_HUB_OFFLINE"] == "0"  # respected, not forced to 1
        assert os.environ["TRANSFORMERS_OFFLINE"] == "0"
        # (D) an export the flag DISPLACED is restored on the next boot.
        # "true" on purpose (a huggingface_hub-recognized truthy spelling):
        # the displaced value must differ from the "1" that setdefault would
        # write, or delete-then-setdefault masquerades as a restore and the
        # restore half of the fix is unpinned (round-3 re-audit 2026-07-18).
        os.environ["HF_HUB_OFFLINE"] = "true"  # operator: hard offline
        os.environ["TRANSFORMERS_OFFLINE"] = "true"
        serve.boot(argv=["--model", str(ckpt), "--max-context", "128", "--allow-downloads"])
        assert os.environ["HF_HUB_OFFLINE"] == "0"  # the flag wins, out loud
        assert serve._BOOT_ENV_WRITES["HF_HUB_OFFLINE"] == ("true", "0")
        serve.boot(argv=["--model", str(ckpt), "--max-context", "128"])
        assert os.environ["HF_HUB_OFFLINE"] == "true"  # the LITERAL export is back
        assert os.environ["TRANSFORMERS_OFFLINE"] == "true"
    finally:
        for name, value in snapshot.items():
            setattr(serve, name, value)
        for key in _HF_ENV_KEYS:
            os.environ.pop(key, None)  # monkeypatch teardown restores originals


def _tiny_ckpt(tmp_path):
    cfg = ForgeConfig(
        vocab_size=64, dim=32, n_layers=2, n_heads=2,
        max_seq_len=256, dropout=0.0, use_gradient_checkpointing=False,
    )
    torch.manual_seed(0)
    ckpt = tmp_path / "tiny_degrade.pth"
    torch.save({"model_state_dict": Enigma(cfg).state_dict(), "config": cfg.to_dict()}, ckpt)
    return ckpt


def test_boot_brings_real_organs_up(monkeypatch, tmp_path):
    """Positive control for the degrade contract: the broad WARN-and-continue
    catches would also swallow a future constructor regression (a typo, a
    signature drift at the call site) and leave every boot silently
    amnesiac/blind while the whole suite stays green (audit 2026-07-22).
    This boots with REAL constructors and asserts the organs actually exist."""
    import os

    from enigma_engine.core.vision_encoder import VISION_PRESETS, VisionEncoder

    snapshot = {name: getattr(serve, name) for name in _RUNTIME_GLOBALS}
    monkeypatch.setattr(serve.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(serve, "_MUTE_STATE", tmp_path / "mute_state.json")
    monkeypatch.setattr(serve, "_BOOT_ENV_WRITES", {})
    for key in _HF_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    cfg = ForgeConfig(
        vocab_size=64, dim=32, n_layers=2, n_heads=2,
        max_seq_len=256, dropout=0.0, use_gradient_checkpointing=False,
    )
    torch.manual_seed(0)
    ckpt = tmp_path / "tiny_organs.pth"
    torch.save({"model_state_dict": Enigma(cfg).state_dict(), "config": cfg.to_dict()}, ckpt)

    enc = VisionEncoder(VISION_PRESETS["tiny"])
    vdim = VISION_PRESETS["tiny"].dim
    eyes_ckpt = tmp_path / "tiny_eyes.pt"
    torch.save(
        {
            "vision_encoder_state_dict": enc.state_dict(),
            "vision_encoder_config": VISION_PRESETS["tiny"].to_dict(),
            "model_state_dict": {
                "vision_projection.0.weight": torch.zeros(32, vdim),
                "vision_projection.0.bias": torch.zeros(32),
                "vision_projection.2.weight": torch.zeros(32, 32),
                "vision_projection.2.bias": torch.zeros(32),
            },
        },
        eyes_ckpt,
    )
    try:
        serve.boot(argv=[
            "--model", str(ckpt), "--max-context", "128",
            "--memory-dir", str(tmp_path / "mem"),
            "--eyes", "--eyes-model", str(eyes_ckpt), "--eyes-preset", "tiny",
        ])
        assert serve._BOOTED is True
        assert serve.MEMORY is not None, "real MemoryStore failed to construct"
        assert serve.MEMORY.remember("User's cat is named Biscuit.")
        assert serve.EYES is not None, "real Eyes failed to construct"
    finally:
        for name, value in snapshot.items():
            setattr(serve, name, value)
        for key in _HF_ENV_KEYS:
            os.environ.pop(key, None)


def test_boot_survives_an_unusable_memory_dir(monkeypatch, tmp_path):
    """A memory dir that cannot be opened -- locked by another process, full,
    unwritable -- must cost her memory, not text serving. MemoryStore mkdirs
    and reads in __init__, and every launcher passes --memory-dir, so an
    unguarded construction takes down EVERY boot."""
    import os

    import enigma_engine.core.memory_store as memory_store

    snapshot = {name: getattr(serve, name) for name in _RUNTIME_GLOBALS}
    monkeypatch.setattr(serve.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(serve, "_MUTE_STATE", tmp_path / "mute_state.json")
    monkeypatch.setattr(serve, "_BOOT_ENV_WRITES", {})
    for key in _HF_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    def _explode(*_a, **_k):
        raise OSError(22, "The process cannot access the file")

    monkeypatch.setattr(memory_store, "MemoryStore", _explode)
    ckpt = _tiny_ckpt(tmp_path)
    try:
        serve.boot(argv=["--model", str(ckpt), "--max-context", "128",
                         "--memory-dir", str(tmp_path / "mem")])
        assert serve._BOOTED is True  # text serving came up
        assert serve.MEMORY is None  # the organ degraded
    finally:
        for name, value in snapshot.items():
            setattr(serve, name, value)
        for key in _HF_ENV_KEYS:
            os.environ.pop(key, None)


def test_boot_survives_eyes_construction_failure(monkeypatch, tmp_path):
    """Eyes(...) moves the encoder onto the device, so a busy GPU raises
    torch.OutOfMemoryError -- a RuntimeError, not EyesError. Catching only
    EyesError killed boot in exactly the gaming case this machine is built
    for. The loader is stubbed so the failure lands on construction."""
    import os

    snapshot = {name: getattr(serve, name) for name in _RUNTIME_GLOBALS}
    monkeypatch.setattr(serve.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(serve, "_MUTE_STATE", tmp_path / "mute_state.json")
    monkeypatch.setattr(serve, "_BOOT_ENV_WRITES", {})
    for key in _HF_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    from enigma_engine.core.vision_encoder import VISION_PRESETS, VisionEncoder

    enc = VisionEncoder(VISION_PRESETS["small"])
    vdim = VISION_PRESETS["small"].dim
    proj_sd = {
        "0.weight": torch.zeros(32, vdim), "0.bias": torch.zeros(32),
        "2.weight": torch.zeros(32, 32), "2.bias": torch.zeros(32),
    }
    monkeypatch.setattr(serve, "_load_eyes", lambda *_a, **_k: (enc, proj_sd, vdim))

    def _oom(*_a, **_k):
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(serve, "Eyes", _oom)
    ckpt = _tiny_ckpt(tmp_path)
    try:
        serve.boot(argv=["--model", str(ckpt), "--max-context", "128", "--eyes"])
        assert serve._BOOTED is True  # text serving came up
        assert serve.EYES is None  # the organ degraded
    finally:
        for name, value in snapshot.items():
            setattr(serve, name, value)
        for key in _HF_ENV_KEYS:
            os.environ.pop(key, None)


def test_boot_max_context_follows_long_config(monkeypatch, tmp_path):
    """A model whose config asks for more than Attention.MAX_CACHE_SEQ_LEN keeps
    its full context: the KV cache allocates config.max_seq_len, so the serve
    clamp must not bound the budget by the fallback constant (v2 targets 4k-8k)."""
    import os

    from enigma_engine.core.model_components import Attention

    snapshot = {name: getattr(serve, name) for name in _RUNTIME_GLOBALS}
    monkeypatch.setattr(serve.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(serve, "_MUTE_STATE", tmp_path / "mute_state.json")
    monkeypatch.setattr(serve, "_BOOT_ENV_WRITES", {})
    for key in _HF_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    long_ctx = Attention.MAX_CACHE_SEQ_LEN * 2
    cfg = ForgeConfig(
        vocab_size=64, dim=32, n_layers=2, n_heads=2,
        max_seq_len=long_ctx, dropout=0.0, use_gradient_checkpointing=False,
    )
    torch.manual_seed(0)
    ckpt = tmp_path / "long_ctx.pth"
    torch.save({"model_state_dict": Enigma(cfg).state_dict(), "config": cfg.to_dict()}, ckpt)
    try:
        serve.boot(argv=["--model", str(ckpt), "--max-context", str(long_ctx)])
        assert serve.ARGS.max_context == long_ctx  # not clamped to the fallback
    finally:
        for name, value in snapshot.items():
            setattr(serve, name, value)
        for key in _HF_ENV_KEYS:
            os.environ.pop(key, None)
