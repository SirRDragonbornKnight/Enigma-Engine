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
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import make_sft_data
import serve_enigma as serve
from enigma_engine.core.chat_format import (
    CHAT_FORMAT_NAME,
    attach_chat_tokens,
    chat_token_ids,
    render_tools_system,
    search_token_ids,
    think_token_ids,
)
from enigma_engine.core.eyes import EyesError
from enigma_engine.core.memory_store import MemoryStore
from enigma_engine.core.model import Enigma
from enigma_engine.core.model_presets import ForgeConfig
from enigma_engine.core.persona import PACK_MANIFEST, Persona
from enigma_engine.core.tokenizer import get_tokenizer, vocab_file_for_size

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


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("knob", ["temperature", "top_p", "min_p", "repetition_penalty"])
def test_a_non_finite_sampling_knob_is_refused_not_sampled(monkeypatch, literal, knob):
    """JSON carries NaN/Infinity as BARE LITERALS and json.loads parses them, so
    a client could hand one straight to the sampler. logits/NaN turns the whole
    row NaN, which wipes the -inf vocab-padding mask BEFORE sample_next_token
    saves its pre-filter copy -- the second-level guard then reads the padded
    columns as allowed and draws uniformly over them (measured 2026-08-22:
    67/200 draws landed in masked rows at NaN, 74/200 at inf). That is <unk>
    spam plus chat-special ids firing tool spans out of noise. It must be a
    client error, and nothing may generate."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(serve, "_BOOTED", True)

    def never(*a, **k):
        raise AssertionError("generation ran on a non-finite sampling knob")

    monkeypatch.setattr(serve, "_gen_ids", never)
    monkeypatch.setattr(serve, "_generate_text", never)
    client = TestClient(serve.app)
    headers = {"content-type": "application/json"}

    chat = ('{"messages": [{"role": "user", "content": "hi"}], '
            f'"{knob}": {literal}}}')
    assert client.post("/v1/chat/completions", content=chat, headers=headers).status_code == 422
    comp = f'{{"prompt": "hi", "{knob}": {literal}}}'
    assert client.post("/v1/completions", content=comp, headers=headers).status_code == 422


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_a_non_finite_voice_speed_is_refused(monkeypatch, literal):
    """VoiceReq.speed rides the same bare-JSON-literal door the sampling knobs
    closed: it reaches Kokoro's rate AND is persisted into voice.json as the
    saved recipe, so a single NaN outlives the request that sent it."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(serve, "_BOOTED", True)

    class _NeverSpeaker:
        def set_voice(self, **kwargs):
            raise AssertionError("the voice was retuned on a non-finite speed")

    monkeypatch.setattr(serve, "SPEAKER", _NeverSpeaker())
    client = TestClient(serve.app)
    resp = client.post(
        "/v1/audio/voice",
        content=f'{{"speed": {literal}}}',
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422
    # ...and a finite speed still reaches the organ
    assert serve.VoiceReq.model_validate_json('{"speed": 1.25}').speed == 1.25


def test_the_temperature_clamp_survives_a_non_finite_value():
    """Defense in depth for the same hole: the clamp is what every generation
    path funnels through, and `0.0 if t <= 0 else max(t, 1e-3)` passed NaN and
    inf through untouched (NaN fails both comparisons, inf wins max). A
    non-finite value falls back to the default the field would have carried."""
    assert serve._clamp_temperature(float("nan"), serve.CHAT_TEMPERATURE) == 0.3
    assert serve._clamp_temperature(float("inf"), serve.CHAT_TEMPERATURE) == 0.3
    assert serve._clamp_temperature(float("-inf"), serve.RAW_TEMPERATURE) == 0.8
    # ...and the finite behavior is unmoved: <= 0 is greedy, tiny is clamped
    assert serve._clamp_temperature(0.0, 0.3) == 0.0
    assert serve._clamp_temperature(-2.0, 0.3) == 0.0
    assert serve._clamp_temperature(1e-9, 0.3) == 1e-3
    assert serve._clamp_temperature(0.7, 0.3) == 0.7


# ---------------------------------------------------------------------------
# intent gates + built-in tool offering (tool-stealing history, 2026-07-06).
#
# EVERY lineage runs on these gates. The 2026-07-24 ruling retiring them for an
# always-offered built-in block stands as DIRECTION, but its serve-side
# execution is PARKED ON MEASUREMENT: gated twice on the adopted sft2
# checkpoint 2026-08-20, the block scored 59/120 (fixed five-tool) and 55/120
# (organ-filtered) against this gated offering's 67/120. The stamp test below
# pins the parked state so a re-flip cannot land silently.
# ---------------------------------------------------------------------------

# The two lineages as literals. Nothing in serve keys on them today -- that is
# the point of the stamp test: the offering does not vary by checkpoint.
V2_VOCAB = 16366
V8_VOCAB = 4718


def _lineage(monkeypatch, vocab_size, instruct=True):
    """Pin the globals a lineage-sensitive offering WOULD read.

    boot() sets both from the checkpoint (meta.chat_format and config
    vocab_size); these tests never load one."""
    monkeypatch.setattr(serve, "INSTRUCT", instruct)
    monkeypatch.setattr(serve, "CONFIG", SimpleNamespace(vocab_size=vocab_size))


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


def test_a_negated_forget_is_a_save_ask_not_a_delete(monkeypatch):
    monkeypatch.setattr(serve, "MEMORY", object())
    # "Don't forget my birthday is in May" wears the forget verb but asks to
    # SAVE. Reading it as a forget suppressed the save and offered deletion --
    # the inversion of the intent, on a destructive tool.
    for text in ("Don't forget my birthday is in May.",
                 "Dont forget that my anniversary is June 3rd.",
                 "Never forget my daughter is called Mia."):
        assert not serve._looks_forgettable(text)
        assert serve._looks_memorable(text)


def test_the_negation_does_not_have_to_touch_the_forget_verb(monkeypatch):
    monkeypatch.setattr(serve, "MEMORY", object())
    # Requiring the negation to sit AGAINST the verb read all of these as
    # deletion asks -- the same save-becomes-delete inversion, one adverb
    # further out. Both gates read one spelling of the family, so a shape
    # either of them recognizes cannot leave the other silent.
    for text in ("Don't ever forget my birthday is in May.",
                 "Never ever forget my wifi password is hunter2.",
                 "You must not forget my anniversary is in June.",
                 "We can't forget that my mom visits in May.",
                 "Please don't you forget my birthday is in May."):
        assert not serve._looks_forgettable(text), text
        assert serve._looks_memorable(text), text


def test_an_executed_builtin_is_visible_in_the_reply(monkeypatch, tok, tmp_path):
    """A built-in EXECUTES server-side and the hop loop consumes it, so the
    surfaced message carries no `tool_calls` for it. Nothing outside the server
    could then tell a built-in that fired from one that never happened: an organ
    probe scored 0 however well she routed, and a restraint probe expecting NO
    call passed while she called one. The reply now reports what ran."""
    from enigma_engine.core.memory_store import MemoryStore

    monkeypatch.setattr(serve, "tokenizer", tok)
    monkeypatch.setattr(serve, "EOS_ID", tok.eos_token_id)
    monkeypatch.setattr(serve, "BOS_ID", tok.bos_token_id)
    monkeypatch.setattr(serve, "ARGS", SimpleNamespace(max_context=512))
    monkeypatch.setattr(serve, "MEMORY", MemoryStore(tmp_path))
    monkeypatch.setattr(serve, "SPEAKER", None)
    monkeypatch.setattr(serve, "PAINTER", None)

    # hop 0 emits a remember call; hop 1 answers in plain text. The span
    # markers are ids, the way the model emits them -- encoding the whole
    # string would leave the closing marker as literal text, because the
    # splitter does not fire on `}<|/tool_call|>` with no space between.
    from enigma_engine.core.chat_format import TOOL_CALL, TOOL_CALL_END

    payload = json.dumps({"name": "remember", "arguments": {"text": "User likes tea."}})
    hops = [[TOOL_CALL] + tok.encode(payload, add_special_tokens=False) + [TOOL_CALL_END],
            tok.encode("Saved that for you.", add_special_tokens=False)]
    seq = iter(hops)

    def fake_gen(ids, max_tokens, *a, **k):
        yield from next(seq, hops[-1])

    monkeypatch.setattr(serve, "_gen_ids", fake_gen)
    resp = serve._chat_instruct(serve.ChatReq(
        messages=[serve.Msg(role="user", content="Remember that I like tea.")],
        max_tokens=64,
    ))

    assert "tool_calls" not in resp["choices"][0]["message"], (
        "fixture assumption gone: the built-in is no longer consumed by the loop, "
        "so the surfaced calls would already have shown it"
    )
    assert resp["enigma"]["tools_run"] == ["remember"]
    assert [r["text"] for r in serve.MEMORY.all()] == ["User likes tea."]


def test_generated_images_are_served_only_from_the_images_dir(monkeypatch, tmp_path):
    """The imagine tool answers with a filesystem path, so the page could only
    print it as a sentence. Serving it needs a fetchable URL -- and that URL is
    the one place a name from a reply reaches the filesystem, so it takes a bare
    name, matches a strict pattern, and re-checks the resolved parent."""
    from fastapi.testclient import TestClient

    images = tmp_path / "images"
    images.mkdir()
    (images / "imagine_ab12cd34.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
    (tmp_path / "secret.png").write_bytes(b"\x89PNG\r\n\x1a\nSECRET")
    monkeypatch.setattr(serve, "IMAGES_DIR", images)
    monkeypatch.setattr(serve, "_BOOTED", True)
    client = TestClient(serve.app)

    ok = client.get("/v1/images/file/imagine_ab12cd34.png")
    assert ok.status_code == 200 and ok.content.startswith(b"\x89PNG")

    for attack in ("../secret.png", "..%2Fsecret.png", "..\\secret.png",
                   "%2e%2e%2fsecret.png", "/etc/passwd", "C:/Windows/win.ini",
                   "imagine_ab12cd34.png.txt", "nope.png", ".png"):
        r = client.get("/v1/images/file/" + attack)
        assert r.status_code == 404, f"{attack} was served"
        assert b"SECRET" not in r.content, f"{attack} leaked a file outside the images dir"


def test_the_page_shows_a_sense_control_only_when_its_organ_exists(monkeypatch):
    """A control that 404s teaches the user the feature is broken rather than
    absent, so the mic starts hidden and /v1/capabilities is what reveals it."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(serve, "_BOOTED", True)
    # builtins are also INSTRUCT-gated now (a base checkpoint's path never
    # reaches the tool loop, review 2026-08-13); this test is about ORGAN
    # gating, so pin the instruct side on.
    monkeypatch.setattr(serve, "INSTRUCT", True)
    for organ in ("MEMORY", "SPEAKER", "EARS", "EYES", "PAINTER"):
        monkeypatch.setattr(serve, organ, None)
    client = TestClient(serve.app)

    caps = client.get("/v1/capabilities").json()
    assert caps["ears"] is False and caps["image_gen"] is False
    assert "speak" not in caps["builtins"] and "imagine" not in caps["builtins"]
    assert "calculate" in caps["builtins"]

    monkeypatch.setattr(serve, "EARS", object())
    monkeypatch.setattr(serve, "PAINTER", object())
    caps = client.get("/v1/capabilities").json()
    assert caps["ears"] is True and "imagine" in caps["builtins"]

    page = client.get("/").text
    assert 'id="mic"' in page and "hidden" in page, "the mic must not show before it is confirmed"
    assert 'id="pic"' in page, "no way to show her a picture"
    assert "/v1/capabilities" in page, "the page never asks which organs exist"
    assert "/v1/audio/transcriptions" in page, "the mic posts nowhere"
    assert "/v1/images/file/" in page, "generated images are never rendered"
    # eyes are reached by sending image_url content, which serve captions before
    # anything else sees it -- data: URLs only.
    assert "image_url" in page and "readAsDataURL" in page


def test_an_ambiguous_forget_can_be_answered_with_an_id(monkeypatch, tmp_path):
    """The store refuses an ambiguous forget rather than guessing, and names the
    candidates. Without an id door that refusal was unanswerable from chat:
    records with identical term sets cannot be told apart by restating them, so
    the same error came back however the ask was reworded."""
    from enigma_engine.core.memory_store import MemoryStore

    mem = MemoryStore(tmp_path)
    mem.add("User likes tea.")
    mem.add("User likes tea.")  # identical terms: no wording separates them
    monkeypatch.setattr(serve, "MEMORY", mem)

    refusal = serve._execute_builtin("forget", {"text": "forget that I like tea"})
    assert refusal.startswith("error:")
    assert "#1" in refusal and "#2" in refusal, "the refusal must name ids to be answerable"
    assert len(mem.all()) == 2, "an ambiguous ask must not delete"

    # The id SELECTS among the records the text already matched.
    assert serve._execute_builtin(
        "forget", {"text": "forget that I like tea", "id": 2}
    ) == "forgot: User likes tea."
    assert [r["id"] for r in mem.all()] == [1]
    assert serve._execute_builtin("forget", {"id": "nope"}).startswith("error: memory id must be")


def test_her_which_one_question_can_actually_be_answered(monkeypatch, tmp_path):
    """The gate decided per request from the user's wording, so after she asked
    "say the one you mean word for word, or give its id", the natural answers
    armed nothing at all -- only a reply containing "forget" reached the tool
    again. A question the user cannot answer is not a question.

    Stateless: the client sends the history, so the pending question is in the
    request."""
    from enigma_engine.core.memory_store import MemoryStore

    mem = MemoryStore(tmp_path)
    mem.add("User likes tea.")
    mem.add("User likes tea.")
    monkeypatch.setattr(serve, "MEMORY", mem)

    refusal = serve._execute_builtin("forget", {"text": "forget that I like tea"})
    assert refusal.startswith("error:")
    # The history carries what the CLIENT actually holds: the surfaced
    # rendering (error transport prefix stripped), not the raw tool result.
    surfaced = serve._surfaced_forget_refusal("forget", refusal)
    assert surfaced is not None

    def offered(reply):
        msgs = [serve.Msg(role="user", content="forget that I like tea"),
                serve.Msg(role="assistant", content=surfaced),
                serve.Msg(role="user", content=reply)]
        return [t["function"]["name"] for t in serve._builtin_tools(reply, False, msgs)]

    for answer in ("#2", "2", "the second one", "id 2", "yes, #2",
                   "the green tea one", "User likes tea."):
        assert "forget" in offered(answer), answer

    # ...and an ordinary exchange still arms nothing: the marker is what opens
    # the door, not the mere presence of history.
    plain = [serve.Msg(role="user", content="hello"),
             serve.Msg(role="assistant", content="Hi there."),
             serve.Msg(role="user", content="what is the weather")]
    assert serve._builtin_tools("what is the weather", False, plain) == []


def test_only_the_real_refusal_surfaces_or_arms(monkeypatch, tmp_path):
    """Fix-arc audit, 2026-07-25: a substring test on the marker phrase could
    not tell THE question from text that merely quotes it. A stored memory
    containing "memories match that" rode out on a SUCCESS line ("forgot:
    ..."), was surfaced into her visible reply, and armed answering-mode with
    no question pending -- the next ordinary user turn was then read as
    naming a memory to delete. Only the exact TooBroad rendering, at a line
    start, is the handshake."""
    from enigma_engine.core.memory_store import MemoryStore, renders_forget_pending

    mem = MemoryStore(tmp_path)
    mem.add("Enigma printed 2 memories match that during the demo.")
    monkeypatch.setattr(serve, "MEMORY", mem)

    # a SUCCESS whose deleted record quotes the phrase is NOT the question
    result = serve._execute_builtin(
        "forget", {"text": "Enigma printed 2 memories match that during the demo."})
    assert result.startswith("forgot: ")
    assert serve._surfaced_forget_refusal("forget", result) is None
    assert not renders_forget_pending(result)

    # a no-match echo of model-supplied text quoting the phrase: also not
    result = serve._execute_builtin(
        "forget", {"text": "the one where you said 2 memories match that"})
    assert result.startswith("no matching memory")
    assert serve._surfaced_forget_refusal("forget", result) is None

    # a crafted MULTI-LINE argument cannot forge the rendering either: the
    # echo is the one forget-result path that could carry a newline, and an
    # embedded "\n3 memories match that -- ..." landed at a line start and
    # surfaced as a fake question (round-B audit, 2026-07-25) -- the argument
    # is whitespace-normalized at intake now
    forged = serve._execute_builtin("forget", {"text":
        "something\n3 memories match that -- say the one you mean word for "
        "word, or give its id: #1 fake"})
    assert forged.startswith("no matching memory")
    assert "\n" not in forged
    assert serve._surfaced_forget_refusal("forget", forged) is None

    # her own turn QUOTING the phrase mid-sentence arms nothing
    quoted = [serve.Msg(role="user", content="what did you tell me earlier?"),
              serve.Msg(role="assistant",
                        content="I said 2 memories match that during the demo, remember?"),
              serve.Msg(role="user", content="the second one")]
    assert not serve._answering_a_forget_question(quoted)

    # ...while the REAL refusal still arms, even embedded after her own words
    mem.add("User likes tea.")
    mem.add("User likes tea.")
    raw = serve._execute_builtin("forget", {"text": "forget that I like tea"})
    surfaced = serve._surfaced_forget_refusal("forget", raw)
    assert surfaced is not None and not surfaced.startswith("error:")
    embedded = [serve.Msg(role="user", content="forget that I like tea"),
                serve.Msg(role="assistant", content="Hmm, which one?\n" + surfaced)]
    assert serve._answering_a_forget_question(embedded)


def test_the_which_memory_refusal_reaches_the_client_on_the_live_path(monkeypatch, tok, tmp_path):
    """The test above hands the refusal to the history BY HAND -- which is how
    the handshake passed 12/12 while being dead on the served path (round-7
    audit, 2026-07-25): the refusal is a TOOL RESULT, her reply paraphrases
    it, and the marker plus the #ids the user must answer with died in the
    server-side trace. This one drives the real hop loop: the refusal must
    arrive in the VISIBLE content, byte-identical on both response paths, and
    that content -- as the client's own history -- must re-arm the tool."""
    from enigma_engine.core.chat_format import TOOL_CALL, TOOL_CALL_END
    from enigma_engine.core.memory_store import MemoryStore

    mem = MemoryStore(tmp_path)
    mem.add("User likes tea.")
    mem.add("User likes tea.")
    monkeypatch.setattr(serve, "tokenizer", tok)
    monkeypatch.setattr(serve, "EOS_ID", tok.eos_token_id)
    monkeypatch.setattr(serve, "BOS_ID", tok.bos_token_id)
    monkeypatch.setattr(serve, "ARGS", SimpleNamespace(max_context=512))
    monkeypatch.setattr(serve, "MEMORY", mem)
    monkeypatch.setattr(serve, "SPEAKER", None)
    monkeypatch.setattr(serve, "PAINTER", None)

    payload = json.dumps({"name": "forget", "arguments": {"text": "forget that I like tea"}})
    hops = [[TOOL_CALL] + tok.encode(payload, add_special_tokens=False) + [TOOL_CALL_END],
            tok.encode("Which one did you mean?", add_special_tokens=False)]

    def arm_generator():
        seq = iter(hops)

        def fake_gen(ids, max_tokens, *a, **k):
            yield from next(seq, hops[-1])

        monkeypatch.setattr(serve, "_gen_ids", fake_gen)

    def req(stream: bool) -> serve.ChatReq:
        return serve.ChatReq(
            messages=[serve.Msg(role="user", content="forget that I like tea")],
            stream=stream, max_tokens=64)

    arm_generator()
    resp = serve._chat_instruct(req(stream=False))
    content = resp["choices"][0]["message"]["content"]
    from enigma_engine.core.memory_store import renders_forget_pending
    assert renders_forget_pending(content), "the refusal never reached the client"
    assert "#1" in content and "#2" in content, "ids are what make the question answerable"
    assert "Which one did you mean?" in content, "her own words must ride along"
    assert "error:" not in content, "a question to the user is not an error line"
    assert len(mem.all()) == 2, "an ambiguous ask must not delete"

    # Byte parity: the TooBroad refusal deletes nothing, so a second run over
    # the same store must produce the identical refusal on the stream path.
    arm_generator()
    assert _drain_stream(serve._chat_instruct(req(stream=True))) == content

    # The surfaced content IS the client's next history turn; it must re-arm.
    follow = [serve.Msg(role="user", content="forget that I like tea"),
              serve.Msg(role="assistant", content=content),
              serve.Msg(role="user", content="the second one")]
    names = [t["function"]["name"] for t in serve._builtin_tools("the second one", False, follow)]
    assert "forget" in names


def test_an_id_cannot_reach_a_memory_the_wording_never_named(monkeypatch, tmp_path):
    """As a door of its own the id deleted whatever record happened to hold it,
    overriding a perfectly good `text`. She has no honest source for an id
    outside a refusal she just received, so every other id is invented -- and an
    invented one was destroying real memories."""
    from enigma_engine.core.memory_store import MemoryStore

    mem = MemoryStore(tmp_path)
    for text in ("User likes tea.", "User's therapist is named Dr Alvarez.",
                 "User's dog is named Rex."):
        mem.add(text)
    monkeypatch.setattr(serve, "MEMORY", mem)

    # an id that is not among the text's matches deletes nothing
    out = serve._execute_builtin("forget", {"text": "forget that I like tea", "id": 2})
    assert out.startswith("error:") and "not one of" in out
    assert len(mem.all()) == 3

    # an id with no wording at all is refused outright
    assert serve._execute_builtin("forget", {"id": 3}).startswith("error: give the wording")
    assert len(mem.all()) == 3

    # int() is not a validator: other scripts' digits and "3_0" are not ids
    for junk in ("3_0", "٣", "３", " 3.0", "1e3", "0x3"):
        assert serve._execute_builtin("forget", {"text": "x", "id": junk}).startswith(
            "error: memory id must be"
        ), junk
    assert len(mem.all()) == 3


def test_a_failed_memory_write_is_reported_as_text_not_a_500(monkeypatch, tmp_path):
    """Built-ins answer errors as text so the model can say what happened. A
    failed rewrite escaped as an exception, 500ing mid-conversation and leaving
    the in-memory store and the file disagreeing until the next boot."""
    from enigma_engine.core import memory_store as ms

    mem = ms.MemoryStore(tmp_path)
    mem.add("User likes tea.")
    monkeypatch.setattr(serve, "MEMORY", mem)
    monkeypatch.setattr(ms, "atomic_write_text",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))

    out = serve._execute_builtin("forget", {"text": "forget that I like tea"})
    assert out.startswith("error: could not update the memory file")
    assert "disk full" in out


def test_talking_about_forgetting_is_not_asking_her_to_forget(monkeypatch):
    """The widened verb's premise is that a false OFFER is cheap, which only
    holds while the tool cannot destroy something on its own -- and the store
    deletes on a single coincidental match. These are statements and questions
    ABOUT forgetting, so they suppress the offer; an imperative has no subject
    and stays armed."""
    monkeypatch.setattr(serve, "MEMORY", object())
    for text in ("I forget where I put my keys",
                 "Did you forget my name?",
                 "Do you forget things often?",
                 "I always forget my umbrella",
                 "I forgot to call her",
                 "I've forgotten his birthday"):
        assert not serve._looks_forgettable(text), text
    # the real asks are untouched
    for text in ("forget that I like tea", "forget I'm tall", "forget where I live",
                 "please forget everything about my dog"):
        assert serve._looks_forgettable(text), text


def test_a_polite_delete_request_is_still_a_delete_request(monkeypatch):
    """The AUXILIARY separates the two shapes: "did you forget" asks about the
    act, "could you forget" is an imperative wearing a question mark. A bare
    "(you) forget" suppressor matched inside the second, disarming a real ask --
    and since remember is gated on `not forgettable`, "could you forget that I
    like tea" then offered to SAVE it. That is the save-instead-of-delete
    inversion the negation guard exists to prevent, arriving through the
    suppressor."""
    monkeypatch.setattr(serve, "MEMORY", object())
    for text in ("can you forget my address",
                 "could you forget that I like tea",
                 "will you forget my old number",
                 "would you forget where I live",
                 "please can you forget my birthday"):
        assert serve._looks_forgettable(text), text
        assert not serve._looks_memorable(text), f"{text} offered to SAVE instead"


def test_the_forget_gate_offers_the_shapes_the_store_can_handle(monkeypatch):
    monkeypatch.setattr(serve, "MEMORY", object())
    # The store's one-candidate rule deletes all of these correctly, but the
    # gate's word list never offered the tool, so the store was never reached
    # and the capability was dead from chat. A miss is a capability she can
    # never learn to use; a false offer is one tool she declines.
    for text in ("forget I'm tall",
                 "please forget everything about my dog",
                 "forget where I live",
                 "forget my dog's name"):
        assert serve._looks_forgettable(text), text


def test_memorable_gate_stays_quiet_on_have_idioms(monkeypatch):
    monkeypatch.setattr(serve, "MEMORY", object())
    assert not serve._looks_memorable("I have no idea.")
    assert not serve._looks_memorable("I have to go now.")
    # ...while NEGATIVE possessions stay save-worthy: excluding the whole "no"
    # family to catch one idiom threw these away.
    assert serve._looks_memorable("I have no siblings.")
    assert serve._looks_memorable("I have no allergies.")
    assert serve._looks_memorable("I have two cats.")


def test_correction_cues_survive_the_smalltalk_they_resemble(monkeypatch):
    """Two attempts to pattern-exclude "Actually, it is a great question." each
    took real corrections with them, so the smalltalk is an ACCEPTED false
    positive: it costs one unnecessary tool offer, while excluding it cost the
    user's correction. Pin the corrections, not the exclusion."""
    monkeypatch.setattr(serve, "MEMORY", object())
    for text in ("Actually, it's a Toyota not a Honda.",
                 "No, it's a Toyota not a Honda.",
                 "Actually, it's the blue one, not the red.",
                 "Actually, that's a mistake, the name is Samantha.",
                 "Actually, my dog is Bruno now."):
        assert serve._looks_memorable(text), text


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
    """The offering, for EVERY lineage: intent-gated, per request.

    The always-offered block that would have replaced this (ruled 2026-07-24)
    was executed and reverted 2026-08-20 -- it lost the sealed gate twice, so
    the gates are the live rule again and these are their live pins, not one
    lineage's carve-out."""
    monkeypatch.setattr(serve, "MEMORY", None)
    monkeypatch.setattr(serve, "SPEAKER", None)
    monkeypatch.setattr(serve, "PAINTER", None)
    # calculate rides along whenever client tools exist (client_mode=True)
    assert serve._builtin_tools("Check the weather in Toronto", True) == [serve._CALC_TOOL]
    # ...but intent-gated otherwise
    assert serve._builtin_tools("Check the weather in Toronto", False) == []
    assert serve._builtin_tools("What is 7 * 8?", False) == [serve._CALC_TOOL]
    # The gates' KNOWN misses and false fires, pinned as the live behavior they
    # are: a word-number ask reaches no calculator, a negated draw still arms
    # the painter. These are what the retirement was for, and they are what a
    # future re-flip has to beat on the sealed set -- not in a docstring.
    monkeypatch.setattr(serve, "PAINTER", object())
    assert serve._builtin_tools("seven times eight", False) == []
    assert serve._builtin_tools("Don't draw me a picture, just describe it.", False) == [
        serve._IMAGINE_TOOL]
    # remember is intent-gated ALWAYS (it stole tool calls when ever-present)
    monkeypatch.setattr(serve, "MEMORY", object())
    offered = serve._builtin_tools("Check the weather in Toronto", True)
    assert serve._REMEMBER_TOOL not in offered


def test_the_offering_rule_is_stamped_and_reads_intent_gated_everywhere(monkeypatch):
    """The measurement-condition stamp, and the PARKED state it records.

    Two sealed transcripts scored under different offering regimes are not
    comparable, and the header could not say which one answered. The key
    survived the 2026-08-20 revert on purpose: it reads "intent-gated" for
    every lineage today, so a future re-flip that forgets to move it fails
    here rather than filing a block-regime run against a gated baseline.

    The lineage globals are varied precisely to pin that the offering does NOT
    key on them any more."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(serve, "_BOOTED", True)
    monkeypatch.setattr(serve, "MEMORY", None)
    monkeypatch.setattr(serve, "SPEAKER", None)
    monkeypatch.setattr(serve, "PAINTER", None)
    client = TestClient(serve.app)

    for vocab, instruct in ((V2_VOCAB, True), (V8_VOCAB, True), (V2_VOCAB, False)):
        _lineage(monkeypatch, vocab, instruct=instruct)
        caps = client.get("/v1/capabilities").json()
        assert caps["builtin_offering"] == "intent-gated", (vocab, instruct)
        # ...and the offering itself agrees: a bare ask arms nothing, whatever
        # the checkpoint
        assert serve._builtin_tools("hello", False) == []
        assert serve._builtin_tools("What is 7 * 8?", False) == [serve._CALC_TOOL]
    # a BASE checkpoint reaches no tool loop at all: empty builtins beside the
    # same stamp
    assert client.get("/v1/capabilities").json()["builtins"] == []


def test_capabilities_reports_availability_not_the_offering(monkeypatch, tmp_path):
    """`builtins` answers "what can this server EXECUTE", which is a different
    question from what she is offered on a given turn (that is _builtin_tools,
    gated on intent). One owner for the organ filter -- _available_builtins --
    so a report and a runtime cannot drift.

    The two single-organ rows are what makes the WIRING load-bearing: with
    voice and image-gen only ever up or down together, keying speak on PAINTER
    (or imagine on SPEAKER) reports the same list either way and every row
    passes a crossed filter."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(serve, "INSTRUCT", True)
    monkeypatch.setattr(serve, "_BOOTED", True)
    client = TestClient(serve.app)
    # a REAL store: the endpoint resolves its dir, so a stand-in object 500s
    store = MemoryStore(str(tmp_path / "mem"))
    for memory, speaker, painter, reported in (
        (None, None, None, ["calculate"]),
        (store, None, None, ["calculate", "forget", "remember"]),
        (None, object(), None, ["calculate", "speak"]),
        (None, None, object(), ["calculate", "imagine"]),
        (None, object(), object(), ["calculate", "imagine", "speak"]),
        (store, object(), object(),
         ["calculate", "forget", "imagine", "remember", "speak"]),
    ):
        monkeypatch.setattr(serve, "MEMORY", memory)
        monkeypatch.setattr(serve, "SPEAKER", speaker)
        monkeypatch.setattr(serve, "PAINTER", painter)
        assert client.get("/v1/capabilities").json()["builtins"] == reported
        # available does not mean offered: a bare "hello" is offered none of it
        assert serve._builtin_tools("hello", False) == []


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

    mem_block = store.render_context(user, tok, max_ids=128, k=serve.MEMORY_RECALL)
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

    mem_block = store.render_context(user, tok, max_ids=128, k=serve.MEMORY_RECALL)
    tools_block = render_tools_system([serve._CALC_TOOL])
    assert out[0]["content"] == "Client system." + "\n\n" + mem_block + "\n\n" + tools_block
    assert "You are Enigma" not in out[0]["content"]


def test_injected_memory_honours_the_recall_setting(monkeypatch, tok, tmp_path):
    """--memory-recall decides how many facts reach her context. The token
    budget trims further; here it is nowhere near binding, so the count is
    the setting."""
    store = MemoryStore(str(tmp_path / "mem"))
    for food in ("peanuts", "shellfish", "kiwi", "sesame", "walnuts"):
        store.add(f"User is allergic to {food}.")
    monkeypatch.setattr(serve, "MEMORY", store)
    monkeypatch.setattr(serve, "tokenizer", tok)
    monkeypatch.setattr(serve, "SPEAKER", None)
    monkeypatch.setattr(serve, "PAINTER", None)

    def recalled() -> int:
        req = serve.ChatReq(messages=[serve.Msg(role="user", content="What am I allergic to?")])
        out = serve._with_context([m.model_dump(exclude_none=True) for m in req.messages], req)
        text = "\n".join(m["content"] for m in out if m["role"] == "system")
        return sum(1 for line in text.splitlines() if line.startswith("- User is allergic"))

    monkeypatch.setattr(serve, "MEMORY_RECALL", 3)
    assert recalled() == 3
    monkeypatch.setattr(serve, "MEMORY_RECALL", 5)
    assert recalled() == 5
    monkeypatch.setattr(serve, "MEMORY_RECALL", 0)
    assert recalled() == 0


def test_the_shipped_recall_default_is_five():
    """How many memories reach her context is a user decision, so the shipped
    number is pinned rather than tuned. The flag and the module constant have
    to carry the same value: `boot()` sets the constant from the flag, and an
    imported-but-unbooted server answers from the constant alone."""
    assert serve._p.get_default("memory_recall") == 5
    assert serve.MEMORY_RECALL == 5


def test_memory_list_refuses_a_negative_k(monkeypatch, tmp_path):
    """A negative k would slice from the END and hand back every record but
    the lowest-ranked -- the opposite of the narrow ask it looks like."""
    store = MemoryStore(str(tmp_path / "mem"))
    for pet in ("Rex", "Bubbles", "Milo"):
        store.add(f"User's pet is named {pet}.")
    monkeypatch.setattr(serve, "MEMORY", store)

    with pytest.raises(serve.HTTPException) as exc:
        serve.memory_list(q="pet", k=-1)
    assert exc.value.status_code == 400
    assert serve.memory_list(q="pet", k=0)["results"] == []
    assert serve.memory_list(q=None, k=0)["results"] == []
    assert len(serve.memory_list(q="pet", k=2)["results"]) == 2


def test_memory_post_writes_the_same_kind_as_the_tool_door(monkeypatch, tmp_path):
    """The endpoint's docstring claims it mirrors the remember TOOL, which
    writes kind "user_fact" -- but its default was "fact", so the same fact
    landed under two kinds depending on which door it came through."""
    monkeypatch.setattr(serve, "MEMORY", MemoryStore(str(tmp_path / "mem")))
    posted = serve.memory_add(serve.MemReq(text="User likes tea."))["memory"]
    tooled = serve._execute_builtin("remember", {"text": "User likes coffee."})
    assert tooled.startswith("saved: ")
    assert posted["kind"] == serve.MEMORY.all()[-1]["kind"] == "user_fact"


def test_memory_post_refuses_a_reserved_kind(monkeypatch, tmp_path):
    """kind "episode" is the store's session-memory reserve: a client minting
    one through the HTTP door would forge a session summary into recall."""
    monkeypatch.setattr(serve, "MEMORY", MemoryStore(str(tmp_path / "mem")))
    for bad in ("episode", "note", ""):
        with pytest.raises(serve.HTTPException) as exc:
            serve.memory_add(serve.MemReq(text="User likes tea.", kind=bad))
        assert exc.value.status_code == 400
        assert "user_fact" in exc.value.detail and "fact" in exc.value.detail
    assert serve.MEMORY.all() == []  # nothing written by a refused kind
    assert serve.memory_add(serve.MemReq(text="User likes tea.", kind="fact"))["memory"]["kind"] == "fact"


def test_an_oversized_memory_is_refused_at_both_doors(monkeypatch, tmp_path):
    """Neither door capped the text, so a pasted megabyte was filed as one
    "fact": re-tokenized on every retrieval and echoed into a context that
    cannot hold it. The STORE owns the cap (one owner, both doors), and each
    door has to surface its refusal in its own language -- a 400 on the HTTP
    side, a tool RESULT she can answer around on the chat side, never a 500."""
    from enigma_engine.core.memory_store import MAX_MEMORY_CHARS

    monkeypatch.setattr(serve, "MEMORY", MemoryStore(str(tmp_path / "mem")))
    too_long = "x" * (MAX_MEMORY_CHARS + 1)

    with pytest.raises(serve.HTTPException) as exc:
        serve.memory_add(serve.MemReq(text=too_long))
    assert exc.value.status_code == 400 and "too long" in exc.value.detail

    tooled = serve._execute_builtin("remember", {"text": too_long})
    assert tooled.startswith("error:") and "too long" in tooled
    assert serve.MEMORY.all() == []  # nothing filed by either refusal

    assert serve.memory_add(serve.MemReq(text="User likes tea."))["ok"]
    assert serve._execute_builtin(
        "remember", {"text": "User likes coffee."}
    ).startswith("saved: ")
    assert len(serve.MEMORY.all()) == 2


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
# ...and parity across a MULTI-BYTE character (audit 2026-08-22). A character
# whose UTF-8 bytes span several tokens decodes to U+FFFD until its last byte
# lands, and the finished character is the same string LENGTH -- so the
# streamer emitted the U+FFFD and no later delta ever corrected it. Measured
# with the real vocab: ids [11881,166,142] streamed "Nice one \ufffd" where
# non-stream returned the emoji.
# ---------------------------------------------------------------------------

# The three ids the audit measured: one emoji, three tokens.
EMOJI_IDS = [11881, 166, 142]


def _scripted_stream_parity(monkeypatch, tokenizer, scripted: list[int]) -> tuple[str, str]:
    """Drive one request both ways over a scripted id stream; return
    (non-stream content, streamed content)."""
    monkeypatch.setattr(serve, "tokenizer", tokenizer)
    monkeypatch.setattr(serve, "EOS_ID", tokenizer.eos_token_id)
    monkeypatch.setattr(serve, "BOS_ID", tokenizer.bos_token_id)
    monkeypatch.setattr(serve, "ARGS", SimpleNamespace(max_context=512))
    monkeypatch.setattr(serve, "MEMORY", None)
    monkeypatch.setattr(serve, "SPEAKER", None)
    monkeypatch.setattr(serve, "PAINTER", None)

    def fake_gen(ids, max_tokens, *a, **k):
        yield from scripted

    monkeypatch.setattr(serve, "_gen_ids", fake_gen)

    def _req(stream: bool) -> serve.ChatReq:
        return serve.ChatReq(
            messages=[serve.Msg(role="user", content="Tell me a story.")],
            stream=stream,
            max_tokens=64,
        )

    nonstream = serve._chat_instruct(_req(stream=False))["choices"][0]["message"]["content"] or ""
    return nonstream, _drain_stream(serve._chat_instruct(_req(stream=True)))


@pytest.mark.parametrize("trailing", ["", " ok"], ids=["terminal", "mid-text"])
def test_a_multi_byte_character_streams_as_itself(monkeypatch, tok_v2, trailing):
    """Terminal AND mid-text: the terminal case is the one an end-of-stream
    flush is needed for, the mid-text case the one a later delta must not
    double-count."""
    scripted = (
        tok_v2.encode("Nice one ", add_special_tokens=False)
        + EMOJI_IDS
        + (tok_v2.encode(trailing, add_special_tokens=False) if trailing else [])
    )
    nonstream, stream = _scripted_stream_parity(monkeypatch, tok_v2, scripted)
    assert "\ufffd" not in nonstream  # the control: non-stream was always right
    assert stream == nonstream


def test_a_genuine_replacement_character_still_reaches_the_wire(monkeypatch, tok_v2):
    """Holding U+FFFD back must not DROP one she actually produced: the
    end-of-stream flush releases whatever is still held."""
    genuine = tok_v2.encode("\ufffd", add_special_tokens=False)
    assert tok_v2.decode(genuine, skip_special_tokens=True) == "\ufffd"
    scripted = tok_v2.encode("look ", add_special_tokens=False) + genuine
    nonstream, stream = _scripted_stream_parity(monkeypatch, tok_v2, scripted)
    assert nonstream.endswith("\ufffd")
    assert stream == nonstream


# ---------------------------------------------------------------------------
# ...and parity when the SPANS are malformed (audit 2026-08-22). The stream
# used to suppress span interiors with a DEPTH COUNTER, which is not what
# parse_assistant_ids does: spans never nest there, an opener TRUNCATES the
# open span and restarts, a closer closes outright, and a span the token
# budget cut is flushed rather than discarded. Every one of those shapes
# streamed different bytes than the same ids returned non-streamed.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tok_v2():
    """The live lineage's vocab -- the only one carrying <search> ids, which
    the budget-cut shape below needs (v1 carves none)."""
    t = get_tokenizer("bpe", vocab_path=vocab_file_for_size(V2_VOCAB))
    attach_chat_tokens(t)
    return t


def _span_shape_ids(tok, shape: str) -> list[int]:
    """One shape's SAMPLED ids, built with the real tokenizer -- both paths are
    then driven over this exact sequence, which is the only way the comparison
    measures the two readers rather than two generations."""
    ct = chat_token_ids(tok)
    think, think_end = think_token_ids(tok)
    tool, tool_end = ct["<|tool_call|>"], ct["<|/tool_call|>"]
    search, _ = search_token_ids(tok)

    def e(text):
        return tok.encode(text, add_special_tokens=False)

    if shape == "well-formed":  # the control: <think>...</think> then content
        return [think, *e("reasoning"), think_end, *e("Hello world.")]
    if shape == "reopened-think":  # <think>a<think>b</think>c
        return [think, *e("a"), think, *e("b"), think_end, *e("c")]
    if shape == "tool-inside-think":  # <think>a<tool_call>{..}</tool_call> c</think> d
        return [think, *e("a"), tool, *e('{"name": "get_weather", "arguments": {}}'),
                tool_end, *e(" c"), think_end, *e(" d")]
    if shape == "budget-cut-span":  # generation ended mid-span: the parser recovers it
        return [*e("before "), search, *e("half a query")]
    raise AssertionError(f"unknown shape {shape!r}")


@pytest.mark.parametrize(
    "shape,expected",
    [
        ("well-formed", "Hello world."),
        # the depth counter never came back down: "c" streamed as nothing
        ("reopened-think", "c"),
        # the inner closer decremented to depth 1, so " c" was suppressed
        ("tool-inside-think", "c d"),
        # the dangling span's text joins CONTENT at parse time; the stream had
        # no catch-up for it (only raw tool text had one)
        ("budget-cut-span", "before half a query"),
    ],
)
def test_stream_matches_nonstream_on_malformed_spans(monkeypatch, tok_v2, shape, expected):
    monkeypatch.setattr(serve, "INSTRUCT", True)
    monkeypatch.setattr(serve, "tokenizer", tok_v2)
    monkeypatch.setattr(serve, "EOS_ID", tok_v2.eos_token_id)
    monkeypatch.setattr(serve, "BOS_ID", tok_v2.bos_token_id)
    monkeypatch.setattr(serve, "ARGS", SimpleNamespace(max_context=512, search_k=3))
    monkeypatch.setattr(serve, "MEMORY", None)
    monkeypatch.setattr(serve, "SPEAKER", None)
    monkeypatch.setattr(serve, "PAINTER", None)
    monkeypatch.setattr(serve, "SEARCHER", None)
    scripted = _span_shape_ids(tok_v2, shape)

    def fake_gen(ids, max_tokens, *a, **k):
        yield from scripted

    monkeypatch.setattr(serve, "_gen_ids", fake_gen)

    def _req(stream: bool) -> serve.ChatReq:
        return serve.ChatReq(
            messages=[serve.Msg(role="user", content="Tell me a story.")],
            stream=stream,
            max_tokens=128,
        )

    nonstream = serve._chat_instruct(_req(stream=False))["choices"][0]["message"]["content"] or ""
    stream = _drain_stream(serve._chat_instruct(_req(stream=True)))
    # The literal is pinned as well as the equality: two paths agreeing on ""
    # would satisfy `stream == nonstream` while dropping the whole reply.
    assert nonstream == expected
    assert stream == nonstream
    # Span INTERIORS still never reach the wire -- parity is not permission to
    # leak her reasoning or the query she is about to run.
    for hidden in ("reasoning", "get_weather", "half a query"):
        if hidden in expected:
            continue
        assert hidden not in stream, f"{hidden!r} leaked onto the wire"


def test_stream_reports_the_same_enigma_extension_as_nonstream(monkeypatch, tok):
    """A looped built-in is consumed by the hop that runs it, so `enigma`
    (spoke + tools_run) is the ONLY report that it fired. The stream path
    carried no such report at all: an eval or a page driving the server with
    stream=true could not tell a speak that fired from one that never did,
    and the page would double-voice a reply she already said out loud."""
    from enigma_engine.core.chat_format import TOOL_CALL, TOOL_CALL_END

    class _Speaker:
        last_error = None

        def speak(self, text):
            return None

    monkeypatch.setattr(serve, "tokenizer", tok)
    monkeypatch.setattr(serve, "EOS_ID", tok.eos_token_id)
    monkeypatch.setattr(serve, "BOS_ID", tok.bos_token_id)
    monkeypatch.setattr(serve, "ARGS", SimpleNamespace(max_context=512))
    monkeypatch.setattr(serve, "MEMORY", None)
    monkeypatch.setattr(serve, "PAINTER", None)
    monkeypatch.setattr(serve, "MUTED", False)
    monkeypatch.setattr(serve, "SPEAKER", _Speaker())

    payload = json.dumps({"name": "speak", "arguments": {"text": "Hello there."}})
    hops = [[TOOL_CALL] + tok.encode(payload, add_special_tokens=False) + [TOOL_CALL_END],
            tok.encode("Said it out loud.", add_special_tokens=False)]
    seq = {"it": iter(hops)}

    def fake_gen(ids, max_tokens, *a, **k):
        yield from next(seq["it"], hops[-1])

    monkeypatch.setattr(serve, "_gen_ids", fake_gen)

    def _req(stream: bool) -> serve.ChatReq:
        return serve.ChatReq(
            messages=[serve.Msg(role="user", content="Say hello out loud.")],
            stream=stream,
            max_tokens=64,
        )

    seq["it"] = iter(hops)
    nonstream = serve._chat_instruct(_req(stream=False))
    assert nonstream["enigma"] == {"spoke": True, "tools_run": ["speak"]}

    seq["it"] = iter(hops)
    chunks = _stream_chunks(serve._chat_instruct(_req(stream=True)))
    terminal = chunks[-1]
    assert terminal["choices"][0]["finish_reason"] is not None  # the frame before [DONE]
    assert terminal["enigma"] == nonstream["enigma"]
    # content frames stay byte-identical to the non-stream join: the extension
    # rides the terminal frame ONLY
    assert not any("enigma" in c for c in chunks[:-1])


def _drive_chat(monkeypatch, tok) -> tuple[dict, str, list[dict]]:
    """One chat request answered both ways: (non-stream payload, the raw SSE
    text, its parsed frames). Generation is scripted, so the only thing that
    can differ between two calls of this is the identity being served."""
    monkeypatch.setattr(serve, "tokenizer", tok)
    monkeypatch.setattr(serve, "EOS_ID", tok.eos_token_id)
    monkeypatch.setattr(serve, "BOS_ID", tok.bos_token_id)
    monkeypatch.setattr(serve, "ARGS", SimpleNamespace(max_context=512))
    monkeypatch.setattr(serve, "MEMORY", None)
    monkeypatch.setattr(serve, "SPEAKER", None)
    monkeypatch.setattr(serve, "PAINTER", None)
    scripted = tok.encode("Hello world.", add_special_tokens=False)

    def fake_gen(ids, max_tokens, *a, **k):
        yield from scripted

    monkeypatch.setattr(serve, "_gen_ids", fake_gen)

    def _req(stream: bool) -> serve.ChatReq:
        return serve.ChatReq(
            messages=[serve.Msg(role="user", content="Tell me a story.")],
            stream=stream,
            max_tokens=64,
        )

    nonstream = serve._chat_instruct(_req(stream=False))
    resp = serve._chat_instruct(_req(stream=True))

    async def _drain():
        return [c if isinstance(c, str) else c.decode("utf-8") async for c in resp.body_iterator]

    raw = "".join(asyncio.run(_drain()))
    frames = [json.loads(line[6:]) for line in raw.splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    return nonstream, raw, frames


def _request_model_defaults() -> list[str]:
    """The `model` a client that omitted the field gets on every request shape
    that has one. Class-body defaults are frozen at IMPORT -- before boot()
    has read a pack -- which is why these are built from a factory."""
    return [
        serve.ChatReq(messages=[]).model,
        serve.CompletionReq(prompt="x").model,
        serve.SpeechReq(input="hello").model,
        serve.ImageGenReq(prompt="a cat").model,
    ]


def test_the_default_model_surface_is_still_hers_byte_for_byte(monkeypatch, tok):
    """Her slug IS "enigma" -- the literal every echo site carried -- so
    deriving the id from the persona may not move one byte of her payloads.
    The raw SSE text is asserted, not just the parsed frames: the id rides
    the wire inside the JSON, and a derivation that renders it differently
    (spaced, quoted, uppercased) parses the same and serves different bytes."""
    nonstream, raw, frames = _drive_chat(monkeypatch, tok)
    assert nonstream["model"] == "enigma"
    assert frames, "the stream produced no frames"
    assert [f["model"] for f in frames] == ["enigma"] * len(frames)
    assert '"model": "enigma"' in raw
    assert raw.count('"model":') == len(frames)  # every frame names it
    assert _request_model_defaults() == ["enigma"] * 4


def test_a_pack_echoes_its_own_id_through_every_frame(monkeypatch, tok):
    """The hole this closes: /v1/models published `atlas` while every
    completion and every streaming chunk the same server sent echoed
    `"model": "enigma"` -- an OpenAI client asking which model answered was
    told Enigma had, by the AI that had not."""
    monkeypatch.setattr(serve, "PERSONA", Persona(name="Atlas", data_dirname=".atlas"))
    nonstream, raw, frames = _drive_chat(monkeypatch, tok)
    assert nonstream["model"] == "atlas"
    assert frames, "the stream produced no frames"
    assert [f["model"] for f in frames] == ["atlas"] * len(frames)
    assert "enigma" not in raw
    assert _request_model_defaults() == ["atlas"] * 4
    assert serve.list_models()["data"][0]["id"] == "atlas"  # and the one endpoint 3a moved


# ---------------------------------------------------------------------------
# the opening SSE frame: OpenAI's role chunk, and what it must not disturb
# ---------------------------------------------------------------------------


def _stream_chunks(resp) -> list[dict]:
    async def _drain():
        frames = []
        async for chunk in resp.body_iterator:
            frames.append(chunk if isinstance(chunk, str) else chunk.decode("utf-8"))
        return frames

    out = []
    for frame in asyncio.run(_drain()):
        for line in frame.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                out.append(json.loads(line[6:]))
    return out


def _assert_opens_with_role(chunks: list[dict], body: str) -> None:
    """The first chunk declares the assistant role and carries no content key,
    so a client joining the content deltas still sees exactly `body`."""
    first = chunks[0]
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"] == {"role": "assistant"}
    assert first["choices"][0]["finish_reason"] is None
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == body


def test_instruct_stream_opens_with_the_assistant_role(monkeypatch, tok):
    monkeypatch.setattr(serve, "tokenizer", tok)
    monkeypatch.setattr(serve, "EOS_ID", tok.eos_token_id)
    monkeypatch.setattr(serve, "BOS_ID", tok.bos_token_id)
    monkeypatch.setattr(serve, "ARGS", SimpleNamespace(max_context=512))
    monkeypatch.setattr(serve, "MEMORY", None)
    monkeypatch.setattr(serve, "SPEAKER", None)
    monkeypatch.setattr(serve, "PAINTER", None)
    scripted = tok.encode("Hello world.", add_special_tokens=False)

    def fake_gen(ids, max_tokens, *a, **k):
        yield from scripted

    monkeypatch.setattr(serve, "_gen_ids", fake_gen)

    req = serve.ChatReq(
        messages=[serve.Msg(role="user", content="Tell me a story.")],
        stream=True,
        max_tokens=64,
    )
    _assert_opens_with_role(_stream_chunks(serve._chat_instruct(req)), "Hello world.")


def test_base_stream_opens_with_the_assistant_role(monkeypatch, tok):
    monkeypatch.setattr(serve, "INSTRUCT", False)
    monkeypatch.setattr(serve, "tokenizer", tok)
    monkeypatch.setattr(serve, "MEMORY", None)
    monkeypatch.setattr(serve, "EYES", None)
    monkeypatch.setattr(serve, "_STOP_TEXTS", ())

    def fake_text(*a, **k):
        yield from ("Hel", "lo.")

    monkeypatch.setattr(serve, "_generate_text", fake_text)

    req = serve.ChatReq(
        messages=[serve.Msg(role="user", content="Tell me a story.")],
        stream=True,
        max_tokens=64,
    )
    _assert_opens_with_role(_stream_chunks(serve.chat(req)), "Hello.")


# ---------------------------------------------------------------------------
# mute: roundtrip, persistence, 204, and the speak-tool gate
# ---------------------------------------------------------------------------


def test_mute_roundtrip_persists_and_gates(monkeypatch, tmp_path):
    state = tmp_path / "mute_state.json"
    monkeypatch.setattr(serve, "_MUTE_STATE", state)
    monkeypatch.setattr(serve, "MUTED", False)
    monkeypatch.setattr(serve, "SPEAKER", object())

    assert serve.set_mute(serve.MuteReq(muted=True)) == {"muted": True, "persisted": True}
    assert serve.MUTED is True
    assert json.loads(state.read_text(encoding="utf-8")) == {"muted": True}
    assert serve.get_mute() == {"muted": True}

    # muted speech endpoint: 204, no audio, no synthesis attempted
    resp = serve.audio_speech(serve.SpeechReq(input="hello"))
    assert resp.status_code == 204
    # muted speak TOOL: honest "muted:" result string, not an exception
    assert serve._execute_builtin("speak", {"text": "hi"}).startswith("muted:")

    assert serve.set_mute(serve.MuteReq(muted=False)) == {"muted": False, "persisted": True}
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
    assert serve.set_talk_mode(serve.TalkReq(enabled=True)) == {"enabled": True, "persisted": True}
    assert serve.TALK_MODE is True
    assert json.loads(state.read_text(encoding="utf-8")) == {"enabled": True}
    assert serve.get_talk_mode() == {"enabled": True}


def test_a_failed_state_write_is_reported_not_swallowed(monkeypatch, tmp_path, capsys):
    """Persistence is what the mute comment PROMISES (a crash-relaunch must
    not silently unmute a muted gaming session). A swallowed OSError answered
    200 claiming durability the disk never got, so the next boot would quietly
    hand back the old state. The switch still flips for this run -- it just
    stops lying about surviving a restart."""
    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(serve.os, "replace", _boom)
    monkeypatch.setattr(serve, "_MUTE_STATE", tmp_path / "mute_state.json")
    monkeypatch.setattr(serve, "_TALK_STATE", tmp_path / "talk_mode.json")
    monkeypatch.setattr(serve, "MUTED", False)
    monkeypatch.setattr(serve, "TALK_MODE", False)

    assert serve.set_mute(serve.MuteReq(muted=True)) == {"muted": True, "persisted": False}
    assert serve.MUTED is True  # in-memory truth still changed
    assert serve.set_talk_mode(serve.TalkReq(enabled=True)) == {"enabled": True, "persisted": False}
    assert serve.TALK_MODE is True

    out = capsys.readouterr().out
    assert out.count("WARN: could not persist") == 2
    assert "mute_state.json" in out and "talk_mode.json" in out
    assert "disk full" in out


def test_the_runtime_state_writers_use_the_shared_atomic_writer():
    """mute_state.json / talk_mode.json / voice.json are exactly what a
    crash-relaunch reads back. Both writers re-implemented temp-and-replace
    WITHOUT an fsync, while the repo already owned an fsync'd writer (the
    memory store uses it) -- so a power loss could commit the rename ahead of
    the data blocks and load the wrong default on the next boot, the case the
    mute-state comment promises against. One writer, not three."""
    import inspect

    from enigma_engine.core.tts import Speaker

    for fn in (serve._write_state_atomic, Speaker._persist):
        src = inspect.getsource(fn)
        assert "atomic_write_text(" in src, f"{fn.__qualname__} bypasses the shared writer"
        assert "os.replace" not in src, f"{fn.__qualname__} still hand-rolls the replace"


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


def test_a_timed_out_speech_request_leaves_no_orphan_wav(monkeypatch, tmp_path):
    """/v1/audio/speech unlinks its temp file when save_wav fails -- but the
    save job was still QUEUED, so the worker re-created the file after the
    request had been answered and cleaned up, and nothing ever removed it. The
    cancel-on-timeout rule closes it: a job whose caller was told it did not
    happen does not happen later."""
    import threading

    import enigma_engine.core.tts as tts_mod
    from enigma_engine.core.tts import Speaker

    class _Backend:
        sample_rate = 24000

        def __init__(self):
            self.synths: list[str] = []

        def synth(self, text):
            import numpy as np

            self.synths.append(text)
            return np.zeros(8, dtype=np.float32)

        def set_recipe(self, recipe):
            pass

    class _BlockingPlayer:
        """Holds the worker inside one utterance until it is released."""

        def __init__(self):
            self.playing = threading.Event()
            self._release = threading.Event()

        def play(self, audio, sample_rate, device=None):
            self.playing.set()

        def wait(self):
            self._release.wait(5)

        def stop(self):
            self._release.set()

    backend, player = _Backend(), _BlockingPlayer()
    spk = Speaker(synth_factory=lambda: backend, player_factory=lambda: player)
    made: list[str] = []
    real_mkstemp = serve.tempfile.mkstemp

    def _spy(*a, **kw):
        kw.setdefault("dir", str(tmp_path))
        fd, path = real_mkstemp(*a, **kw)
        made.append(path)
        return fd, path

    monkeypatch.setattr(serve.tempfile, "mkstemp", _spy)
    monkeypatch.setattr(serve, "SPEAKER", spk)
    monkeypatch.setattr(serve, "MUTED", False)
    try:
        spk.speak("pinning the worker.")
        assert player.playing.wait(5), "worker never reached playback"
        monkeypatch.setattr(tts_mod, "_JOB_TIMEOUT_S", 0.05)
        with pytest.raises(serve.HTTPException) as exc:
            serve.audio_speech(serve.SpeechReq(input="say this out loud"))
        assert exc.value.status_code == 500 and "timed out" in exc.value.detail
        assert made and not Path(made[0]).exists()  # the endpoint cleaned up

        monkeypatch.setattr(tts_mod, "_JOB_TIMEOUT_S", 5.0)
        player.stop()  # release the worker; it drains the queue in order
        spk.speak("after.", wait=True)  # returns only once everything ahead ran
        assert not Path(made[0]).exists(), "the abandoned save re-created the file"
        assert "say this out loud" not in backend.synths
    finally:
        spk.close()


# ---------------------------------------------------------------------------
# organ doors: whose fault an error is (ears), and what may reach the GPU
# ---------------------------------------------------------------------------


def test_a_junk_upload_is_the_clients_error_not_the_servers(monkeypatch):
    """Whisper choking on the bytes a CLIENT uploaded was reported as a 500 --
    telling every junk file it had broken the server, and burying the real
    organ failures in the same status. The raise site knows which is which."""
    from fastapi.testclient import TestClient

    from enigma_engine.core.asr import ASRError, Ears

    class _JunkAudio:
        def transcribe(self, path):
            raise RuntimeError("Invalid data found when processing input")

    class _BrokenOrgan:
        def transcribe(self, path):
            raise ASRError("could not load whisper 'base'")

    monkeypatch.setattr(serve, "_BOOTED", True)
    client = TestClient(serve.app)
    upload = {"file": ("clip.wav", b"not audio at all", "audio/wav")}

    monkeypatch.setattr(serve, "EARS", Ears(model_factory=_JunkAudio))
    assert client.post("/v1/audio/transcriptions", files=upload).status_code == 400

    # ...while an organ that is broken in ITSELF is still the server's problem.
    monkeypatch.setattr(serve, "EARS", _BrokenOrgan())
    assert client.post("/v1/audio/transcriptions", files=upload).status_code == 500


class _RecordingPainter:
    """Stands in for the diffusion organ: records the size it was asked for
    and writes a stub PNG where the endpoint expects to read one."""

    def __init__(self):
        self.sizes: list[tuple[int, int]] = []

    def generate(self, prompt, out_path, steps=None, width=512, height=512):
        self.sizes.append((width, height))
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x89PNG\r\n\x1a\n")
        return out


@pytest.mark.parametrize("size", [
    "4096x4096",   # the OOM ask: a CUDA OOM beside the served model fragments
    "2048x512",    # ...and one oversized side is enough to do it
    "0x0",
    "-512x-512",
    "512x0",
    "100x100",     # under the floor
    "513x512",     # not a multiple of 8: the SD VAE downsamples by 8
])
def test_an_unusable_image_size_never_reaches_the_gpu(monkeypatch, tmp_path, size):
    """n was bounded because VRAM is shared with the LLM; size was not, so
    "4096x4096", "0x0" and negatives went straight down to diffusers."""
    painter = _RecordingPainter()
    monkeypatch.setattr(serve, "PAINTER", painter)
    monkeypatch.setattr(serve, "IMAGES_DIR", tmp_path / "images")

    with pytest.raises(serve.HTTPException) as exc:
        serve.images_generations(serve.ImageGenReq(prompt="a cat", size=size))
    assert exc.value.status_code == 400
    assert "256" in exc.value.detail and "1024" in exc.value.detail
    assert painter.sizes == [], "a refused size still reached the painter"


@pytest.mark.parametrize("size,expected", [("512x512", (512, 512)), ("1024x768", (1024, 768))])
def test_a_usable_image_size_still_paints(monkeypatch, tmp_path, size, expected):
    painter = _RecordingPainter()
    monkeypatch.setattr(serve, "PAINTER", painter)
    monkeypatch.setattr(serve, "IMAGES_DIR", tmp_path / "images")

    out = serve.images_generations(serve.ImageGenReq(prompt="a cat", size=size))
    assert painter.sizes == [expected]
    assert len(out["data"]) == 1 and out["data"][0]["b64_json"]


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
# main()'s pre-boot port probe: refusing must be cheap
# ---------------------------------------------------------------------------


def test_a_busy_port_is_refused_before_the_checkpoint_load(monkeypatch):
    """boot() reads and sha256s a multi-GB .pth and brings the organs up, and
    only THEN does uvicorn try to bind -- so a second server aimed at the daily
    port paid the whole load just to die on the bind (audit 2026-08-22). The
    probe is a refusal, never a takeover: the live server keeps the port."""
    import socket

    held = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    held.bind(("127.0.0.1", 0))
    held.listen(1)
    port = held.getsockname()[1]
    try:
        with pytest.raises(SystemExit) as exc:
            serve._require_free_port("127.0.0.1", port)
        assert str(port) in str(exc.value)
        assert "Stop Enigma" in str(exc.value)  # names the likely owner and the way out
        # ...and the listener it refused for is untouched: still accepting.
        probe = socket.create_connection(("127.0.0.1", port), timeout=2)
        probe.close()
    finally:
        held.close()

    # A free port passes AND is left free -- the probe releases what it binds,
    # or uvicorn would bind against this very socket a moment later.
    free = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    free.bind(("127.0.0.1", 0))
    free_port = free.getsockname()[1]
    free.close()
    serve._require_free_port("127.0.0.1", free_port)
    serve._require_free_port("127.0.0.1", free_port)  # twice: nothing was kept


def test_a_multi_family_host_is_refused_when_any_family_is_busy():
    """getaddrinfo can hand back several families for one name, and uvicorn
    binds all of them. Probing only row 0 cleared a port whose OTHER family
    was already serving, so the second server paid the whole checkpoint load
    and died on the bind anyway -- the exact failure the probe exists to stop
    (measured 2026-08-22: on this box "localhost" is ::1 THEN 127.0.0.1).

    Expectations are DERIVED from getaddrinfo, so a machine where localhost
    resolves to one family only skips rather than fails."""
    import socket

    held = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    held.bind(("127.0.0.1", 0))
    held.listen(1)
    port = held.getsockname()[1]
    try:
        rows = socket.getaddrinfo("localhost", port, type=socket.SOCK_STREAM)
        if not any(row[0] is socket.AF_INET for row in rows):
            pytest.skip("localhost does not resolve to IPv4 on this machine")
        if rows[0][0] is socket.AF_INET:
            pytest.skip("localhost resolves IPv4 first here; row 0 already covered the busy family")

        with pytest.raises(SystemExit) as exc:
            serve._require_free_port("localhost", port)
        assert "127.0.0.1" in str(exc.value)  # names the address that was busy
        assert str(port) in str(exc.value)
        # ...and the listener it refused for is untouched: still accepting.
        probe = socket.create_connection(("127.0.0.1", port), timeout=2)
        probe.close()
    finally:
        held.close()

    # Every family released again: the same host clears once nothing holds it.
    serve._require_free_port("localhost", port)


def test_main_probes_the_port_before_it_boots(monkeypatch):
    """The ORDER is the whole fix: a probe that runs after boot() saves nothing.
    main() is driven with both steps stubbed, and the call sequence recorded."""
    calls: list[str] = []
    monkeypatch.setattr(serve, "_require_free_port", lambda h, p: calls.append(f"probe {h}:{p}"))
    monkeypatch.setattr(serve, "boot", lambda: calls.append("boot"))
    monkeypatch.setattr(serve.uvicorn, "run", lambda *a, **k: calls.append("uvicorn"))
    monkeypatch.setattr(serve, "ARGS", SimpleNamespace(host="127.0.0.1", port=8000))
    monkeypatch.setattr(serve.sys, "argv", ["serve_enigma.py", "--port", "8123"])

    serve.main()
    assert calls == ["probe 127.0.0.1:8123", "boot", "uvicorn"]


# ---------------------------------------------------------------------------
# boot() end to end on a tiny checkpoint (CPU-forced; every global restored)
# ---------------------------------------------------------------------------

# Every name boot() declares `global` -- a name boot writes but this list
# misses is never restored, so it leaks into every later test in the session.
_RUNTIME_GLOBALS = [
    "ARGS", "CONFIG", "model", "tokenizer", "DEVICE", "_BF16_GEN", "STEP", "META",
    "MODEL_PATH", "MODEL_SHA256",
    "INSTRUCT", "MEMORY", "MEMORY_RECALL", "SPEAKER", "MUTED", "TALK_MODE", "EARS",
    "EYES", "PAINTER", "SEARCHER", "EOS_ID", "BOS_ID",
    "_BOOTED", "PERSONA", "_VOICE_STATE", "IMAGES_DIR", "_STOP_TEXTS",
    "_MUTE_STATE", "_TALK_STATE",
]


def test_the_runtime_globals_list_mirrors_boots_own_global_statements():
    """The list above is a MIRROR of boot(), hand-maintained -- and nothing
    made it track. A global added to boot and forgotten here is written by
    every boot test and restored by none, so it leaks into the rest of the
    session as a checkpoint, a persona, or a state path from another test.
    The source is the authority; read the names out of it."""
    tree = ast.parse((Path(serve.__file__).read_text(encoding="utf-8")))
    boot = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "boot"
    )
    declared = {name for n in ast.walk(boot) if isinstance(n, ast.Global) for name in n.names}
    assert declared == set(_RUNTIME_GLOBALS)


_HF_ENV_KEYS = (
    "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
    "HF_HUB_DISABLE_TELEMETRY", "HF_HUB_DISABLE_IMPLICIT_TOKEN",
)


def _hermetic_state(monkeypatch, tmp_path):
    """Point everything boot() derives from PERSONA.home at tmp, and return it.

    Patching serve._MUTE_STATE / serve._TALK_STATE is no longer enough: boot
    REBINDS both from the persona's home, so an unpatched home would have a
    test run read -- and migrate the repo's data/mute_state.json into -- the
    developer's own ~/.enigma_engine. The home itself is the seam, so the
    legacy migration sources are pointed at absent tmp files too: a boot test
    must not depend on whether this checkout happens to carry one."""
    home = tmp_path / "home"
    monkeypatch.setattr(serve.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(serve, "_LEGACY_MUTE_STATE", tmp_path / "legacy" / "mute_state.json")
    monkeypatch.setattr(serve, "_LEGACY_TALK_STATE", tmp_path / "legacy" / "talk_mode.json")
    return home


def test_boot_tiny_checkpoint(monkeypatch, tmp_path):
    """The full startup path on a 2-layer toy model, SIX boots: the first
    exercises the --allow-downloads env branch AND the KV-cache clamp
    (--max-context 4096 vs max_seq_len 256 -- the 2026-07-17 version never
    entered either branch); the second, flagless boot must RESTORE the
    offline default despite the first boot's leftover "0" (the double-boot
    hole, re-audit 2026-07-18); legs C/D pin the operator-export semantics.
    CUDA is masked off so this never touches the GPU; the persona home and
    env are patched hermetic and restored (unpatched, boot() reads the
    developer's own ~/.enigma_engine/talk_mode.json)."""
    import os

    snapshot = {name: getattr(serve, name) for name in _RUNTIME_GLOBALS}
    monkeypatch.setattr(serve.torch.cuda, "is_available", lambda: False)
    _hermetic_state(monkeypatch, tmp_path)
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
        # /v1/models must say WHICH weights: two same-arch checkpoints have
        # identical keys, shapes, and steps -- only the file hash separates
        # them, and the eval transcript records it from here.
        import hashlib as _hashlib
        assert serve.MODEL_SHA256 == _hashlib.sha256(ckpt.read_bytes()).hexdigest()
        assert serve.MODEL_PATH == str(ckpt.resolve())
        _entry = serve.list_models()["data"][0]
        assert _entry["checkpoint"]["sha256"] == serve.MODEL_SHA256
        assert _entry["checkpoint"]["step"] == 7
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


def test_boot_declares_the_chat_specials_decodable(monkeypatch, tmp_path, capsys):
    """Sampling -inf's every logit past the live vocab. The chat/tool specials
    are registered in the FIRST alignment-padding rows and trained there, so a
    boot that does not declare them deletes the model's own `<|tool_call|>` and
    `<|im_end|>` from the distribution: measured on the live v8 checkpoint, a
    weather ask put p=0.997 on `<|tool_call|>` and the server answered with an
    empty string -- every tool, every built-in, every time.

    Vocab is the real one here on purpose: a toy vocab puts the chat ids past
    the head entirely, which is the OTHER branch (asserted below)."""
    from enigma_engine.core.chat_format import chat_token_ids

    snapshot = {name: getattr(serve, name) for name in _RUNTIME_GLOBALS}
    monkeypatch.setattr(serve.torch.cuda, "is_available", lambda: False)
    _hermetic_state(monkeypatch, tmp_path)
    real_vocab = len(get_tokenizer("bpe").token_to_id)
    cfg = ForgeConfig(
        vocab_size=real_vocab, dim=32, n_layers=2, n_heads=2,
        max_seq_len=256, dropout=0.0, use_gradient_checkpointing=False,
    )
    torch.manual_seed(0)
    ckpt = tmp_path / "real_vocab.pth"
    torch.save(
        {"model_state_dict": Enigma(cfg).state_dict(), "config": cfg.to_dict(),
         "meta": {"chat_format": CHAT_FORMAT_NAME}},
        ckpt,
    )
    try:
        serve.boot(argv=["--model", str(ckpt), "--max-context", "128"])
        expected = max(chat_token_ids(serve.tokenizer).values()) + 1
        assert serve.model.live_vocab_size == expected
        assert expected > serve.model.config.vocab_size, "specials must sit past the real vocab"
        # the padding BEYOND the specials stays masked -- the original guard holds
        step = torch.zeros(1, serve.model.output.weight.shape[0])
        masked = serve.model._live_vocab_logits(step)
        assert torch.isfinite(masked[0, :expected]).all()
        assert torch.isinf(masked[0, expected:]).all()

        # Other branch: an INSTRUCT model whose head is too small for the chat
        # ids cannot emit them at all. Say so and keep serving text, don't die
        # at boot. (It must carry meta.chat_format, or the base-checkpoint rule
        # below claims it first.)
        small = ForgeConfig(
            vocab_size=64, dim=32, n_layers=2, n_heads=2,
            max_seq_len=256, dropout=0.0, use_gradient_checkpointing=False,
        )
        small_ckpt = tmp_path / "small_head_instruct.pth"
        torch.save({"model_state_dict": Enigma(small).state_dict(), "config": small.to_dict(),
                    "meta": {"chat_format": CHAT_FORMAT_NAME}}, small_ckpt)
        serve.boot(argv=["--model", str(small_ckpt), "--max-context", "128"])
        assert getattr(serve.model, "live_vocab_size", None) is None
        assert "tool calls and <|im_end|> are unavailable" in capsys.readouterr().out

        # THIRD branch: a checkpoint declaring MORE vocab than the tokenizer
        # table holds. Boot must still DECLARE the decodable boundary and warn
        # about the aliasing. Two earlier versions got this wrong in opposite
        # directions: checking only the upper bound CRASHED here, and then
        # skipping the declaration left the mask at config.vocab_size -- which
        # for a vocab that is a multiple of 64 masks NOTHING and hands sampling
        # every undecodable row.
        wide = ForgeConfig(
            vocab_size=real_vocab + 22, dim=32, n_layers=2, n_heads=2,
            max_seq_len=256, dropout=0.0, use_gradient_checkpointing=False,
        )
        wide_ckpt = tmp_path / "wide_vocab.pth"
        torch.save({"model_state_dict": Enigma(wide).state_dict(), "config": wide.to_dict(),
                    "meta": {"chat_format": CHAT_FORMAT_NAME}}, wide_ckpt)
        serve.boot(argv=["--model", str(wide_ckpt), "--max-context", "128"])
        assert serve._BOOTED is True, "boot died on the vocab-mismatch pairing"
        expected = max(chat_token_ids(serve.tokenizer).values()) + 1
        assert serve.model.live_vocab_size == expected
        assert expected < serve.model.config.vocab_size
        assert "ALIAS trained vocab" in capsys.readouterr().out
        # and the mask really bites: everything past the decodable boundary is out
        step = torch.zeros(1, serve.model.output.weight.shape[0])
        assert torch.isinf(serve.model._live_vocab_logits(step)[0, expected:]).all()

        # FOURTH: a BASE checkpoint never trained those rows. Declaring them
        # decodable would hand random-init rows to argmax -- the exact thing
        # the pad-row guard exists to stop -- and the base decode path renders
        # specials literally. T2/T3 produce this checkpoint class.
        base = ForgeConfig(
            vocab_size=real_vocab, dim=32, n_layers=2, n_heads=2,
            max_seq_len=256, dropout=0.0, use_gradient_checkpointing=False,
        )
        base_ckpt = tmp_path / "base_no_chat.pth"
        torch.save({"model_state_dict": Enigma(base).state_dict(), "config": base.to_dict()},
                   base_ckpt)  # no meta.chat_format -> INSTRUCT False
        serve.boot(argv=["--model", str(base_ckpt), "--max-context", "128"])
        assert serve.INSTRUCT is False
        assert getattr(serve.model, "live_vocab_size", None) is None
        untrained = serve.model._live_vocab_logits(
            torch.zeros(1, serve.model.output.weight.shape[0])
        )
        assert torch.isinf(untrained[0, base.vocab_size:]).all(), \
            "untrained chat rows were left samplable on a base checkpoint"
    finally:
        for name, value in snapshot.items():
            setattr(serve, name, value)


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
    _hermetic_state(monkeypatch, tmp_path)
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
    _hermetic_state(monkeypatch, tmp_path)
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
    _hermetic_state(monkeypatch, tmp_path)
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
    _hermetic_state(monkeypatch, tmp_path)
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


# ---------------------------------------------------------------------------
# Per-persona runtime state (mute + talk-mode) and the one-time migration
# ---------------------------------------------------------------------------


def _legacy_state(tmp_path, muted, enabled):
    """Write the repo-anchored state files boot() migrates from."""
    mute = tmp_path / "legacy" / "mute_state.json"
    talk = tmp_path / "legacy" / "talk_mode.json"
    mute.parent.mkdir(parents=True, exist_ok=True)
    mute.write_text(json.dumps({"muted": muted}), encoding="utf-8")
    talk.write_text(json.dumps({"enabled": enabled}), encoding="utf-8")
    return mute, talk


def test_boot_migrates_legacy_repo_state_into_the_persona_home(monkeypatch, tmp_path, capsys):
    """Mute and talk-mode moved out of the repo checkout into her data home,
    and a move that dropped the old truth would silently unmute a muted
    machine on the next launch -- the one thing the mute state exists to
    prevent. First boot COPIES, says both paths out loud, and leaves the
    legacy file for a rollback to find."""
    import os

    snapshot = {name: getattr(serve, name) for name in _RUNTIME_GLOBALS}
    monkeypatch.setattr(serve.torch.cuda, "is_available", lambda: False)
    home = _hermetic_state(monkeypatch, tmp_path)
    monkeypatch.setattr(serve, "_BOOT_ENV_WRITES", {})
    for key in _HF_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    legacy_mute, legacy_talk = _legacy_state(tmp_path, muted=True, enabled=True)
    ckpt = _tiny_ckpt(tmp_path)
    try:
        serve.boot(argv=["--model", str(ckpt), "--max-context", "128"])
        assert serve._MUTE_STATE == home / ".enigma_engine" / "mute_state.json"
        assert serve._TALK_STATE == home / ".enigma_engine" / "talk_mode.json"
        assert serve.MUTED is True and serve.TALK_MODE is True  # the truth crossed over
        assert json.loads(serve._MUTE_STATE.read_text(encoding="utf-8")) == {"muted": True}
        assert legacy_mute.exists() and legacy_talk.exists(), "the migration must copy, never move"
        out = capsys.readouterr().out
        assert "migrated runtime state" in out
        assert str(legacy_mute) in out and str(serve._MUTE_STATE) in out
    finally:
        for name, value in snapshot.items():
            setattr(serve, name, value)
        for key in _HF_ENV_KEYS:
            os.environ.pop(key, None)


def test_a_corrupt_legacy_state_file_migrates_without_taking_boot_down(monkeypatch, tmp_path, capsys):
    """The legacy files are whatever a crashed or half-written install left in
    the checkout, and the migration reads them as BYTES -- it cannot tell a
    truncated mute_state.json from a good one. So the corrupt case must land
    the same way the missing one does: boot completes, the copy is made and
    said out loud, and the unreadable state falls back to the documented
    defaults (unmuted, silent) instead of a traceback in the middle of
    startup."""
    import os

    snapshot = {name: getattr(serve, name) for name in _RUNTIME_GLOBALS}
    monkeypatch.setattr(serve.torch.cuda, "is_available", lambda: False)
    home = _hermetic_state(monkeypatch, tmp_path)
    monkeypatch.setattr(serve, "_BOOT_ENV_WRITES", {})
    for key in _HF_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    legacy_mute = tmp_path / "legacy" / "mute_state.json"
    legacy_talk = tmp_path / "legacy" / "talk_mode.json"
    legacy_mute.parent.mkdir(parents=True, exist_ok=True)
    # Garbage on both sides of the decode: bytes that are not UTF-8 at all,
    # and text that decodes fine and is not JSON.
    legacy_mute.write_bytes(b'\xff\xfe{"muted": tr')
    legacy_talk.write_bytes(b"not json at all")
    ckpt = _tiny_ckpt(tmp_path)
    try:
        serve.boot(argv=["--model", str(ckpt), "--max-context", "128"])
        assert serve._BOOTED is True  # startup survived the unreadable state
        assert serve.MUTED is False and serve.TALK_MODE is False  # the defaults
        # The copy is best-effort and byte-faithful: it does not parse, so the
        # corruption crosses over verbatim and the legacy file stays put.
        assert serve._MUTE_STATE.read_bytes() == legacy_mute.read_bytes()
        assert serve._TALK_STATE.read_bytes() == legacy_talk.read_bytes()
        assert legacy_mute.exists() and legacy_talk.exists()
        out = capsys.readouterr().out
        assert "migrated runtime state" in out
        assert "Traceback" not in out
        assert serve._MUTE_STATE == home / ".enigma_engine" / "mute_state.json"
    finally:
        for name, value in snapshot.items():
            setattr(serve, name, value)
        for key in _HF_ENV_KEYS:
            os.environ.pop(key, None)


def test_persona_home_state_wins_over_the_legacy_copy(monkeypatch, tmp_path, capsys):
    """The migration is ONE-TIME. A later boot must read her home, not re-seed
    it from a stale repo file -- otherwise every launch would resurrect the
    mute truth from before the move."""
    import os

    snapshot = {name: getattr(serve, name) for name in _RUNTIME_GLOBALS}
    monkeypatch.setattr(serve.torch.cuda, "is_available", lambda: False)
    home = _hermetic_state(monkeypatch, tmp_path)
    monkeypatch.setattr(serve, "_BOOT_ENV_WRITES", {})
    for key in _HF_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    _legacy_state(tmp_path, muted=True, enabled=True)
    (home / ".enigma_engine").mkdir(parents=True)
    (home / ".enigma_engine" / "mute_state.json").write_text(
        json.dumps({"muted": False}), encoding="utf-8")
    (home / ".enigma_engine" / "talk_mode.json").write_text(
        json.dumps({"enabled": False}), encoding="utf-8")
    ckpt = _tiny_ckpt(tmp_path)
    try:
        serve.boot(argv=["--model", str(ckpt), "--max-context", "128"])
        assert serve.MUTED is False and serve.TALK_MODE is False
        assert "migrated runtime state" not in capsys.readouterr().out
    finally:
        for name, value in snapshot.items():
            setattr(serve, name, value)
        for key in _HF_ENV_KEYS:
            os.environ.pop(key, None)


def test_a_pack_does_not_adopt_the_legacy_repo_state(monkeypatch, tmp_path, capsys):
    """The legacy repo files are ENIGMA's mute and talk-mode truth. Migrating
    them into whichever persona booted first would hand a brand-new AI the
    state of the one this checkout has been serving -- and talk-mode ON, when
    the documented default is that she starts SILENT. The gate is the persona,
    not the order of boots: her home stays empty and nothing is said."""
    import os

    snapshot = {name: getattr(serve, name) for name in _RUNTIME_GLOBALS}
    monkeypatch.setattr(serve.torch.cuda, "is_available", lambda: False)
    home = _hermetic_state(monkeypatch, tmp_path)
    monkeypatch.setattr(serve, "_BOOT_ENV_WRITES", {})
    for key in _HF_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(serve.app, "title", serve.app.title)
    monkeypatch.setattr(serve.app, "openapi_schema", serve.app.openapi_schema)
    legacy_mute, legacy_talk = _legacy_state(tmp_path, muted=True, enabled=True)
    pack = tmp_path / "atlas.json"
    pack.write_text(json.dumps({"name": "Atlas"}), encoding="utf-8")
    ckpt = _tiny_ckpt(tmp_path)
    try:
        serve.boot(argv=["--model", str(ckpt), "--max-context", "128", "--persona", str(pack)])
        assert serve._MUTE_STATE == home / ".atlas" / "mute_state.json"
        assert serve._TALK_STATE == home / ".atlas" / "talk_mode.json"
        assert not serve._MUTE_STATE.exists() and not serve._TALK_STATE.exists()
        assert serve.MUTED is False and serve.TALK_MODE is False
        assert "migrated runtime state" not in capsys.readouterr().out
        assert legacy_mute.exists() and legacy_talk.exists()
    finally:
        for name, value in snapshot.items():
            setattr(serve, name, value)
        for key in _HF_ENV_KEYS:
            os.environ.pop(key, None)


def test_a_second_persona_does_not_inherit_her_state(monkeypatch, tmp_path):
    """One shared state file was a real one-AI-per-machine guard: two AIs
    served from this checkout would have muted and un-muted each other. State
    follows PERSONA.home like the voice recipe and the images do."""
    import os

    snapshot = {name: getattr(serve, name) for name in _RUNTIME_GLOBALS}
    monkeypatch.setattr(serve.torch.cuda, "is_available", lambda: False)
    home = _hermetic_state(monkeypatch, tmp_path)
    monkeypatch.setattr(serve, "_BOOT_ENV_WRITES", {})
    for key in _HF_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    # boot() writes app.title (and drops the cached schema); monkeypatch
    # records both so serving Atlas here cannot leak into the rest of the
    # session -- `app` is a module object, not one of the boot globals the
    # snapshot restores.
    monkeypatch.setattr(serve.app, "title", serve.app.title)
    monkeypatch.setattr(serve.app, "openapi_schema", serve.app.openapi_schema)
    hers = home / ".enigma_engine"
    hers.mkdir(parents=True)
    (hers / "mute_state.json").write_text(json.dumps({"muted": True}), encoding="utf-8")
    (hers / "talk_mode.json").write_text(json.dumps({"enabled": True}), encoding="utf-8")
    pack = tmp_path / "atlas.json"
    pack.write_text(json.dumps({"name": "Atlas"}), encoding="utf-8")
    ckpt = _tiny_ckpt(tmp_path)
    try:
        serve.boot(argv=["--model", str(ckpt), "--max-context", "128", "--persona", str(pack)])
        assert serve._MUTE_STATE == home / ".atlas" / "mute_state.json"
        assert serve._TALK_STATE == home / ".atlas" / "talk_mode.json"
        assert serve.MUTED is False and serve.TALK_MODE is False  # her state, not Enigma's
        assert json.loads((hers / "mute_state.json").read_text(encoding="utf-8")) == {"muted": True}
        # ...and the API says WHO it serves, down to the OpenAPI metadata
        assert serve.app.title == "Atlas (from-scratch)"
        assert serve.app.openapi()["info"]["title"] == "Atlas (from-scratch)"
    finally:
        for name, value in snapshot.items():
            setattr(serve, name, value)
        for key in _HF_ENV_KEYS:
            os.environ.pop(key, None)


def test_boot_serves_a_pack_DIRECTORY(monkeypatch, tmp_path, write_persona_pack):
    """`serve --persona <dir>` was the point of the directory format, and
    nothing here had ever booted one -- the boot tests all pass a bare
    pack.json, which takes a different branch of Persona.load. The manifest
    inside the directory has to reach every name boot derives, and the home has
    to land under the patched profile: a boot reading the real one writes
    Atlas's runtime state into the developer's own machine."""
    import os

    snapshot = {name: getattr(serve, name) for name in _RUNTIME_GLOBALS}
    monkeypatch.setattr(serve.torch.cuda, "is_available", lambda: False)
    home = _hermetic_state(monkeypatch, tmp_path)
    monkeypatch.setattr(serve, "_BOOT_ENV_WRITES", {})
    for key in _HF_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(serve.app, "title", serve.app.title)
    monkeypatch.setattr(serve.app, "openapi_schema", serve.app.openapi_schema)
    pack = write_persona_pack()
    assert pack.is_dir() and (pack / PACK_MANIFEST).is_file(), "a DIRECTORY pack"
    ckpt = _tiny_ckpt(tmp_path)
    try:
        serve.boot(argv=["--model", str(ckpt), "--max-context", "128", "--persona", str(pack)])
        assert serve.PERSONA.name == "Atlas"
        assert serve.PERSONA.home == home / ".atlas"
        assert tmp_path in serve.PERSONA.home.parents  # never the real profile
        assert serve._VOICE_STATE == home / ".atlas" / "voice.json"
        assert serve.IMAGES_DIR == home / ".atlas" / "images"
        assert serve._MUTE_STATE == home / ".atlas" / "mute_state.json"
        # ...and it is Atlas the user sees, in the page and in the API metadata
        assert "<title>Atlas</title>" in serve.chat_page()
        assert "<h1>Atlas</h1>" in serve.chat_page()
        assert serve.app.title == "Atlas (from-scratch)"
    finally:
        for name, value in snapshot.items():
            setattr(serve, name, value)
        for key in _HF_ENV_KEYS:
            os.environ.pop(key, None)


def test_boot_carries_a_spaced_persona_name_through_the_turn_machinery(
        monkeypatch, tmp_path, write_persona_pack):
    """A space is legal in a name (`_SAFE_NAME` allows it) and it is the shape
    that splits a label: the transcript marker, the stop text that cuts on it
    and the tools preamble must all carry BOTH words. A name split at the space
    prompts her as one AI and stops on another -- the failure that leaves a
    whole fabricated assistant turn in the reply."""
    import os

    snapshot = {name: getattr(serve, name) for name in _RUNTIME_GLOBALS}
    monkeypatch.setattr(serve.torch.cuda, "is_available", lambda: False)
    _hermetic_state(monkeypatch, tmp_path)
    monkeypatch.setattr(serve, "_BOOT_ENV_WRITES", {})
    for key in _HF_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(serve.app, "title", serve.app.title)
    monkeypatch.setattr(serve.app, "openapi_schema", serve.app.openapi_schema)
    pack = write_persona_pack({PACK_MANIFEST: json.dumps({
        "name": "Atlas Prime",
        "data_dirname": ".atlas_prime",
        "name_meaning": "the one who carries the weight",
    })})
    ckpt = _tiny_ckpt(tmp_path)
    try:
        serve.boot(argv=["--model", str(ckpt), "--max-context", "128", "--persona", str(pack)])
        assert serve.PERSONA.transcript_label == "\nAtlas Prime:"
        assert serve._STOP_TEXTS == ("\nUser:", "\nAtlas Prime:")
        # the transcript ends on the marker the stop text cuts on -- one name
        transcript = serve._render_transcript([serve.Msg(role="user", content="hello")])
        assert transcript == "User: hello\nAtlas Prime:"
        fabricated = "Sure.\nAtlas Prime: and a turn nobody asked for"
        assert serve._find_stop(fabricated, serve._STOP_TEXTS) == len("Sure.")

        # ...and the preamble serve prepends to a tools block, through serve's
        # own assembly rather than the dataclass property.
        user = "compute 372 + 519"
        assert serve._builtin_tools(user, False) == [serve._CALC_TOOL]  # not vacuous
        req = serve.ChatReq(messages=[serve.Msg(role="user", content=user)])
        out = serve._with_context([m.model_dump(exclude_none=True) for m in req.messages], req)
        assert out[0]["content"].startswith("You are Atlas Prime. You can use tools")
    finally:
        for name, value in snapshot.items():
            setattr(serve, name, value)
        for key in _HF_ENV_KEYS:
            os.environ.pop(key, None)


def test_the_default_boot_says_it_is_HER_and_publishes_her_own_model_id(
        monkeypatch, tmp_path):
    """A port can only be asked WHO is serving if the server says so, and the
    two endpoints an outside reader already polls are the ones that must
    answer: /v1/capabilities for the launcher deciding whether a listener is
    Enigma, /v1/models for the client naming the model that replied.

    Her payload is the pin. The model id is derived now, and every field of
    the default entry is what the endpoint published as literals -- a
    derivation that renders her as anything but "enigma"/"enigma" has moved
    her surface, whatever it does for a pack."""
    import os

    snapshot = {name: getattr(serve, name) for name in _RUNTIME_GLOBALS}
    monkeypatch.setattr(serve.torch.cuda, "is_available", lambda: False)
    _hermetic_state(monkeypatch, tmp_path)
    monkeypatch.setattr(serve, "_BOOT_ENV_WRITES", {})
    for key in _HF_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(serve.app, "title", serve.app.title)
    monkeypatch.setattr(serve.app, "openapi_schema", serve.app.openapi_schema)
    ckpt = _tiny_ckpt(tmp_path)
    try:
        serve.boot(argv=["--model", str(ckpt), "--max-context", "128"])
        caps = serve.capabilities()
        assert caps["persona"] == "Enigma"
        assert caps["persona_is_default"] is True

        entry = serve.list_models()["data"][0]
        assert {k: v for k, v in entry.items() if k != "checkpoint"} == {
            "id": "enigma", "object": "model", "owned_by": "enigma"}
    finally:
        for name, value in snapshot.items():
            setattr(serve, name, value)
        for key in _HF_ENV_KEYS:
            os.environ.pop(key, None)


def test_a_pack_boot_answers_with_the_PACKS_identity_on_both_endpoints(
        monkeypatch, tmp_path, write_persona_pack):
    """The same serve_enigma.py, the same port, a different AI -- which is
    exactly why a process name cannot decide ownership. Both self-reports
    have to follow the pack, and is_default is what tells a reader that the
    server on this port is not the AI this checkout is named for."""
    import os

    snapshot = {name: getattr(serve, name) for name in _RUNTIME_GLOBALS}
    monkeypatch.setattr(serve.torch.cuda, "is_available", lambda: False)
    _hermetic_state(monkeypatch, tmp_path)
    monkeypatch.setattr(serve, "_BOOT_ENV_WRITES", {})
    for key in _HF_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(serve.app, "title", serve.app.title)
    monkeypatch.setattr(serve.app, "openapi_schema", serve.app.openapi_schema)
    pack = write_persona_pack()
    ckpt = _tiny_ckpt(tmp_path)
    try:
        serve.boot(argv=["--model", str(ckpt), "--max-context", "128", "--persona", str(pack)])
        caps = serve.capabilities()
        assert caps["persona"] == "Atlas"
        assert caps["persona_is_default"] is False

        entry = serve.list_models()["data"][0]
        assert entry["id"] == "atlas" and entry["owned_by"] == "atlas"
    finally:
        for name, value in snapshot.items():
            setattr(serve, name, value)
        for key in _HF_ENV_KEYS:
            os.environ.pop(key, None)
