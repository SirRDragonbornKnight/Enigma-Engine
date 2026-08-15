"""Persona pack: identity as DATA, so the trainer can mold a different AI
instead of a second Enigma.

The bar for this indirection is that it changes NOTHING about her. Every value
the default persona produces must be byte-identical to the literal it replaced,
or the refactor moved her identity while claiming to preserve it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import serve_enigma as serve
from enigma_engine.core.persona import Persona


def test_the_default_persona_is_enigma_byte_for_byte():
    p = Persona()
    assert p.name == "Enigma"
    assert p.home == Path.home() / ".enigma_engine"
    assert p.transcript_label == "\nEnigma:"
    assert p.tools_preamble == (
        "You are Enigma. You can use tools when they are needed; "
        "answer directly when they are not."
    )


def test_serve_still_uses_exactly_the_strings_it_had():
    """The sites the pack now feeds, checked against their previous literals."""
    assert serve._VOICE_STATE == Path.home() / ".enigma_engine" / "voice.json"
    assert serve.IMAGES_DIR == Path.home() / ".enigma_engine" / "images"
    assert serve._STOP_TEXTS == ("\nUser:", "\nEnigma:")


def test_the_chat_page_is_titled_with_whoever_is_served(monkeypatch):
    """The page is a module constant built at import, so the name is
    substituted at render time -- the placeholder must never reach a
    browser, and the name is escaped because a pack is data."""
    page = serve.chat_page()
    assert "<title>Enigma</title>" in page
    assert "<h1>Enigma</h1>" in page
    assert "__PERSONA_NAME__" not in page

    monkeypatch.setattr(serve, "PERSONA", Persona(name="Atlas"))
    page = serve.chat_page()
    assert "<title>Atlas</title>" in page
    assert "<h1>Atlas</h1>" in page


def test_another_persona_gets_its_own_home_and_voice(tmp_path):
    """The shared data home was one of the real one-AI-per-machine guards: two
    AIs on this box would have overwritten each other's voice recipe and
    images."""
    pack = tmp_path / "atlas.json"
    pack.write_text(json.dumps({"name": "Atlas"}), encoding="utf-8")
    p = Persona.load(pack)
    assert p.name == "Atlas"
    assert p.home == Path.home() / ".atlas"
    assert p.home != Persona().home
    assert p.transcript_label == "\nAtlas:"
    assert p.tools_preamble.startswith("You are Atlas.")


def test_a_pack_carries_name_semantic_content_rather_than_deriving_it(tmp_path):
    """Her identity answers explain what the WORD means ("a closed box, in the
    good sense"). No template derives that from a different name, so a pack
    supplies its own or says nothing -- it is never generated."""
    assert "closed box" in Persona().name_meaning
    pack = tmp_path / "atlas.json"
    pack.write_text(json.dumps({"name": "Atlas"}), encoding="utf-8")
    assert Persona.load(pack).name_meaning == ""


@pytest.mark.parametrize("blob", [
    {"name": "../evil"},          # the name reaches a directory name
    {"name": "Bad\nName"},        # ...and a stop sequence
    {"name": ""},
    {"name": "Ok", "data_dirname": "a/b"},
    {"name": "Enigm" + chr(0xE1)},   # ...and console prints: cp1252 CRASHES
    {"name": chr(0x415) + "nigma"},  # ...even on a Cyrillic homoglyph of "E"
    {"name": "Bad\tName"},        # printable only -- a tab is not a name
])
def test_an_unsafe_pack_is_refused(tmp_path, blob):
    """The name is not just displayed: it becomes a directory, a stop
    sequence, and console output (the ASCII rule -- a cp1252 console crashes
    on a unicode print, which is what the repo's ASCII gate exists for). The
    safe set is printable ASCII, and a pack outside it refuses at load rather
    than failing somewhere downstream."""
    pack = tmp_path / "bad.json"
    pack.write_text(json.dumps(blob), encoding="utf-8")
    with pytest.raises(SystemExit):
        Persona.load(pack)


def test_a_malformed_or_missing_pack_refuses_honestly(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{nope", encoding="utf-8")
    with pytest.raises(SystemExit):
        Persona.load(bad)
    with pytest.raises(SystemExit):
        Persona.load(tmp_path / "absent.json")
    # ...but NO pack at all is Enigma, not an error: this repo is her.
    assert Persona.load().name == "Enigma"
