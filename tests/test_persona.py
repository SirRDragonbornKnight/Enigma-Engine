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
from enigma_engine.core.persona import ENIGMA, RESERVED_PORTS, Persona


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

    monkeypatch.setattr(serve, "PERSONA", Persona(name="Atlas", data_dirname=".atlas"))
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
    {"name": "Ok", "data_dirname": "."},   # carries no separator, and still
    {"name": "Ok", "data_dirname": ".."},  # ...escapes: profile root / parent
    {"name": "Ok", "data_dirname": "nul"}, # ...and so does a Win32 device name
])
def test_an_unsafe_pack_is_refused(tmp_path, blob):
    """The name is not just displayed: it becomes a directory, a stop
    sequence, and console output (the ASCII rule -- a cp1252 console crashes
    on a unicode print, which is what the repo's ASCII gate exists for). The
    safe set is printable ASCII, and a pack outside it refuses at load rather
    than failing somewhere downstream.

    data_dirname has a second escape the character screen misses: a dot entry
    is a legal bare filename that resolves OUT of the home it is joined to,
    and serve's boot writes runtime state into whatever home resolves to."""
    pack = tmp_path / "bad.json"
    pack.write_text(json.dumps(blob), encoding="utf-8")
    with pytest.raises(SystemExit):
        Persona.load(pack)


def test_a_dot_dirname_is_refused_by_the_dataclass_itself():
    """Not only through load(): the escape is refused where the value is
    bound, so a caller constructing a Persona directly cannot land `home` on
    the profile's parent either."""
    with pytest.raises(ValueError):
        Persona(data_dirname="..")


@pytest.mark.parametrize("dirname", [
    "...",       # Win32 STRIPS trailing dots and spaces at the filesystem, so
    " .",        # ...each of these is a bare component to Path() and a
    ".. ",       # ...different directory to every write serve makes: the
    "atlas.",    # ...profile root, its parent, or an aliased sibling
    "nul",       # the reserved device names, matched case-insensitively...
    "NUL",
    "nul.txt",   # ...and matched on the portion before the first dot, so an
    "com3",      # ...extension does not escape them either
])
def test_a_win32_hole_in_the_data_dirname_is_refused(dirname):
    """Path(x).name == x is not enough on this OS. Win32 rewrites a component
    that ends in a dot or a space, and reserves the device names in every
    directory -- a home named `nul` accepts every write and reads back empty,
    and `atlas.` puts serve's runtime state in a directory nobody named."""
    with pytest.raises(ValueError):
        Persona(name="Atlas", data_dirname=dirname)


@pytest.mark.parametrize("name, dirname", [
    ("Atlas", ".Enigma_Engine"),   # not-default by the exact compare...
    ("Atlas", ".enigma_engine"),   # ...and the exact spelling under a new name
    ("Enigma", ".ENIGMA_ENGINE"),  # ...and her own name over a cased home
])
def test_another_identity_cannot_land_on_her_data_home(name, dirname):
    """Directory names are case-insensitive on Win32, so a value that differs
    from hers only in case IS her home on disk -- a pack carrying it serves
    someone else out of Enigma's voice recipe, images and runtime state."""
    with pytest.raises(ValueError):
        Persona(name=name, data_dirname=dirname)


def test_her_own_identity_and_an_ordinary_pack_still_construct():
    """The collision rule gates on IDENTITY, not on the string: Enigma's own
    home is hers, whether she arrives as the default or spelled out by a pack
    that also carries extra keys of its own."""
    assert Persona().home == Path.home() / ".enigma_engine"
    spelled_out = Persona(
        name=ENIGMA["name"],
        data_dirname=ENIGMA["data_dirname"],
        name_meaning=ENIGMA["name_meaning"],
        extra={"creator": "SirRulean"},
    )
    assert spelled_out.is_default is True
    assert Persona(name="Atlas", data_dirname=".atlas").home == Path.home() / ".atlas"


@pytest.mark.parametrize("meaning", [
    "a closed box\n\nin the good sense",  # a blank line is a PARAGRAPH break,
    "a closed box\rin the good sense",    # ...deduped apart from its document
    "a closed box,\tin the good sense",
    "a closed box" + chr(0x1E),           # ...and U+001E is the RECORD
    "a closed box" + chr(0),              # ...separator: it splits the document
    "a closed box" + chr(127),
])
def test_a_control_character_in_name_meaning_is_refused(meaning):
    """name_meaning is rendered raw into ONE line of the pretrain identity
    document, and that line is a whole record in a shard: U+001E cuts the
    record in two, a blank line splits it into independently deduped
    paragraphs, and every other control character bakes into the corpus."""
    with pytest.raises(ValueError):
        Persona(name="Atlas", data_dirname=".atlas", name_meaning=meaning)


def test_name_meaning_may_be_empty_or_carry_unicode():
    """A pack may say nothing about its name, and the repo's ASCII rule is
    CONSOLE-bound -- training text carries unicode, her own anchors included."""
    assert Persona(name="Atlas", data_dirname=".atlas", name_meaning="").name_meaning == ""
    meaning = "el que carga el peso, con acento: " + chr(0xE9)
    assert Persona(
        name="Atlas", data_dirname=".atlas", name_meaning=meaning
    ).name_meaning == meaning


@pytest.mark.parametrize("blob", [
    {"name": 123},
    {"name": "Ok", "data_dirname": 7},
    {"name": "Ok", "name_meaning": ["x"]},
    {"name": None},                       # JSON null is PRESENT, not absent
    {"name": "Ok", "data_dirname": 0},    # FALSY non-strings, which the `or`
    {"name": "Ok", "data_dirname": None}, # ...quietly replaced with a slug
])
def test_a_mistyped_pack_field_refuses_instead_of_tracebacking(tmp_path, blob):
    """A pack is untrusted JSON and the field screens only run on strings: a
    numeric name died inside the name regex with a raw TypeError, a non-str
    name reached _slug as an AttributeError, and a FALSY non-string
    data_dirname (0, null) slipped past the `or` into the derived default
    instead of refusing at all. Every one of them is a refusal now."""
    pack = tmp_path / "typed.json"
    pack.write_text(json.dumps(blob), encoding="utf-8")
    with pytest.raises(SystemExit):
        Persona.load(pack)


def test_a_pack_spelling_out_her_values_is_still_her(tmp_path):
    """is_default gates what is ENIGMA's alone -- the legacy repo-anchored
    mute and talk-mode state. Identity is the three core VALUES, so a pack
    that spells them out is her; an `extra` key of its own says nothing about
    who she is, and whole-dataclass equality read one as a different AI and
    skipped her migration."""
    same = tmp_path / "hers.json"
    same.write_text(json.dumps({
        "name": "Enigma",
        "data_dirname": ".enigma_engine",
        "name_meaning": Persona().name_meaning,
        "creator": "SirRulean",
    }), encoding="utf-8")
    loaded = Persona.load(same)
    assert loaded.extra == {"creator": "SirRulean"}
    assert loaded.is_default is True

    other = tmp_path / "atlas.json"
    other.write_text(json.dumps({"name": "Atlas"}), encoding="utf-8")
    assert Persona.load(other).is_default is False


def test_a_pack_directory_loads_its_mechanical_fields_from_the_manifest(tmp_path):
    """A pack is a DIRECTORY -- the content files beside a manifest -- and the
    manifest is the same JSON a bare pack file has always been. Reading it here
    is what makes `serve --persona <dir>` work with no change to serve at all,
    and it is why the two spellings load to the same persona."""
    blob = {"name": "Atlas", "name_meaning": "the one who carries the weight"}
    pack = tmp_path / "atlas"
    pack.mkdir()
    (pack / "pack.json").write_text(json.dumps(blob), encoding="utf-8")

    loaded = Persona.load(pack)
    assert loaded.name == "Atlas"
    assert loaded.home == Path.home() / ".atlas"
    assert loaded.name_meaning == "the one who carries the weight"

    # ...and a bare pack FILE still loads exactly as it did before directories
    # existed: the mechanical half is all serve ever needed.
    bare = tmp_path / "atlas.json"
    bare.write_text(json.dumps(blob), encoding="utf-8")
    assert Persona.load(bare) == loaded


def test_a_pack_directory_without_a_manifest_refuses(tmp_path):
    """A directory with content files but no manifest is a half-built pack, and
    the silent alternative is worse than the refusal: with no name of its own it
    would load ENIGMA's defaults and serve her data home under someone else's
    folder."""
    pack = tmp_path / "atlas"
    pack.mkdir()
    (pack / "anchors.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="pack.json"):
        Persona.load(pack)


def test_a_pack_declares_the_port_its_launchers_bind(tmp_path):
    """A second AI needs a port of her own, and the pack is where she says
    which -- explicit beats an auto-assignment nobody can predict from the
    outside. Enigma declares none: her chain has always bound 8000, and the
    launcher default is still what says so."""
    pack = tmp_path / "atlas.json"
    pack.write_text(json.dumps({"name": "Atlas", "port": 8300}), encoding="utf-8")
    assert Persona.load(pack).port == 8300
    assert Persona().port is None
    assert Persona.load().port is None
    # ...and the port is not part of WHO she is: Enigma on another port is a
    # flag, not a second identity, so is_default must not gate on it.
    assert Persona(port=8300).is_default is True


@pytest.mark.parametrize("port, why", [
    (8000, "collides with the daily serve port"),   # her own daily serve
    (8123, "collides with the eval scratch port"),  # ...and the eval scratch
    (0, "outside 1024-65535"),                      # ...and nothing can listen
    (70000, "outside 1024-65535"),                  # ...past the last port
    ("8001", "must be an integer, not str"),        # a pack is untrusted JSON
    (True, "must be an integer, not bool"),         # ...and bool IS an int
])
def test_an_unusable_pack_port_refuses(tmp_path, port, why):
    """The port reaches a bind and a launcher's ownership check, so an
    unusable one has to refuse where it is read rather than fail as a bind
    error hours later in a hidden server's log. `true` is the sharp case:
    bool is a subclass of int, so an isinstance check alone binds port 1."""
    pack = tmp_path / "atlas.json"
    pack.write_text(json.dumps({"name": "Atlas", "port": port}), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        Persona.load(pack)
    assert why in str(exc.value)
    assert str(pack) in str(exc.value)  # WHICH pack, not just what is wrong
    # ...and through the dataclass too, so a caller building one by hand is
    # refused by the same lines.
    with pytest.raises(ValueError):
        Persona(name="Atlas", data_dirname=".atlas", port=port)


def test_the_reserved_ports_are_the_two_this_machine_already_spends():
    """Named rather than open-coded: the eval scratch port lives in
    eval_behavior's SCRATCH_PORTS and the daily port in the launcher chain, so
    a reader of either can find the refusal that protects it."""
    assert sorted(RESERVED_PORTS) == [8000, 8123]


def test_a_malformed_or_missing_pack_refuses_honestly(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{nope", encoding="utf-8")
    with pytest.raises(SystemExit):
        Persona.load(bad)
    with pytest.raises(SystemExit):
        Persona.load(tmp_path / "absent.json")
    # ...but NO pack at all is Enigma, not an error: this repo is her.
    assert Persona.load().name == "Enigma"
