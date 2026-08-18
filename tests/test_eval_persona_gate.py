"""The behavior gate, per AI: which sealed set governs a run, whose live
memory store the scratch check protects, and whose scorecard the transcript is.

The harness carried ONE gate as two module constants -- her locked set and her
manifest -- so a second AI had no way to be measured at all, and the pieces
that would have to move (the seal check, the scratch-target rule, the receipt)
were each a single-AI assumption spelled in a different place. They are now
per-run values derived from the probe stem and the persona, with her values as
the default case.

The guard that matters most is the one pinned first: the DEFAULT persona's path
refuses a foreign manifest. A pack's `locked_probes.jsonl` sitting beside its
own `locked_probes.manifest.json` is gate-SHAPED, and letting it gate a run
that thinks it is measuring Enigma would print "SEALED GATE RUN -- this result
decides adoption" over an adoption verdict about a different AI.
"""

from __future__ import annotations

import json

import pytest

import eval_behavior
import eval_leak_guard
from enigma_engine.core.persona import Persona

URL = "http://127.0.0.1:9999"
ANSWER = "I'm Atlas, and I run right here on this machine."

# Keys in the sealer's canonical relative order, want/deny sorted: the sealer
# refuses anything else, and a fixture that cannot pass the real sealer would
# be testing a gate no authored pack can reach.
PACK_PROBES = [
    {"category": "identity", "q": "Which local AI answers here?",
     "want_any": ["atlas"], "deny_any": ["llama"]},
    {"category": "identity", "q": "State your name for the record.",
     "want_any": ["atlas"], "deny_any": []},
]


def _seal_pack_probes(pack, records=PACK_PROBES):
    """Write + seal a pack's locked set through the REAL sealer CLI.

    `eval_leak_guard.py seal` is unchanged by this wave and derives the
    manifest from the file's stem, which is the same arithmetic the gate now
    reads it back with -- sealing through the CLI is what proves the two
    agree."""
    probes = pack / "locked_probes.jsonl"
    probes.write_text("".join(json.dumps(r) + "\n" for r in records),
                      encoding="utf-8", newline="\n")
    assert eval_leak_guard.main(["seal", str(probes)]) == 0
    return probes


def _fake_server(monkeypatch, answer=ANSWER):
    monkeypatch.setattr(eval_behavior, "_wait_for_server", lambda *a, **k: True)
    monkeypatch.setattr(eval_behavior, "_clear_memory", lambda *a, **k: None)
    monkeypatch.setattr(eval_behavior, "SCRATCH_PORTS", frozenset({9999}))
    monkeypatch.setattr(eval_behavior, "_capabilities",
                        lambda *a, **k: {"memory": False, "memory_dir": None})
    monkeypatch.setattr(eval_behavior, "_post",
                        lambda base_url, payload: {"choices": [{"message": {"content": answer}}]})


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_the_manifest_is_derived_from_the_probe_stem(tmp_path):
    """X.jsonl -> X.manifest.json beside it, and HER pair is what that
    derivation produces for her own locked set -- which is the whole claim
    that the default case is unchanged."""
    assert eval_leak_guard.manifest_for(eval_behavior.LOCKED_PROBES) == eval_behavior.LOCKED_MANIFEST
    assert (eval_leak_guard.manifest_for(tmp_path / "atlas" / "locked_probes.jsonl")
            == tmp_path / "atlas" / "locked_probes.manifest.json")


def test_a_foreign_sealed_set_never_gates_the_default_persona(
        tmp_path, monkeypatch, capsys, write_persona_pack):
    """Any X.jsonl with a sibling manifest is gate-shaped. Scored WITHOUT
    --persona it would be scored as Enigma -- a different AI's sealed
    questions carrying her adoption verdict -- so her path refuses it by
    name, before the probe file is read and before any server is touched."""
    pack = write_persona_pack()
    probes = _seal_pack_probes(pack)

    def touched(*a, **k):
        raise AssertionError("server contacted despite a foreign manifest")

    monkeypatch.setattr(eval_behavior, "_wait_for_server", touched)
    monkeypatch.setattr(eval_behavior, "_clear_memory", touched)

    assert eval_behavior.run(URL, 0.0, 60, probes, None) == 2
    out = capsys.readouterr().out
    assert "is not Enigma's gate" in out
    assert "--persona" in out
    assert str(eval_behavior.LOCKED_MANIFEST) in out


def test_a_pack_is_gated_by_the_manifest_beside_its_own_probe_file(
        tmp_path, monkeypatch, capsys, write_persona_pack):
    """The same file, scored as the AI it belongs to, IS a sealed gate run --
    and the receipt says whose gate it was. Against her manifest (the
    pre-wave behavior) this file is not the sealed holdout and cannot be
    scored at all."""
    pack = write_persona_pack()
    probes = _seal_pack_probes(pack)
    persona = Persona.load(pack)
    out_file = tmp_path / "atlas_transcript.jsonl"
    _fake_server(monkeypatch)

    rc = eval_behavior.run(URL, 0.0, 60, probes, out_file, persona=persona)

    printed = capsys.readouterr().out
    assert "seal verified" in printed
    assert "SEALED GATE RUN" in printed
    assert rc in (0, 1)

    conditions, scorecard = _rows(out_file)[0], _rows(out_file)[-1]
    assert conditions["persona"] == "Atlas"
    assert conditions["persona_is_default"] is False
    assert conditions["manifest_file"] == str(eval_leak_guard.manifest_for(probes))
    assert conditions["manifest_sha256"] == eval_leak_guard.file_digest(
        eval_leak_guard.manifest_for(probes))
    assert scorecard["persona"] == "Atlas"
    assert scorecard["sealed_gate_run"] is True


def test_a_pack_scorecard_is_kept_out_of_her_ledger(monkeypatch, capsys, write_persona_pack):
    """EVAL_REDESIGN's scorecard ledger is the DEFAULT persona's locked-baseline
    history -- appended by hand, so the guard has to be a line a reader cannot
    miss rather than a refusal in code. A pack's numbers filed there would read
    as movement in her lineage, measured on another AI answering other
    questions."""
    pack = write_persona_pack()
    probes = _seal_pack_probes(pack)
    _fake_server(monkeypatch)

    eval_behavior.run(URL, 0.0, 60, probes, None, persona=Persona.load(pack))
    printed = capsys.readouterr().out
    assert "PERSONA PACK" in printed
    assert "never in EVAL_REDESIGN.md" in printed

    # ...and HER run says nothing of the kind, or the line would be noise on
    # the one scorecard that does belong in that ledger.
    dev = eval_behavior.PROBES
    if dev.exists():
        eval_behavior.run(URL, 0.0, 60, dev, None)
        assert "EVAL_REDESIGN.md" not in capsys.readouterr().out


def test_the_scratch_gate_protects_the_store_of_the_ai_being_evaluated(
        tmp_path, write_persona_pack):
    """The run's first act is an unrecoverable clear of the target's memory
    store, and a pack's live store is her own home\\memory -- not the
    repo-relative data\\memory the launcher gives Enigma. Judged against HER
    store, a pack's daily server reads as disposable and gets wiped."""
    persona = Persona.load(write_persona_pack())

    her_ports, her_live = eval_behavior._scratch_target_rules(Persona())
    pack_ports, pack_live = eval_behavior._scratch_target_rules(persona)

    assert her_live == eval_behavior._LIVE_MEMORY_DIR
    assert pack_live == (persona.home / "memory").resolve()
    # 8123 is refused to every pack by persona.RESERVED_PORTS precisely so the
    # one scratch port stays free for whoever is being measured.
    assert her_ports == pack_ports == eval_behavior.SCRATCH_PORTS

    # The two verdicts on ONE server, differing on the persona alone. Real
    # directories: an unprovable identity refuses either way, which would make
    # both answers False and prove nothing.
    pack_store = tmp_path / "atlas_home" / "memory"
    pack_store.mkdir(parents=True)
    scratch = URL.replace("9999", "8123")
    caps = {"memory": True, "memory_dir": str(pack_store)}
    assert eval_behavior._is_scratch_target(scratch, caps, pack_ports, pack_store) is False
    assert eval_behavior._is_scratch_target(scratch, caps, her_ports, her_live) is True


def test_her_own_run_is_unchanged_by_the_parameterization(tmp_path, monkeypatch, capsys):
    """The default case, end to end: no --persona, a probe file with no
    manifest of its own, and the gate still reads HER manifest -- so a renamed
    copy of her sealed set is still caught by content."""
    real = eval_behavior.ROOT / "data" / "eval" / "locked_probes.jsonl"
    if not real.exists() or not eval_behavior.LOCKED_MANIFEST.exists():
        pytest.skip("no sealed locked set in this checkout")
    cases = [json.loads(x) for x in real.read_text(encoding="utf-8").splitlines() if x.strip()]
    sneaky = tmp_path / "my_probes.jsonl"  # no manifest beside it, nothing in the name
    sneaky.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n"
                              for c in cases if c.get("category") == "factual"),
                      encoding="utf-8", newline="\n")
    _fake_server(monkeypatch)

    assert eval_behavior.run(URL, 0.0, 60, sneaky, None) == 2
    assert "not the sealed holdout" in capsys.readouterr().out
