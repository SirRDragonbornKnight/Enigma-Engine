"""finetune's --out defaulted to models/enigma_sft -- an EXISTING artifact
(the v8 lineage's SFT receipt, the dpo_from provenance chain) -- with no
refusal guard, so a bare default run would have overwritten the receipt with
a v2-inited model (review 2026-08-13; the DPO treatment, fourth writer).

The resume exemption is derived from the DIR RELATIONSHIP, not the flag: a
stale copy-pasted --out on a resume used to rotate over whatever artifact it
named (audit 2026-08-13). --sanity never writes (the out mkdir sits below
the sanity return). Args come from the REAL parser throughout, so a renamed
argparse dest cannot desync guard and CLI.
"""
from __future__ import annotations

import pytest


def _args(extra):
    from finetune_enigma import build_parser

    return build_parser().parse_args(["--data", "d.jsonl"] + extra)


def test_out_is_required(capsys):
    from finetune_enigma import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--data", "d.jsonl"])
    assert "--out" in capsys.readouterr().err
    args = _args(["--out", "models/scratch_run"])
    assert args.out == "models/scratch_run"


def test_a_fresh_run_refuses_an_existing_artifact(tmp_path):
    from finetune_enigma import startup_artifact_guard

    taken = tmp_path / "run1"
    taken.mkdir()
    # the rotation name, not just model.pth -- an interrupted run leaves
    # latest.pth/prev.pth and no model.pth, and those are receipts too
    (taken / "latest.pth").write_bytes(b"weights")
    with pytest.raises(SystemExit, match="already exists"):
        startup_artifact_guard(_args(["--out", str(taken)]))


def test_resume_is_exempt_only_into_its_own_dir(tmp_path):
    """Resume continuing ITS OWN dir is the one sanctioned in-place write;
    resume with a DIVERGENT --out that holds an artifact refuses like any
    other run (the stale copy-pasted --out hole, audit 2026-08-13)."""
    from finetune_enigma import startup_artifact_guard

    own = tmp_path / "runA"
    own.mkdir()
    (own / "latest.pth").write_bytes(b"weights")
    serving = tmp_path / "serving"
    serving.mkdir()
    (serving / "model.pth").write_bytes(b"weights")

    startup_artifact_guard(_args(["--resume", str(own / "latest.pth"), "--out", str(own)]))
    startup_artifact_guard(_args(["--resume", str(own / "latest.pth"),
                                  "--out", str(tmp_path / "fresh")]))
    with pytest.raises(SystemExit, match="already exists"):
        startup_artifact_guard(_args(["--resume", str(own / "latest.pth"),
                                      "--out", str(serving)]))


def test_sanity_may_aim_at_an_existing_dir(tmp_path):
    from finetune_enigma import startup_artifact_guard

    taken = tmp_path / "run1"
    taken.mkdir()
    (taken / "latest.pth").write_bytes(b"weights")
    startup_artifact_guard(_args(["--out", str(taken), "--sanity"]))
    startup_artifact_guard(_args(["--out", str(tmp_path / "fresh")]))


def test_main_refuses_before_touching_anything(tmp_path, monkeypatch):
    """The guard must be WIRED into main and fire FIRST (audit 2026-08-13:
    no test drove either trainer's main). --data and --init both point at
    nothing, so getting past the guard raises a DIFFERENT error than the
    refusal demanded here."""
    import finetune_enigma

    taken = tmp_path / "run"
    taken.mkdir()
    (taken / "model.pth").write_bytes(b"weights")
    monkeypatch.setattr("sys.argv", ["finetune_enigma.py",
                                     "--data", str(tmp_path / "absent.jsonl"),
                                     "--init", str(tmp_path / "absent.pth"),
                                     "--out", str(taken)])
    with pytest.raises(SystemExit, match="already exists"):
        finetune_enigma.main()


def test_sanity_does_not_create_the_out_dir():
    """--sanity used to mkdir --out before returning, which then tripped the
    launchers' own first-launch Test-Path guard on the real run (audit
    2026-08-13). No cheap behavior seam exists short of a full sanity run
    (it needs a real checkpoint + data), so the source ORDER is the
    contract: the one mkdir of --out sits AFTER the sanity return."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "finetune_enigma.py").read_text(encoding="utf-8")
    assert src.count("out.mkdir(") == 1, "the --out mkdir is no longer unique -- re-audit this pin"
    assert src.count("[sanity]") == 1, "the sanity print marker is no longer unique -- re-audit this pin"
    assert src.index("[sanity]") < src.index("out.mkdir("), \
        "the --out mkdir moved back above the sanity return"
