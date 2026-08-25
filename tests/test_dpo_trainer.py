"""The DPO trainer's Stage-C hardening (2026-08-10 review).

Each test pins a defect the epoch/beta sweeps paid GPU time to expose:
  - the cancellation detector must fire on the shape that cost three sweeps,
    and stay quiet on healthy opposition,
  - the tail of the train list must be TRAINED, not silently dropped (at the
    adopted 1-epoch recipe a dropped pair was never seen at all),
  - group_split must carry the class tag through, because per-class accuracy
    is the instrument that would have named the cancellation in one run,
  - the saved checkpoint must not carry optimizer state (no --resume path in
    DPO; the AdamW moments were ~1.9 GB of dead weight per artifact,
    measured against the live T5 checkpoint 2026-08-13).
"""
from __future__ import annotations

import math

from dpo_enigma import group_split, surface_collisions


# ---------------------------------------------------------------------------
# surface_collisions
# ---------------------------------------------------------------------------


def test_collision_detector_fires_on_the_measured_shape():
    """The exact Stage-C collision: a short rejected decline contained inside
    a longer chosen decline from another pair. Containment, not Jaccard --
    Jaccard scores this low because the chosen side keeps explaining."""
    records = [
        {"class": "decline", "prompt": "q1",
         "chosen": "I can't know that -- it's hers, and nobody's told me.",
         "rejected": "Size 8, I believe."},
        {"class": "answer", "prompt": "q2",
         "chosen": "Tokyo.",
         "rejected": "I can't know that."},
    ]
    hits = surface_collisions(records)
    assert hits, "the measured cancellation shape was not detected"
    i, j, score = hits[0]
    assert (i, j) == (0, 1)
    # == 1.0, not >= 0.6: the detector only ever appends hits >= threshold,
    # so a >= assertion was a tautology (audit 2026-08-13). Full containment
    # is the measured value AND what a Jaccard revert fails (4/10 = 0.4).
    assert score == 1.0


def test_collision_detector_is_quiet_on_healthy_opposition():
    """Identity pairs oppose CONTENT, not surfaces -- no warning there. The
    second fixture sits just UNDER the threshold (containment 0.5) so a
    loosened threshold fails here, not only a catastrophically broken
    detector (audit 2026-08-13: the old zero-overlap fixture proved
    nothing about 0.6)."""
    records = [
        {"class": "identity", "prompt": "who are you?",
         "chosen": "I'm Enigma -- from-scratch weights, running right here.",
         "rejected": "I'm ChatGPT, a large language model trained by OpenAI."},
        {"class": "identity", "prompt": "who made you?",
         "chosen": "My user built me from scratch on this machine.",
         "rejected": "I was created by Google and run on Google's cloud."},
    ]
    assert surface_collisions(records) == []

    near = [
        {"class": "a", "prompt": "p1",
         "chosen": "alpha beta gamma delta",
         "rejected": "unrelated words entirely here"},
        {"class": "b", "prompt": "p2",
         "chosen": "other tokens completely",
         "rejected": "alpha beta xray yankee"},  # 2/4 = 0.5 vs chosen[0]
    ]
    assert surface_collisions(near) == []


def test_collision_detector_ignores_a_pair_against_itself():
    """A record's own chosen/rejected overlap is the PREFERENCE, not a
    collision -- only cross-pair overlap cancels. The fixture's self-overlap
    is FULL containment (1.0), so only the i == j exclusion keeps the result
    empty -- the old fixture sat at 0.57, below threshold, and deleting the
    exclusion stayed green (audit 2026-08-13)."""
    records = [
        {"class": "decline", "prompt": "q",
         "chosen": "I can't know that, and I won't guess.",
         "rejected": "I can't know that, and I won't guess. It is blue."},
    ]
    assert surface_collisions(records) == []


def test_collision_detector_fires_on_the_live_focused_file():
    """The committed dpo_focused.jsonl CARRIES the measured collision (the
    graded decline phrases sit chosen in 63 pairs and rejected in 44). If a
    future rebuild removes it, this test flips to guard the opposite claim --
    update it WITH the corpus, from measurement."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "data" / "sft" / "dpo_focused.jsonl"
    if not path.exists():
        import pytest
        pytest.skip("dpo_focused.jsonl not built (gitignored data)")
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    hits = surface_collisions(records)
    assert hits, "the focused set's measured collision is no longer detected"
    classes = {(records[i].get("class"), records[j].get("class")) for i, j, _ in hits}
    assert any(c == "decline" for c, _ in classes), classes


# ---------------------------------------------------------------------------
# tail coverage + group_split class plumbing
# ---------------------------------------------------------------------------


def _fake_pairs(n: int, cls: str = "decline"):
    # (group_key, class, chosen_row, rejected_row) -- rows are opaque here
    return [(f"prompt-{i}", cls, ("c", i), ("r", i)) for i in range(n)]


def test_every_pair_trains_every_epoch():
    """The old loop ran len // mb batches; with 146 train rows and mb 4 the
    last 2 rows were dropped EVERY epoch -- and at 1 epoch, never trained
    (T5's own log: 333 train rows, 83 steps x 4 = 332). Rewritten on the
    batches() seam (2026-08-13): the walk is now BEHAVIOR on the trainer's
    own function, not a re-implemented loop; the one remaining source pin is
    that the train loop actually consumes that function."""
    from pathlib import Path

    from dpo_enigma import batches

    rows = [("cls", ("c", i), ("r", i)) for i in range(146)]
    got = list(batches(rows, 4))
    assert sum(len(b) for b in got) == 146, "the tail is dropped again"
    assert len(got) == math.ceil(146 / 4)
    assert list(batches([], 4)) == []
    assert list(batches(rows, 1000)) == [rows]

    src = (Path(__file__).resolve().parents[1] / "dpo_enigma.py").read_text(encoding="utf-8")
    assert "for batch in batches(train_pairs" in src, \
        "the train loop no longer consumes batches() -- the tail guarantee is untested"
    assert "range(0, steps_per_epoch * args.micro_batch" not in src, \
        "the tail-dropping slice bound is back"


def test_group_split_carries_the_class_tag():
    pairs = _fake_pairs(40, cls="decline") + [
        (f"ans-{i}", "answer", ("c", 100 + i), ("r", 100 + i)) for i in range(40)
    ]
    train, val = group_split(pairs, val_frac=0.10, seed=7)
    assert train and val
    for cls, _c, _r in train + val:
        assert cls in ("decline", "answer")
    # both classes still present on the train side
    assert {cls for cls, _, _ in train} == {"decline", "answer"}


def test_group_split_still_groups_by_prompt():
    """Same prompt, two records -> same side (the 2026-07-15 leak)."""
    pairs = [("dup", "identity", ("dup-c", 1), ("dup-r", 1)),
             ("dup", "identity", ("dup-c", 2), ("dup-r", 2))] + _fake_pairs(20)
    train, val = group_split(pairs, val_frac=0.3, seed=3)
    dup_train = sum(1 for cls, c, _ in train if c[0] == "dup-c")
    dup_val = sum(1 for cls, c, _ in val if c[0] == "dup-c")
    assert (dup_train, dup_val) in ((2, 0), (0, 2)), "twin records split across sides"


# ---------------------------------------------------------------------------
# checkpoint contents
# ---------------------------------------------------------------------------


def test_dpo_save_carries_no_optimizer_state():
    """Behavior pin (was a regex over source -- audit 2026-08-13): the
    payload builder is the ONE place the save dict is assembled, so its key
    set is the contract. DPO has no --resume, so optimizer state is
    unreadable dead weight (~1.9 GB per artifact, measured on the live T5
    checkpoint 2026-08-13)."""
    from dpo_enigma import artifact_payload

    payload = artifact_payload(model_state={"w": 1}, config_dict={"dim": 8},
                               step=42, schedule={"trainer": "dpo"}, meta={"m": 1})
    assert set(payload) == {"model_state_dict", "config", "step", "schedule", "meta"}, \
        "the DPO checkpoint payload changed shape -- optimizer state (or worse) crept in"
    assert payload["step"] == 42
    assert payload["model_state_dict"] == {"w": 1}

    # Call-site pin: the builder being clean means nothing if main() stops
    # routing the save through it (audit 2026-08-13). Two occurrences = the
    # def and the one call inside atomic_torch_save(...).
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "dpo_enigma.py").read_text(encoding="utf-8")
    assert src.count("artifact_payload(") == 2, \
        "main no longer routes the save through artifact_payload"


# ---------------------------------------------------------------------------
# Round-1 hardening (2026-08-13 audit): CLI safety, dark instrument,
# triageable advisory, save-before-report
# ---------------------------------------------------------------------------


def test_out_is_required(capsys):
    """--out used to default to models/enigma_dpo -- the v8 ROLLBACK dir. A
    default run would have overwritten the revert target (audit 2026-08-13).
    The stderr assertion keeps this from going vacuous the day a SECOND
    required arg appears (argparse exits 2 for any missing one)."""
    import pytest

    from dpo_enigma import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--init", "x.pth"])
    assert "--out" in capsys.readouterr().err
    args = build_parser().parse_args(["--init", "x.pth", "--out", "models/scratch_run"])
    assert args.out == "models/scratch_run"


def test_refuses_an_out_dir_that_already_holds_a_model(tmp_path):
    """pretokenize's versioned-never-rebuilt rule, third writer: refuse at
    STARTUP, before any GPU time, if the target already holds a model.pth."""
    import pytest

    from dpo_enigma import refuse_existing_artifact

    taken = tmp_path / "run1"
    taken.mkdir()
    (taken / "model.pth").write_bytes(b"weights")
    with pytest.raises(SystemExit) as exc:
        refuse_existing_artifact(taken)
    assert "model.pth" in str(exc.value)

    refuse_existing_artifact(tmp_path / "run2")  # no dir yet -> fine
    empty = tmp_path / "run3"
    empty.mkdir()
    refuse_existing_artifact(empty)  # dir exists but holds no model -> fine

    # An interrupted run leaves latest.pth/prev.pth and no model.pth --
    # those are receipts too (review 2026-08-13).
    rotated = tmp_path / "run4"
    rotated.mkdir()
    (rotated / "latest.pth").write_bytes(b"weights")
    with pytest.raises(SystemExit):
        refuse_existing_artifact(rotated)
    only_prev = tmp_path / "run5"
    only_prev.mkdir()
    (only_prev / "prev.pth").write_bytes(b"weights")
    with pytest.raises(SystemExit):
        refuse_existing_artifact(only_prev)

    # --out naming a FILE is the model.pth-typo class: <file>/model.pth
    # never exists, so the old guard passed and a DPO run trained to
    # completion before dying at the post-training mkdir (audit 2026-08-13).
    with pytest.raises(SystemExit):
        refuse_existing_artifact(taken / "model.pth")
    # ...and a .pth NAME is the same typo even when nothing exists there yet.
    with pytest.raises(SystemExit):
        refuse_existing_artifact(tmp_path / "fresh_name.pth")


def test_sanity_may_aim_at_an_existing_dir(tmp_path):
    """--sanity never writes, so refusing it was pointless friction (review
    2026-08-13); every WRITING run still refuses. Args come from the REAL
    parser so a renamed argparse dest cannot desync guard and CLI."""
    import pytest

    from dpo_enigma import build_parser, startup_artifact_guard

    taken = tmp_path / "run1"
    taken.mkdir()
    (taken / "model.pth").write_bytes(b"weights")
    startup_artifact_guard(build_parser().parse_args(
        ["--init", "x.pth", "--out", str(taken), "--sanity"]))
    with pytest.raises(SystemExit):
        startup_artifact_guard(build_parser().parse_args(
            ["--init", "x.pth", "--out", str(taken)]))


def test_main_refuses_before_touching_the_checkpoint(tmp_path, monkeypatch):
    """The guard must be WIRED into main and fire FIRST -- a correct guard
    nobody calls protects nothing, and no test drove main() until this one
    (audit 2026-08-13). The --init here does not exist, so getting past the
    guard would raise init-not-found instead of the refusal demanded."""
    import pytest

    import dpo_enigma

    taken = tmp_path / "run"
    taken.mkdir()
    (taken / "model.pth").write_bytes(b"weights")
    monkeypatch.setattr("sys.argv", ["dpo_enigma.py", "--init", str(tmp_path / "absent.pth"),
                                     "--out", str(taken)])
    with pytest.raises(SystemExit, match="already exists"):
        dpo_enigma.main()


def test_an_untagged_corpus_warns_that_the_instrument_is_dark():
    """dpo_pairs.jsonl carries ZERO class tags (measured 2026-08-13), so the
    per-class report silently collapses to the blended number it exists to
    replace. The trainer must SAY so, and must stay quiet when real tags are
    present."""
    from dpo_enigma import untagged_warning

    msg = untagged_warning({"untagged": 350})
    assert msg is not None and "dpo_focused.jsonl" in msg
    assert untagged_warning({"decline": 63, "answer": 28}) is None
    assert untagged_warning({"decline": 63, "untagged": 5}) is None


def test_collision_report_is_triageable():
    """584 hits with ONE example printed was unreadable (audit 2026-08-13).
    Rebuilt on a fixture the first version could not discriminate with (3
    records = 3 hits, all scores 1.0): 5 records, 6 hits, distinct scores,
    asymmetric class-pair counts -- so a wrong count, swapped orientation,
    missing sort, or broken [:3] cap each fail a distinct assertion."""
    from dpo_enigma import collision_report

    records = [
        {"class": "decline", "prompt": "q1",
         "chosen": "I can't know that -- it's hers, and nobody's told me.",
         "rejected": "Purple, probably."},
        {"class": "answer", "prompt": "q2",
         "chosen": "Tokyo.",
         "rejected": "nobody's told me anything real"},
        {"class": "decline", "prompt": "q3",
         "chosen": "I can't know that.",
         "rejected": "Size 8, I believe."},
        {"class": "answer", "prompt": "q4",
         "chosen": "Paris.",
         "rejected": "I can't know that."},
        {"class": "memory_recall", "prompt": "q5",
         "chosen": "I can't know that -- you haven't told me.",
         "rejected": "I can't know that either."},
    ]
    hits = surface_collisions(records)
    assert len(hits) == 6, f"fixture drifted: {hits}"
    scores = sorted(s for _, _, s in hits)
    assert scores[0] < 0.7 < scores[-1], "fixture no longer carries distinct scores"

    report = collision_report(records, hits)
    assert "WARN" in report
    assert "6 chosen/rejected string pairs" in report      # hit count, not record count (5)
    # per-class-pair counts WITH orientation -- a chosen/rejected swap fails
    assert "decline(chosen)/answer(rejected)=3" in report
    assert "decline(chosen)/memory_recall(rejected)=2" in report
    assert "memory_recall(chosen)/answer(rejected)=1" in report
    # top-3 cap on 6 hits, ranked by score: the 0.60 and 0.80 hits fall off,
    # which an unsorted [:3] would NOT achieve (the 0.60 hit is hits[0])
    assert report.count("vs rejected") == 3
    assert "1.00: chosen" in report
    assert "0.60" not in report and "0.80" not in report


def test_the_final_report_runs_after_the_save():
    """The end-of-run class_accuracy pass is a full forward over every train
    row; sitting between train-end and the save it could OOM and eat the
    finished checkpoint (audit 2026-08-13). No honest behavior seam exists
    short of running the trainer, so the source ORDER is the contract."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "dpo_enigma.py").read_text(encoding="utf-8")
    # Uniqueness first: .index() reads the FIRST occurrence, so a future
    # comment containing either string would silently disarm the order
    # assertion (audit 2026-08-13).
    assert src.count("atomic_torch_save(") == 1
    assert src.count("class_accuracy(train_pairs)") == 1
    assert src.index("atomic_torch_save(") < src.index("class_accuracy(train_pairs)"), \
        "the class report moved back between training and the save"


# ---------------------------------------------------------------------------
# --block vs the model's context window
# ---------------------------------------------------------------------------


def test_over_block_refused():
    import pytest

    import dpo_enigma

    with pytest.raises(SystemExit) as e:
        dpo_enigma._refuse_over_block(4096, 2048)
    assert "max_seq_len" in str(e.value)


def test_at_or_under_block_passes():
    import dpo_enigma

    dpo_enigma._refuse_over_block(2048, 2048)
    dpo_enigma._refuse_over_block(1024, 2048)


def test_block_guard_sits_on_the_load_path():
    import inspect

    import dpo_enigma

    src = inspect.getsource(dpo_enigma.main)
    assert "_refuse_over_block(" in src
