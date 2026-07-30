"""Arc-C v2 trainer pieces: the wsd_sqrt schedule and post-hoc checkpoint EMA.

wsd_sqrt = the IMU-1 ablation's shape (hold at peak, 1-sqrt(t) decay to a
0.01x floor). ema_checkpoints.py = offline EMA over one lineage's archived
checkpoints (beta 0.8 over the final ~10 = the recipe's free win); it must
refuse cross-lineage input rather than average incompatible weights.
"""

from __future__ import annotations

import math

import pytest
import torch

import ema_checkpoints
import pretrain_enigma
from ema_checkpoints import ema_state_dicts
from enigma_engine.core.optim import get_lr

PEAK = 6e-4
TOTAL = 1000
WARMUP = 100


def _lr(step: int) -> float:
    return get_lr(step, WARMUP, TOTAL, PEAK, schedule="wsd_sqrt", decay_frac=0.1)


# ---------------------------------------------------------------------------
# wsd_sqrt schedule
# ---------------------------------------------------------------------------


def test_wsd_sqrt_warmup_and_hold():
    assert _lr(0) == PEAK * 1 / WARMUP
    assert _lr(WARMUP) == PEAK
    assert _lr(899) == PEAK  # last hold step before decay_start=900


def test_wsd_sqrt_decays_to_floor_not_zero():
    assert _lr(TOTAL) == pytest.approx(0.01 * PEAK)
    assert _lr(TOTAL + 50) == pytest.approx(0.01 * PEAK)  # continuation stays at floor


def test_wsd_sqrt_shape_is_one_minus_sqrt():
    # Midpoint of the decay window (step 950 of 900..1000): t = 0.5.
    expected = PEAK * (0.01 + 0.99 * (1 - math.sqrt(0.5)))
    assert _lr(950) == pytest.approx(expected)


def test_wsd_sqrt_monotone_nonincreasing_after_warmup():
    lrs = [_lr(s) for s in range(WARMUP, TOTAL + 1)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))


def test_existing_schedules_untouched():
    # cosine and wsd are checkpoint-recorded contracts; adding wsd_sqrt must
    # not perturb them.
    assert get_lr(500, WARMUP, TOTAL, PEAK) == pytest.approx(
        PEAK * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (500 - WARMUP) / (TOTAL - WARMUP))))
    )
    assert get_lr(TOTAL - 1, WARMUP, TOTAL, PEAK, schedule="wsd", decay_frac=0.1) == pytest.approx(
        PEAK * 1 / 100
    )


# ---------------------------------------------------------------------------
# EMA math
# ---------------------------------------------------------------------------


def _sd(value: float, dtype=torch.float32) -> dict:
    return {"w": torch.full((4, 4), value, dtype=dtype), "count": torch.tensor([1])}


def test_ema_math_beta_08():
    out = ema_state_dicts([_sd(0.0), _sd(1.0)], beta=0.8)
    assert torch.allclose(out["w"], torch.full((4, 4), 0.2))
    # three-way: ((0*0.8 + 1*0.2)*0.8 + 2*0.2) = 0.56
    out3 = ema_state_dicts([_sd(0.0), _sd(1.0), _sd(2.0)], beta=0.8)
    assert torch.allclose(out3["w"], torch.full((4, 4), 0.56))


def test_ema_int_tensors_take_last_and_dtype_preserved():
    a, b = _sd(0.0, torch.bfloat16), _sd(1.0, torch.bfloat16)
    a["count"] = torch.tensor([7])
    b["count"] = torch.tensor([9])
    out = ema_state_dicts([a, b], beta=0.8)
    assert out["w"].dtype == torch.bfloat16
    assert out["count"].item() == 9


def test_ema_refuses_bad_input():
    with pytest.raises(ValueError, match=">= 2"):
        ema_state_dicts([_sd(0.0)], beta=0.8)
    bad_keys = {"other": torch.zeros(4, 4)}
    with pytest.raises(ValueError, match="key set"):
        ema_state_dicts([_sd(0.0), bad_keys], beta=0.8)
    bad_shape = {"w": torch.zeros(2, 2), "count": torch.tensor([1])}
    with pytest.raises(ValueError, match="shape"):
        ema_state_dicts([_sd(0.0), bad_shape], beta=0.8)


# ---------------------------------------------------------------------------
# CLI: lineage guards + output format
# ---------------------------------------------------------------------------


VALID_CFG = {"vocab_size": 64, "dim": 32, "n_layers": 1, "n_heads": 2, "max_seq_len": 32}


def _ckpt(tmp_path, name: str, value: float, step: int, config=None):
    p = tmp_path / name
    torch.save(
        {
            "model_state_dict": {"w": torch.full((4, 4), value)},
            "config": config or dict(VALID_CFG),
            "step": step,
        },
        p,
    )
    return p


def test_cli_writes_loadable_ema(tmp_path):
    a = _ckpt(tmp_path, "a.pth", 0.0, 100)
    b = _ckpt(tmp_path, "b.pth", 1.0, 200)
    out = tmp_path / "ema.pth"
    assert ema_checkpoints.main([str(a), str(b), "--out", str(out), "--beta", "0.8"]) == 0
    ck = torch.load(out, weights_only=False)
    assert torch.allclose(ck["model_state_dict"]["w"], torch.full((4, 4), 0.2))
    assert ck["config"] == dict(VALID_CFG)
    assert ck["step"] == 200
    assert ck["ema"]["beta"] == 0.8 and ck["ema"]["n"] == 2


def test_cli_refuses_wrong_order(tmp_path):
    a = _ckpt(tmp_path, "a.pth", 0.0, 200)
    b = _ckpt(tmp_path, "b.pth", 1.0, 100)
    with pytest.raises(SystemExit, match="oldest first"):
        ema_checkpoints.main([str(a), str(b), "--out", str(tmp_path / "e.pth")])


def test_the_anneal_starts_with_the_decay_phase():
    """The recipe's "anneal on the best tokens" had nowhere to land: the
    sampler drew uniformly over the whole train stream, so a file of curated
    tokens changed nothing about what the model saw."""
    assert pretrain_enigma.anneal_first_step(1000, 0.10) == 900
    assert pretrain_enigma.anneal_first_step(1000, 0.0) == 1000   # no decay, no anneal
    assert pretrain_enigma.anneal_first_step(1000, 1.0) == 0      # all decay


@pytest.mark.parametrize("frac,curated", [(0.0, 0), (0.25, 2), (0.5, 4), (1.0, 8)])
def test_the_anneal_fraction_splits_the_micro_batch(frac, curated):
    assert pretrain_enigma.anneal_counts(8, frac) == (curated, 8 - curated)


@pytest.mark.parametrize("frac", [-1.0, 1.5, float("nan")])
def test_an_out_of_range_anneal_fraction_cannot_break_the_batch(frac):
    """This runs INSIDE the training loop and first fires at the decay
    boundary, days into a run. A negative count would ask numpy for a negative
    draw size, one over the batch size would return more rows than the step
    expects, and NaN raised outright -- all at the most expensive moment."""
    curated, general = pretrain_enigma.anneal_counts(8, frac)
    assert 0 <= curated <= 8
    assert curated + general == 8


def test_the_anneal_is_off_by_default_and_is_run_math_when_on():
    """Off, the sampler is the same single uniform draw the live lineage used.
    On, it changes WHAT the decay phase sees, so it belongs in the checkpoint's
    schedule -- a resume that dropped it would finish the tail on a different
    diet than it started."""
    import inspect

    assert "anneal_tokens" in pretrain_enigma.SCHEDULE_KEYS \
        and "anneal_frac" in pretrain_enigma.SCHEDULE_KEYS, \
        "the anneal must be recorded in the schedule, not treated as a CLI knob"
    src = inspect.getsource(pretrain_enigma.main)
    assert "--anneal-tokens" in src
    # default 0 is what keeps the live lineage's data order untouched
    after = src.split('"--anneal-tokens"', 1)[1][:300]
    assert "default=0" in after.replace(" ", "")
    # and the sampler only diverges from the uniform draw when it is ON
    assert "anneal_lo is not None" in src


def test_an_anneal_region_that_dies_at_the_decay_boundary_is_refused_at_boot():
    """Both bad sizes pass argparse, boot clean, and then kill the run DAYS in,
    at the decay phase's first curated draw -- and a resume restores the same
    schedule from the checkpoint and dies at the same step (round-7 audit,
    2026-07-25: --anneal-tokens <= block+1 gave randint an empty range). The
    only helpful refusal is the one at boot."""
    region = pretrain_enigma.anneal_region
    # off: no region, no complaint
    assert region(0, train_end=10_000, block=64) is None
    # a healthy region is exactly train_end - N
    assert region(1_000, train_end=10_000, block=64) == 9_000
    # the boundary landmine: too small to fit one sample window + draw position
    with pytest.raises(SystemExit, match="decay boundary"):
        region(65, train_end=10_000, block=64)  # == block + 1
    with pytest.raises(SystemExit, match="decay boundary"):
        region(1, train_end=10_000, block=64)
    # the smallest legal region boots (block + 2 leaves one draw position)
    assert region(66, train_end=10_000, block=64) == 10_000 - 66
    # the other end: no general region left ahead of the curated one
    with pytest.raises(SystemExit, match="general region"):
        region(9_990, train_end=10_000, block=64)
    # and negative stays refused
    with pytest.raises(SystemExit, match="negative"):
        region(-1, train_end=10_000, block=64)


def test_source_placement_that_defeats_training_is_refused():
    """The T1 curated shard was designed to tokenize LAST so the anneal could
    read it -- and the END of the bin is exactly what val is carved from, so
    the shard would have landed 100% in val and never been trained on (round-7
    audit, 2026-07-25). The corpus metadata now records each source's token
    extent and boot checks every coupling its own fix-arc audit then found
    still open: an UN-repeated source can lie entirely in val (one absent
    stackexchange dir away on a fresh checkout), a repeated one can overlap
    the fenced val-gen window, and a per-pass span under one block makes the
    copies co-occupy a training window."""
    refuse = pretrain_enigma.refuse_repeated_source_in_val
    meta = {"repeated_sources": {"Curated": 5},
            "source_token_extents": {"Curated": [100, 900], "SE/x": [900, 2000]}}
    # curated ends before the val tail: fine (SE STRADDLING val is the norm)
    refuse(meta, train_end=1_500)
    # repeated source reaches into val: refused
    with pytest.raises(SystemExit, match="val tail"):
        refuse(meta, train_end=800)
    # ANY source lying entirely in val is refused, repeated or not -- this is
    # the round-7 failure reappearing with nothing declared
    plain = {"source_token_extents": {"Curated": [1_600, 2_000], "A": [0, 1_600]}}
    with pytest.raises(SystemExit, match="entirely in the val tail"):
        refuse(plain, train_end=1_500)
    # a repeated source overlapping the fenced val-gen window is refused: the
    # fence redraws around it, so those copies are held out while their train
    # siblings memorize the eval window
    with pytest.raises(SystemExit, match="fenced window"):
        refuse(meta, train_end=1_500, fenced=((500, 700),))
    refuse(meta, train_end=1_500, fenced=((900, 1_100),))  # disjoint: fine
    # per-pass span must exceed one training window, or every copy shares it
    with pytest.raises(SystemExit, match="per-pass span"):
        refuse(meta, train_end=1_500, block=160)  # span 800 / x5 = 160 <= block
    refuse(meta, train_end=1_500, block=159)
    # a corpus whose metadata predates the extents record is untouched
    refuse({}, train_end=10)
    refuse({"repeated_sources": {"Curated": 5}}, train_end=10)


def test_source_val_windows_hold_out_every_source_but_never_a_repeated_one():
    """One window per source is the representative val signal a contiguous
    window cannot give ([val-gen] measured 100% FineWeb-Edu at every
    --val-tokens, 2026-07-28). The two adversarial cases are the ones that
    would silently corrupt training if the builder got them wrong: a REPEATED
    source's window holds one copy out while its train siblings memorize it,
    and an unclipped window past train_end would overlap the val tail."""
    windows = pretrain_enigma.source_val_windows
    meta = {
        "repeated_sources": {"Curated": 5},
        "source_token_extents": {
            "Curated": [0, 10_000],       # repeated: must get NO window
            "Web": [10_000, 50_000],      # plain: window at its tail
            "SE/x": [50_000, 100_000],    # straddles train_end: clipped
            "Tiny": [100_000, 100_050],   # narrower than block+2 after clip
            "Broken": "not-an-extent",    # malformed: skipped, not crashed
        },
    }
    got = windows(meta, train_end=60_000, block=64, per_source=1_000)
    labels = [w[0] for w in got]
    # the repeated source is EXCLUDED even though it is wide enough -- this is
    # the assertion that fails if the repeated-check is dropped
    assert "Curated" not in labels
    assert got == [("Web", 49_000, 50_000), ("SE/x", 59_000, 60_000)]
    # the clipped window ends at train_end, not at the extent's true end --
    # an unclipped builder returns 99_000/100_000 here and leaks the val tail
    assert all(hi <= 60_000 for _, _, hi in got)
    # off switch and pre-extents corpora produce no windows, not a crash
    assert windows(meta, train_end=60_000, block=64, per_source=0) == []
    assert windows({}, train_end=60_000, block=64, per_source=1_000) == []


def test_the_per_source_fences_survive_a_resume():
    """The windows are FENCES. A resume that dropped --val-per-source would
    let train sampling eat every held-out window, and the rest of the run's
    [val-src] would score data it just trained on. Same trap class as
    no_grad_ckpt; membership in SCHEDULE_KEYS is the whole fix."""
    assert "val_per_source" in pretrain_enigma.SCHEDULE_KEYS


def test_resume_restores_the_grad_checkpoint_flag():
    """--no-grad-ckpt DISABLES a config default that is on, and costs 30-40%
    throughput. Unrecorded in the schedule, a bare --resume reverts it to the
    argparse default and the rest of a multi-day run finishes at two thirds
    speed. The restore is a blind setattr over the recorded dict, so the whole
    fix is membership -- and the negative half is what pins it: a schedule
    MISSING the key leaves the CLI default standing."""
    assert "no_grad_ckpt" in pretrain_enigma.SCHEDULE_KEYS
    # --seed must NOT be restored: re-seeding replays the sampler from step 0.
    assert "seed" not in pretrain_enigma.SCHEDULE_KEYS
    # the archive cadence is sized against the launch's decay tail, so a resume
    # that re-imposed a caller's value would lose it
    assert "archive_every" in pretrain_enigma.SCHEDULE_KEYS
    # the pause-on-a-checkpoint protocol assumes the save cadence survives a
    # resume; unrecorded, a bare --resume reverted it to 250 with no boot line
    assert "save_every" in pretrain_enigma.SCHEDULE_KEYS

    # The restore is a blind setattr over the recorded dict, so membership is
    # the whole fix -- PROVIDED the restore runs before the flag is read. That
    # ordering is the part a dict test cannot see, and it is what makes the
    # difference between a restored flag and a decorative one.
    import inspect

    src = inspect.getsource(pretrain_enigma.main)
    restore = src.index("setattr(args, k, v)")
    applied = src.index("if not args.no_grad_ckpt:")
    assert restore < applied, \
        "the schedule restore must run BEFORE --no-grad-ckpt is read, or the " \
        "recorded value never reaches the model"

    # ...and a checkpoint predating the key falls back to the CLI default,
    # which is the pre-fix behaviour. Every lineage on disk today is in that
    # state, so the gap must be computable and reported rather than assumed
    # away. (The boot message itself is verified by running a real legacy
    # checkpoint, not here -- a string match on source would pass with the
    # message deleted from the branch that prints it.)
    legacy = {"lr": 3e-3, "block": 2048}
    missing = [k for k in pretrain_enigma.SCHEDULE_KEYS if k not in legacy]
    assert "no_grad_ckpt" in missing
    # ...which is exactly why a checkpoint predating the key must SAY so. Every
    # lineage on disk today was written without it, so this is the common path,
    # not the edge case.
    legacy = {"lr": 3e-3, "block": 2048}
    missing = [k for k in pretrain_enigma.SCHEDULE_KEYS if k not in legacy]
    assert "no_grad_ckpt" in missing


def test_val_source_split_is_reported_and_a_single_domain_warns(capsys):
    """val is the last val_n tokens, so its domain is whatever the walk placed
    last. On v2b that is one StackExchange site, and every [val] number the run
    prints is that site's loss under the name "val". The warning must fire on
    the dominated split and STAY SILENT on a mixed one, or it is noise."""
    report = pretrain_enigma.report_val_sources
    # one source owns the whole tail -> named, and warned about
    solo = {"source_token_extents": {"Web": [0, 900], "SE/worldbuilding": [900, 1000]}}
    share = report(solo, train_end=900, n=1000)
    out = capsys.readouterr().out
    assert share == [("SE/worldbuilding", 100)]
    assert "WARNING" in out and "SE/worldbuilding" in out

    # an evenly mixed tail is reported but NOT warned about
    mixed = {"source_token_extents": {"Web": [0, 950], "SE/x": [950, 1000]}}
    share = report(mixed, train_end=900, n=1000)
    out = capsys.readouterr().out
    assert sorted(share) == [("SE/x", 50), ("Web", 50)]
    assert "val sources (2)" in out
    assert "WARNING" not in out

    # with a [val-gen] window covering it, the split is stated but NOT warned
    # about -- the recommended launch line must not cry wolf on every boot
    report(solo, train_end=900, n=1000, have_general_window=True)
    out = capsys.readouterr().out
    assert "SE/worldbuilding" in out and "val-gen" in out
    assert "WARNING" not in out

    # a corpus predating the extents record cannot be checked -- and must SAY
    # so. Silence here reads as "val is fine", which is the one thing it does
    # not establish.
    assert report({}, train_end=900, n=1000) == []
    out = capsys.readouterr().out
    assert "not recorded" in out and "unknown" in out
    assert "WARNING" not in out, "an uncheckable corpus is not a finding"


def test_the_general_window_is_measured_not_just_pointed_at(capsys):
    """[val-gen] is contiguous too, so it lands inside ONE source as easily as
    the tail does. Telling the operator to read [val-gen] while never checking
    what [val-gen] contains moves the mislabel instead of fixing it -- on the
    real v2b corpus the recommended --val-general-end yields a window that is
    100% FineWeb-Edu, and nothing said so."""
    report = pretrain_enigma.report_val_sources
    # tail is one site AND the general window is one source: the tail note is
    # fine, but the window the operator is sent to must raise its own alarm.
    meta = {"source_token_extents": {
        "Web": [0, 500], "Edu": [500, 900], "SE/worldbuilding": [900, 1000]}}
    report(meta, train_end=900, n=1000, have_general_window=True, val_gen=(800, 900))
    out = capsys.readouterr().out
    assert "val-gen sources (1)" in out and "Edu" in out
    assert "WARNING" in out and "[val-gen]" in out, (
        "a single-source general window must warn -- this is the case that shipped silent"
    )

    # a general window spanning several sources is reported and NOT warned
    # about, so the warning stays worth reading.
    report(meta, train_end=900, n=1000, have_general_window=True, val_gen=(300, 700))
    out = capsys.readouterr().out
    assert "val-gen sources (2)" in out
    assert "WARNING" not in out, "a mixed general window is the goal, not a finding"


def _stamp(path, meta):
    ck = torch.load(path, weights_only=False)
    ck["meta"] = meta
    torch.save(ck, path)
    return path


def test_the_ema_carries_the_instruct_marker_through(tmp_path):
    """serve reads meta.chat_format to decide a checkpoint is INSTRUCT, and only
    an INSTRUCT checkpoint gets its trained chat/tool rows declared decodable.
    Dropping meta turned an EMA of SFT checkpoints into a BASE-looking file whose
    <|tool_call|> row is masked out of every reply -- the same silent tool death
    the live-vocab fix exists to prevent, re-entered through a lineage tool."""
    instruct = {"chat_format": "enigma-chat-v1"}
    a = _stamp(_ckpt(tmp_path, "a.pth", 0.0, 100), instruct)
    b = _stamp(_ckpt(tmp_path, "b.pth", 1.0, 200), instruct)
    out = tmp_path / "ema.pth"
    assert ema_checkpoints.main([str(a), str(b), "--out", str(out)]) == 0
    assert torch.load(out, weights_only=False)["meta"] == instruct

    # a base lineage must not have one invented for it
    c = _ckpt(tmp_path, "c.pth", 0.0, 100)
    d = _ckpt(tmp_path, "d.pth", 1.0, 200)
    base_out = tmp_path / "base.pth"
    assert ema_checkpoints.main([str(c), str(d), "--out", str(base_out)]) == 0
    assert "meta" not in torch.load(base_out, weights_only=False)

    # ...and mixing lineages is refused, as it already is for config
    with pytest.raises(SystemExit, match="disagree on `meta`"):
        ema_checkpoints.main([str(c), str(b), "--out", str(tmp_path / "mixed.pth")])


def test_cli_refuses_cross_lineage_configs(tmp_path):
    a = _ckpt(tmp_path, "a.pth", 0.0, 100)
    b = _ckpt(tmp_path, "b.pth", 1.0, 200, config={**VALID_CFG, "dim": 64, "n_heads": 4})
    with pytest.raises(SystemExit, match="lineage"):
        ema_checkpoints.main([str(a), str(b), "--out", str(tmp_path / "e.pth")])


def test_cli_accepts_config_key_drift_within_a_lineage(tmp_path):
    # audit 2026-07-20: raw dict equality refused archive windows straddling a
    # code upgrade -- one save carries a retired key, the next carries a field
    # added later. Same architecture must average fine.
    old_era = {**VALID_CFG, "early_exit_layer": 0}  # retired key, filtered on load
    new_era = {**VALID_CFG, "norm_scheme": "pre"}  # explicit default, same arch
    a = _ckpt(tmp_path, "a.pth", 0.0, 100, config=old_era)
    b = _ckpt(tmp_path, "b.pth", 1.0, 200, config=new_era)
    out = tmp_path / "ema.pth"
    assert ema_checkpoints.main([str(a), str(b), "--out", str(out)]) == 0
    assert out.exists()


def test_cli_warns_when_no_steps_recorded(tmp_path, capsys):
    # audit 2026-07-20: the order guard was silently vacuous for stepless
    # checkpoints -- a wrong-order EMA ran to completion with no signal.
    a = _ckpt(tmp_path, "a.pth", 0.0, None)
    b = _ckpt(tmp_path, "b.pth", 1.0, None)
    assert ema_checkpoints.main([str(a), str(b), "--out", str(tmp_path / "e.pth")]) == 0
    assert "CANNOT be" in capsys.readouterr().out


def test_ema_refuses_non_tensor_entries():
    bad = {"w": torch.zeros(4, 4), "count": "seven"}
    with pytest.raises(ValueError, match="not a tensor"):
        ema_state_dicts([_sd(0.0), bad], beta=0.8)


def test_cli_malformed_config_blob_gets_clean_refusal(tmp_path):
    # round-3 audit: only ValueError was caught; a string-typed field raises
    # TypeError inside config normalization and escaped as a raw traceback.
    a = _ckpt(tmp_path, "a.pth", 0.0, 100)
    b = _ckpt(tmp_path, "b.pth", 1.0, 200, config={**VALID_CFG, "vocab_size": "x"})
    with pytest.raises(SystemExit, match="not a valid ForgeConfig"):
        ema_checkpoints.main([str(a), str(b), "--out", str(tmp_path / "e.pth")])


def test_cli_refuses_overwriting_a_source(tmp_path):
    a = _ckpt(tmp_path, "a.pth", 0.0, 100)
    b = _ckpt(tmp_path, "b.pth", 1.0, 200)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        ema_checkpoints.main([str(a), str(b), "--out", str(b)])
