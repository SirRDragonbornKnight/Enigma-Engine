"""Training-log reader (training_progress.py): the only view into a run that
lasts days, and it parses a log line the trainer is free to reformat.

It had no test, and the first reformat blinded it -- adding a windowed rate and
a peak-VRAM figure to the step line made STEP_RE miss every line, so `render()`
printed "No 'step' lines found" with no percentage, ETA, loss or plot for the
whole run. These pin the formats the reader must survive."""

from __future__ import annotations

import training_progress


STEP_LINES = {
    # the v1 lineage's format, still in train_large.log
    "legacy": "step 10250/287882 loss 1.0967 lr 6.00e-05 54,440 tok/s 56.600B",
    # windowed rate + cumulative average + peak VRAM (cuda)
    "windowed_cuda": ("step 10250/287882 loss 1.0967 lr 6.00e-05 "
                      "31,400 tok/s (avg 30,120) peak 21.4GB 56.600B"),
    # same run on cpu: no peak figure at all
    "windowed_cpu": ("step 10250/287882 loss 1.0967 lr 6.00e-05 "
                     "31,400 tok/s (avg 30,120) 56.600B"),
    # a DIVERGED run: loss_acc is printed with :.4f, which renders nan/inf.
    # Refusing these froze the dashboard on the last finite step, where it kept
    # showing a live-looking percentage and ETA forever -- worse than saying
    # nothing, because the run had actually stopped learning.
    "diverged_nan": ("step 10250/287882 loss nan lr 6.00e-05 "
                     "31,400 tok/s (avg 30,120) peak 21.4GB 56.600B"),
    "diverged_inf": ("step 10250/287882 loss inf lr 6.00e-05 "
                     "31,400 tok/s (avg 30,120) peak 21.4GB 56.600B"),
}


def test_every_step_line_format_the_trainer_emits_still_parses():
    """All three must yield the SAME fields -- a reader that only handles the
    format in front of it today is how this broke."""
    for name, line in STEP_LINES.items():
        m = training_progress.STEP_RE.match(line)
        assert m, f"{name} step line did not parse: {line!r}"
        assert m.group(1) == "10250", name
        assert m.group(2) == "287882", name
        assert m.group(6) == "56.600", f"{name} lost the cumulative token total"
        # the loss field must survive float() whatever it holds
        assert float(m.group(3)) == float(m.group(3)) or "diverged" in name

    # the RATE captured is the windowed one, not the cumulative average --
    # the average is what hid a mid-run slowdown in the first place
    assert training_progress.STEP_RE.match(STEP_LINES["windowed_cuda"]).group(5) == "31,400"


def test_non_step_lines_are_not_swallowed():
    """The pattern skips the middle of the line, so it has to stay anchored:
    a greedy version matched val lines and banners too."""
    for line in ("[val] step 250 loss 3.1 ppl 22.2",
                 "  [val-gen] step 250 loss 3.4 ppl 30.0",
                 "training: 196,608 tok/step (mb 6 x ga 16 x 2048) | ckpt=False",
                 "val sources (1): SE/worldbuilding 100%",
                 "WARNING: throughput 12,000 tok/s is below 70% of this run's best"):
        assert not training_progress.STEP_RE.match(line), f"wrongly matched: {line!r}"


def test_a_log_of_mixed_formats_reads_to_the_last_step(tmp_path):
    """A resumed run appends new-format lines to a log full of old ones."""
    log = tmp_path / "train_mixed.log"
    log.write_text(
        "training: 196,608 tok/step (mb 6 x ga 16 x 2048)\n"
        + STEP_LINES["legacy"] + "\n"
        + "[val] step 10250 loss 1.10 ppl 3.0\n"
        + STEP_LINES["windowed_cuda"].replace("10250/", "10260/") + "\n",
        encoding="utf-8",
    )
    last, vals, tok_per_step = training_progress.parse(str(log))
    assert last is not None, "render() would print 'No step lines found'"
    assert last[0] == 10260, "the newest step must win, whichever format it is in"
    assert last[4] == 31400
    assert vals == [(10250, 3.0)]
    assert tok_per_step == 196608
