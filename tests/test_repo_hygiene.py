"""Repository hygiene regression gates.

Pass 156z9cv (May 11, 2026): Mojibake regression gate.

History: SUGGESTIONS stamps 156z9cr/cs/ct/cu all logged "mojibake at
inference.py L1167-1168" as a small bounded site. The May 11 audit
re-grepped and found 2341 mojibake markers in inference.py and 7 in
rl_training.py — three orders of magnitude beyond what was logged.
Root cause was a cp1252-mis-decoded-as-UTF-8 round-trip on the
inference.py module docstring ASCII art (and a smaller version on
rl_training.py).  Both files were re-fixed via ``ftfy.fix_text`` plus
one surgical replace for a triple-encoded ``⚡`` sequence.

This test prevents a future re-introduction by flagging the
canonical mojibake triad characters (``â``, ``Â``, ``Ã``) appearing
anywhere in package source.  All three are valid in legitimate
text (Portuguese, French, etc.) — but in a primarily English code
base they are overwhelmingly a tell of double-encoded UTF-8.  The
gate is intentionally narrow:

  * Allow legitimate Unicode (box drawing, arrows, bullets, math).
  * Reject the three "mojibake leading bytes" that almost never
    appear in honest English source.

If a future change legitimately needs one of these characters in
a string literal or comment (e.g. a non-English test fixture or a
deliberate test of mojibake detection), allowlist that file path
inside ``ALLOWED_FILES`` with a clear comment justifying it.
"""

from __future__ import annotations

import ast
import pathlib

# Files explicitly allowed to contain mojibake triad characters.
# Add a comment when listing a file here so future readers understand
# why the exception exists.
ALLOWED_FILES: set[str] = {
    # this very file defines the triad literals it hunts
    "tests/test_repo_hygiene.py",
}

# The three canonical mojibake "leading bytes" when cp1252 text is
# mis-decoded as UTF-8 and then re-encoded as UTF-8.  In honest
# English source these almost never appear; when they do, it is
# nearly always double-encoded text.
MOJIBAKE_TRIAD: tuple[str, ...] = ("â", "Â", "Ã")

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "enigma_engine"
_SKIP_PARTS = {"__pycache__", "venv", ".venv", "_archive", ".git", "node_modules"}


def _first_party_py():
    """Root scripts + package + tests. The old gate scanned only
    enigma_engine/, leaving the most console-bound files (serve_enigma,
    eval_behavior, teach_enigma...) outside every hygiene rule -- which is
    why manual ASCII sweeps kept 'finishing' and then recurring (test-suite
    audit 2026-07-17)."""
    yield from sorted(REPO_ROOT.glob("*.py"))
    for base in (PACKAGE_ROOT, REPO_ROOT / "tests"):
        for p in sorted(base.rglob("*.py")):
            if not any(part in _SKIP_PARTS for part in p.parts):
                yield p


def test_first_party_source_is_free_of_mojibake_markers() -> None:
    """Every first-party ``.py`` file must be free of the three canonical
    mojibake leading bytes.

    Pass 156z9cv: regression gate for the inference.py + rl_training.py
    mojibake corruption that was under-reported in prior stamps.
    2026-07-17: scope widened from enigma_engine/ to the whole first-party
    tree (root scripts included).
    """
    assert PACKAGE_ROOT.is_dir(), f"package root missing: {PACKAGE_ROOT}"

    failures: list[tuple[str, dict[str, int]]] = []
    for path in _first_party_py():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in ALLOWED_FILES:
            continue
        text = path.read_text(encoding="utf-8")
        counts = {ch: text.count(ch) for ch in MOJIBAKE_TRIAD if ch in text}
        if counts:
            failures.append((rel, counts))

    if failures:
        lines = ["mojibake markers found in first-party source:"]
        for rel, counts in failures:
            summary = ", ".join(f"{ch!r}={n}" for ch, n in counts.items())
            lines.append(f"  {rel}: {summary}")
        lines.append(
            "If a character is legitimate, add the file path to "
            "ALLOWED_FILES in tests/test_repo_hygiene.py with a "
            "justification comment."
        )
        raise AssertionError("\n".join(lines))


# ---------------------------------------------------------------------------
# Console-ASCII gate (CLAUDE.md: "Console output must be ASCII" -- cp1252
# consoles crash printing unicode). Until 2026-07-17 NO test enforced this
# rule anywhere; every "sweep now zero" was a manual grep that later turned
# out to have scanned too narrowly (three separate times). This gate defines
# the scope once: string literals inside the direct console sinks.
#
# Known limitation (accepted): strings that reach the console indirectly
# (appended to a warnings list that a caller prints later) are out of static
# reach. The rule for those stays manual; the sinks below are where every
# past regression actually lived.
# ---------------------------------------------------------------------------

_LOGGER_METHODS = {"debug", "info", "warning", "warn", "error", "critical", "exception"}

# Escape hatch for a future false positive (e.g. a non-console `.exit()` or
# `._emit_progress()` on some other object): "path/to/file.py" silences the
# whole file, "path/to/file.py:123" one line. Justify every entry with a
# comment. Empty = the gate currently has zero false positives.
CONSOLE_ALLOWED: set[str] = set()


def _nonascii_strings(node: ast.AST):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            bad = sorted({f"U+{ord(c):04X}" for c in sub.value if ord(c) > 127})
            if bad:
                yield sub.lineno, ",".join(bad)


def test_console_bound_strings_are_ascii() -> None:
    violations: list[str] = []
    for path in _first_party_py():
        rel = path.relative_to(REPO_ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:  # a file that cannot parse is its own bug
            violations.append(f"  {rel}: SyntaxError {exc}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            # _emit_progress: the Trainer docstring documents wiring
            # on_progress straight to print -- a de-facto console sink.
            # sys.exit(msg) prints msg to stderr like SystemExit.
            if isinstance(f, ast.Name) and f.id in {"print", "SystemExit"}:
                targets = [node]
            elif isinstance(f, ast.Attribute) and (
                f.attr in _LOGGER_METHODS or f.attr in {"_emit_progress", "exit"}
            ):
                targets = [node]
            elif isinstance(f, ast.Attribute) and f.attr == "add_argument":
                targets = [kw.value for kw in node.keywords if kw.arg == "help"]
            elif (isinstance(f, ast.Name) and f.id == "ArgumentParser") or (
                # every call site in this repo uses the Attribute form
                # argparse.ArgumentParser(...) -- the bare-Name-only match
                # made description/epilog scanning dead code (re-audit
                # 2026-07-18)
                isinstance(f, ast.Attribute) and f.attr == "ArgumentParser"
            ):
                targets = []
                for kw in node.keywords:
                    if kw.arg not in ("description", "epilog"):
                        continue
                    if isinstance(kw.value, ast.Name) and kw.value.id == "__doc__":
                        # description=__doc__: the module docstring IS the
                        # --help text -- scan it (round-2 re-audit; four
                        # scripts use this form)
                        doc = ast.get_docstring(tree, clean=False)
                        if doc:
                            targets.append(ast.Constant(value=doc, lineno=node.lineno, col_offset=0))
                    else:
                        targets.append(kw.value)
            else:
                continue
            for target in targets:
                for lineno, chars in _nonascii_strings(target):
                    if rel in CONSOLE_ALLOWED or f"{rel}:{lineno}" in CONSOLE_ALLOWED:
                        continue
                    violations.append(f"  {rel}:{lineno}: {chars}")
    assert not violations, (
        "non-ASCII in console-bound strings (crashes cp1252 consoles; "
        "CLAUDE.md ASCII rule):\n" + "\n".join(sorted(set(violations)))
    )


# ---------------------------------------------------------------------------
# Training-content dash convention -- the console gate's inverted twin. The
# hand-authored sources below feed the corpora she is trained on, and a prose
# dash there is content, not console output: it must be the spaced em-dash the
# identity anchors have always used. Both rules meet inside these same files,
# so scope is decided per literal, never per file -- console-bound literals
# keep ASCII, docstrings and comments are out of scope for both.
# ---------------------------------------------------------------------------

# The hand-authored training-content sources. knowledge_corpus.py is in scope
# too: its tables also render the continued-pretrain facts stream, and
# test_knowledge_data.test_pretrain_text_lines_are_clean now whitelists exactly
# U+2014 on top of ASCII, so the two rules agree instead of contradicting.
# Its two separator constants (_SENTENCE_SEPS/_FRAGMENT_SEPS) split answers on
# the prose dash, so they carry the em-dash with the content they parse.
TRAINING_CONTENT_FILES: tuple[str, ...] = (
    # The REFERENCE spelling, and unguarded until 2026-08-20: the file whose
    # 154 em-dashes decided the convention was the one authored source this
    # gate did not read, so nothing stood between it and a drift back.
    "identity_anchors.py",
    "identity_paraphrases.py",
    "enigma_engine/core/persona_content.py",
    "make_sft_data.py",
    "make_dpo_data.py",
    "knowledge_corpus.py",
)

# Helpers that raise or print on their caller's behalf. Their message literals
# reach the console, and the sink match above cannot see through the call.
_INDIRECT_CONSOLE_SINKS = {"_refuse"}

ASCII_DASH = " -- "
EM_DASH = " — "


def _console_or_docstring_lines(tree: ast.AST) -> set[int]:
    """Lines whose string literals are a docstring or console-bound.

    Line ranges rather than nodes: a console sink's literals can sit anywhere
    inside its call, and marking the whole call is what makes "everything
    else is content" a decidable question."""
    lines: set[int] = set()

    def mark(node) -> None:
        lines.update(range(node.lineno, node.end_lineno + 1))

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                mark(body[0].value)
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise):
            mark(node)
            continue
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Name) and (
            f.id in {"print", "SystemExit", "ArgumentParser"} or f.id in _INDIRECT_CONSOLE_SINKS
        ):
            mark(node)
        elif isinstance(f, ast.Attribute) and (
            f.attr in _LOGGER_METHODS
            or f.attr in {"_emit_progress", "exit", "add_argument", "ArgumentParser"}
            or f.attr in _INDIRECT_CONSOLE_SINKS
        ):
            mark(node)
    return lines


def test_training_content_spells_its_prose_dash_as_an_em_dash() -> None:
    """No training-table literal may carry the spaced ASCII dash.

    identity_anchors.py spelled all 154 of its prose dashes as the em-dash
    while the other authored sources spelled the same dash " -- ", so one
    corpus carried two surfaces for one punctuation mark. Kokoro's G2P is the
    tiebreak: it reads the em-dash as prosodic punctuation and DELETES " -- "
    from the phoneme stream, so the ASCII form went unspoken. Unified
    2026-08-19.

    A console-bound literal in these same files is the console gate's, not
    this one's, and keeps ASCII."""
    violations: list[str] = []
    for rel in TRAINING_CONTENT_FILES:
        path = REPO_ROOT / rel
        assert path.exists(), f"{rel} is a documented training-content source"
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:  # a file that cannot parse is its own bug
            violations.append(f"  {rel}: SyntaxError {exc}")
            continue
        skip = _console_or_docstring_lines(tree)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and ASCII_DASH in node.value and node.lineno not in skip):
                violations.append(f"  {rel}:{node.lineno}: {node.value.strip()[:70]!r}")
    assert not violations, (
        "training-content literals still carry the spaced ASCII dash; the "
        f"convention is {EM_DASH!r} (console-bound literals keep ASCII):\n"
        + "\n".join(sorted(set(violations)))
    )


def test_detached_launchers_survive_cmd_quote_stripping():
    """cmd.exe strips the outermost quote pair from the line after /c when
    that line holds several quoted tokens. Both the venv interpreter path and
    this repo's own path contain spaces, so a command built as

        /c "<py>" -u script.py --resume "<ckpt>" >> "<log>" 2>&1

    reaches cmd mangled: it launches nothing, writes no log, and returns no
    error. The whole command therefore carries ONE extra enclosing pair.

    Pinned because the failure is silent in the worst place -- a resume that
    launches nothing looks exactly like a resume that is training, and the
    script that does this runs hidden under Task Scheduler. Covers EVERY
    detached launcher building an $inner line, not just the one that broke:
    extend_length.ps1's line survived on a technicality (exactly two quote
    marks lands in cmd's preserve-quotes rule) and was one quoted argument
    away from the identical silent failure. That launcher was deleted
    2026-08-26; the loop stays a loop for whatever detaches next."""
    checked = 0
    for name in ("resume_training.ps1",):
        launcher = REPO_ROOT / name
        assert launcher.exists(), f"{name} is a documented detached launcher"
        text = launcher.read_text(encoding="utf-8")
        inner = [ln for ln in text.splitlines()
                 if ln.strip().startswith("$inner") and "=" in ln]
        assert inner, f"{name} no longer builds an $inner command line"
        for line in inner:
            body = line.split("=", 1)[1].strip()
            assert body.startswith('"/c `"`"'), (
                f"{name}: the cmd command line must open with /c and TWO "
                "quote marks, or cmd strips the pair that protects the "
                "interpreter path:\n  " + line
            )
            assert body.rstrip().endswith('`""'), (
                f"{name}: the extra enclosing quote must also CLOSE, or cmd "
                "sees an unterminated quoted line:\n  " + line
            )
            checked += 1
    assert checked >= 1, "expected at least one $inner line per launcher"


def test_packed_file_writers_join_records_with_the_separator():
    """The record separator (U+001E) is a WRITER-side contract: the reader
    splits on it, so a single write site reverting to "\n\n".join() silently
    returns that source to one-document-per-5MB -- every <s>/</s> between its
    records vanishes at the next retokenize and the full suite stays green.
    The curated shard writer is in scope because it shipped exactly that way
    (21,218 documents in 11 files, found 2026-07-30) while every eye was on
    the HF fetchers."""
    writers = {
        "collect_pretraining_data.py": 12,   # 4 fineweb + 4 generic HF + 4 stack
        "make_pretrain_curated.py": 1,
    }
    for name, expected in writers.items():
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        legacy = text.count('"\n\n".join(batch_text)') + text.count('"\n\n".join(chunk)')
        assert legacy == 0, (
            f"{name}: {legacy} packed-file write site(s) join with a blank "
            "line -- indistinguishable from a paragraph break, so the reader "
            "cannot recover the record boundaries"
        )
        sep_joins = text.count("DOC_SEP_JOIN.join(")
        assert sep_joins == expected, (
            f"{name}: expected {expected} DOC_SEP_JOIN write site(s), found "
            f"{sep_joins} -- a write path was added or removed without the "
            "separator contract following it"
        )


def test_start_process_launchers_avoid_the_four_detach_traps():
    """The Start-Process family of detached launchers, pinned against the four
    traps that each cost a real launch (2026-07-30, corpus re-collect and
    retokenize):

    1. -ArgumentList given an ARRAY joins with spaces and does NOT quote the
       elements, so a repo path with a space reaches python split in half and
       it dies instantly with "can't open file". The argument list must be ONE
       string whose script path is explicitly quoted.
    2. Python block-buffers stdout when redirected to a file, so the log sits
       at 0 bytes and a stall is indistinguishable from a healthy quiet buffer
       for over an hour. PYTHONUNBUFFERED must be set.
    3. $proc.Handle must be touched WHILE THE CHILD IS ALIVE or ExitCode reads
       empty under PowerShell 5.1 and the done-marker cannot tell a finished
       run from a killed one.
    4. Wait-Process -Id leaves ExitCode unpopulated; the -PassThru object's
       WaitForExit() is what actually populates it.

    A green LastTaskResult describes the LAUNCHER, never the child -- which is
    why these are pinned in a test rather than trusted to a task result.

    Start-Enigma.ps1 is checked against trap 1 ONLY, at the end: it detaches a
    server it never waits for, so traps 3 and 4 (reading a child's ExitCode)
    have nothing to bite, and serve flushes every print itself rather than
    relying on PYTHONUNBUFFERED. Trap 1 was live in it -- it passed an ARRAY,
    and worked only because no serve argument had ever carried a space, which
    -Persona <pack-dir> ends the first time a pack lives in "My Packs".

    Every run_*.ps1 in the repo is here, T4 and T5 included: their runs are
    COMPLETE, but the recipe is what the next long job gets copied from, and a
    finished launcher carrying trap language is the one that gets reused. They
    run two steps each, so the argument string is named per step and handed to
    a shared Invoke-Step -- the trap is the same, the variable is spelled
    differently. (Their first-launch-only refusals are a separate guard: this
    test is about the DETACH recipe, not about re-runnability.)"""
    launchers = {
        "run_collect_rebuild.ps1": ("$argString",),
        "run_pretokenize_v2c.ps1": ("$argString",),
        "run_pretrain_t3.ps1": ("$argString",),
        "run_t4_facts_sft.ps1": ("$factsArgs", "$sftArgs"),
        "run_t5_sft2_dpo.ps1": ("$sftArgs", "$dpoArgs"),
        # One step, so one argument variable -- the detach recipe is the same
        # either way, which is the whole point of pinning it per launcher.
        "run_sft4.ps1": ("$sftArgs",),
        "run_sft5.ps1": ("$sftArgs",),
    }
    for name, arg_vars in launchers.items():
        launcher = REPO_ROOT / name
        assert launcher.exists(), f"{name} is a documented detached launcher"
        text = launcher.read_text(encoding="utf-8")
        # Code lines only: these launchers name the trap in PROSE on purpose,
        # so the next reader does not re-introduce what they were fixed for.
        code_lines = [ln for ln in text.splitlines() if not ln.strip().startswith("#")]

        # -ArgumentList must receive ONE variable holding ONE string. An array
        # literal or a comma list there is trap 1 wearing its original clothes.
        passed = [ln.split("-ArgumentList", 1)[1].split()[0]
                  for ln in code_lines if "-ArgumentList" in ln]
        assert passed, f"{name} no longer hands an argument string to Start-Process"
        for token in passed:
            assert token.startswith("$"), (
                f"{name}: pass ONE pre-quoted string to -ArgumentList; an array "
                f"is joined without quoting and splits any path containing a "
                f"space:\n  -ArgumentList {token}"
            )
        for var in arg_vars:
            arg_lines = [ln for ln in code_lines if ln.strip().startswith(var + " ")]
            assert arg_lines, f"{name} no longer builds {var}"
            assert arg_lines[0].split("=", 1)[1].strip().startswith('"`"$'), (
                f"{name}: the script path must be quoted INSIDE the argument "
                "string:\n  " + arg_lines[0]
            )
        assert 'PYTHONUNBUFFERED' in text, (
            f"{name}: set PYTHONUNBUFFERED or the log stays empty while the job runs"
        )
        assert "$proc.Handle" in text, (
            f"{name}: touch $proc.Handle while the child lives or ExitCode is lost"
        )
        assert "$proc.WaitForExit()" in text, (
            f"{name}: use $proc.WaitForExit(), not Wait-Process, to populate ExitCode"
        )
        code = "\n".join(code_lines)
        assert "Wait-Process" not in code, (
            f"{name}: Wait-Process leaves ExitCode unpopulated"
        )

    # Trap 1 in the USER-FACING launcher: the same Start-Process call, the same
    # -ArgumentList, and every path on its line now comes from a persona pack.
    start = (REPO_ROOT / "Start-Enigma.ps1").read_text(encoding="utf-8")
    assert "-ArgumentList $argString" in start, (
        "Start-Enigma.ps1: pass ONE pre-quoted string to -ArgumentList; an "
        "array is joined without quoting and splits any path containing a space"
    )
    start_args = [ln for ln in start.splitlines() if ln.strip().startswith("$argString")]
    assert start_args, "Start-Enigma.ps1 no longer builds an $argString"
    assert start_args[0].split("=", 1)[1].strip().startswith('"`"serve_enigma.py`"'), (
        "Start-Enigma.ps1: the script path must be quoted INSIDE the argument "
        "string (it stays repo-relative -- -WorkingDirectory is the engine "
        "dir):\n  " + start_args[0]
    )
    for flag in ("--memory-dir", "--persona", "--voice-name"):
        assert f'{flag} `"' in start, (
            f"Start-Enigma.ps1: {flag} takes a path or a name from a pack -- "
            "quote its value or the first space splits it into two arguments"
        )


# The lineage every entry point picks when the user names none. Moving it is a
# deliberate act (BACKLOG adoption receipt), so it is written once, here.
ADOPTED_LINEAGE = "enigma_v2_sft2"


def test_bare_run_entry_points_default_to_the_adopted_lineage():
    """A stale default fails SILENTLY when the old checkpoint is still on
    disk. `models/enigma_pretrain_large/model.pth` is kept as the v1 rollback,
    so a bare `python sample_enigma.py` loaded it, printed an ordinary
    "loaded ..." line, and sampled the lineage nobody serves -- an adoption
    wave updated the launchers and missed this one. Every entry point that
    picks a checkpoint FOR the user picks the same one, and when the adopted
    lineage moves this is the list that notices."""
    picks = {"sample_enigma.py": "--ckpt",
             "bench_generate.py": "--model",
             "align_vision.py": "--model"}
    for name, flag in picks.items():
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        defaults = [ln for ln in text.splitlines()
                    if f'"{flag}"' in ln and "default=" in ln]
        assert len(defaults) == 1, (
            f"{name}: expected exactly one {flag} default, found {defaults}"
        )
        assert ADOPTED_LINEAGE in defaults[0], (
            f"{name}: {flag} defaults to a lineage that is not the adopted "
            f"{ADOPTED_LINEAGE}:\n  {defaults[0].strip()}"
        )


def test_her_window_turns_the_autoplay_gate_off():
    """WebView2 enforces Chromium's autoplay policy, which mutes a WAV reply
    that finishes after the last click -- the audio plays into silence and
    nothing anywhere errors. pywebview hardcodes AdditionalBrowserArguments,
    so this env var is the only channel there is, and it lives as one literal
    in one file: drop it and she goes quiet with the suite still green."""
    text = (REPO_ROOT / "enigma_window.py").read_text(encoding="utf-8")
    assert "--autoplay-policy=no-user-gesture-required" in text, (
        "enigma_window.py must set WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS "
        "before pywebview loads, or WebView2 mutes her replies"
    )


def test_collect_extra_carries_the_fetch_deps():
    """requirements.txt owns requests+zstandard with in-file comments naming the
    collector; a pip install .[collect] must not die one dependency over."""
    import re
    import tomllib
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    names = {re.split(r"[<>=\[; ]", d, 1)[0] for d in data["project"]["optional-dependencies"]["collect"]}
    assert {"requests", "zstandard"} <= names
