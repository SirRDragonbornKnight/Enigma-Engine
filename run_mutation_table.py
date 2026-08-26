"""Re-runnable mutation table: each entry re-applies a known bug (or disables a
guard) and proves the recorded killing test still fails. Refuses a dirty tree;
restores BYTE-IDENTICALLY. Run: venv\\Scripts\\python.exe run_mutation_table.py

Two hard-won rules baked in (plan audit 2026-08-25):
- BYTES I/O only. read_text/write_text CRLF-flips LF worktree files on Windows
  (measured: dpo_enigma.py 0->508 CRLF on one round trip) and git's autocrlf
  normalization HIDES the flip from `git status` -- the porcelain check cannot
  catch what it needs to catch, so the restore itself must be byte-exact.
- KILLED means the test RAN and FAILED: pytest exit 1 with 'failed' in the
  tail. Any other nonzero (2/3/4/5 = interrupted/internal/usage/no-tests) is
  TEST-ERROR, never a kill -- a stale node id or wrong interpreter must not
  mint a fake receipt.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (file, original_snippet_bytes, mutant_snippet_bytes, killing_test_nodeid)
ENTRIES = [
    ("pretrain_enigma.py",
     b"if not (corpus_vocab <= ck_vocab <= corpus_vocab + _VOCAB_PAD_SLACK):",
     b"if False and not (corpus_vocab <= ck_vocab <= corpus_vocab + _VOCAB_PAD_SLACK):",
     "tests/test_v2_trainer.py::test_vocab_mismatch_refused_both_directions"),
    ("dpo_enigma.py",
     b"if block > max_seq_len:",
     b"if False and block > max_seq_len:",
     "tests/test_dpo_trainer.py::test_over_block_refused"),
    ("collect_vision_data.py",
     b'tmp = path.with_suffix(path.suffix + ".tmp")',
     b'tmp = path  # MUTANT: non-atomic',
     "tests/test_collect_vision_data.py::test_write_jsonl_leaves_the_original_on_midwrite_failure"),
    ("serve_enigma.py",
     b"if host.lower() not in _LOOPBACK_HOSTS and not unsafe_lan:",
     b"if False and host.lower() not in _LOOPBACK_HOSTS and not unsafe_lan:",
     "tests/test_serve_enigma.py::test_non_loopback_bind_refused_without_flag"),
    ("serve_enigma.py",
     b"if total > cap:",
     b"if total >= cap * 10**6:",
     "tests/test_serve_enigma.py::test_capped_reader_refuses_oversize"),
    # The next two killing tests are SOURCE-GREP presence pins (inspect.getsource
    # then `needle in src`), so the usual `if False and ...` mutant leaves the
    # needle sitting in the source and the pin stays GREEN. Their mutants must
    # REMOVE the call text outright; the leading indent is part of the snippet.
    ("pretrain_enigma.py",
     b"        _refuse_out_overwrite(args.resume, args.sanity, args.eval_only, out)",
     b"        pass",
     "tests/test_v2_trainer.py::test_pretrain_out_guard_sits_on_the_resolution_path"),
    ("align_vision.py",
     b'        refuse_existing_patterns(Path(args.out), ("*.pt",))',
     b"        pass",
     "tests/test_encoder_out_guards.py::test_all_four_writers_carry_a_guard"),
    ("eval_behavior.py",
     b"    if overall_n == 0:",
     b"    if False and overall_n == 0:",
     "tests/test_eval_grading.py::test_comparator_refuses_a_zero_gated_run"),
]


def main() -> int:
    # -uno (tracked files only): the runner mutates only TRACKED files, so an
    # untracked file -- including this runner itself before its first commit --
    # can never be damage this check needs to distinguish. Without -uno the
    # script refuses to run until it is committed, a self-deadlock. Tracked-file
    # dirtiness still refuses.
    dirty = subprocess.run(["git", "status", "--porcelain", "-uno"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        print("REFUSED: tree is dirty -- run this only on a committed tree "
              "(damage must be distinguishable from work).")
        return 2
    failures = []
    for rel, orig, mutant, test in ENTRIES:
        p = ROOT / rel
        src = p.read_bytes()
        if src.count(orig) != 1:
            failures.append(f"{rel}: snippet count {src.count(orig)} != 1 -- table stale")
            continue
        p.write_bytes(src.replace(orig, mutant))
        try:
            r = subprocess.run([sys.executable, "-m", "pytest", test, "-q", "-x"],
                               cwd=ROOT, capture_output=True, text=True)
        finally:
            p.write_bytes(src)
        ran_and_failed = r.returncode == 1 and "failed" in r.stdout
        if ran_and_failed:
            verdict = "KILLED"
        elif r.returncode == 0:
            verdict = "SURVIVED"
        else:
            verdict = f"TEST-ERROR exit {r.returncode}"
        print(f"{verdict}: {rel} :: {test}")
        if verdict != "KILLED":
            failures.append(f"{verdict}: {rel} :: {test}")
        if p.read_bytes() != src:
            failures.append(f"{rel}: RESTORE NOT BYTE-IDENTICAL -- investigate before anything else")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"\nall {len(ENTRIES)} mutants killed, restores byte-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
