# GUI Baseline Measurements — Phase 0b

**Status:** Measurement protocol shipped. Numbers pending operator run. Author: AA code maker. Date: May 13, 2026.
**Pass anchor:** GUI-ARCH-0b.

This document is the **measurement contract** for the current CustomTkinter GUI. Every metric in [ARCH_DECISION.md](ARCH_DECISION.md) §5 (rubric) and §4 (gate G3) requires a baseline number on the incumbent stack so Phase 1 POCs are scored against reality, not against arbitrary absolutes.

> **Why this matters.** Per §4 Learned Principles (`AA code maker.md`): *"Beat baseline" targets require measuring the baseline first. A gate or metric written as "≤ baseline" is vacuous if the baseline is never captured.* This file fixes that gap before Phase 1 starts.

---

## 1. Environment

Fill in once. All Phase 1 POC runs must use the **same machine** — running PySide6 on a 4090 desktop and Tauri on a laptop is not a fair comparison.

| Field | Value |
|---|---|
| Machine | _e.g. RTX 5090 desktop, 32 GB RAM_ |
| OS | Windows ___ (build ___) |
| Python | ___ (`python --version`) |
| Installed extras | `pip list \| findstr /i "torch customtkinter pyside6"` |
| GUI commit | `git rev-parse HEAD` |
| Date measured | YYYY-MM-DD |

## 2. Measurement protocol

**Automation:** M1, M2 and M5 are now spliced into the live GUI behind an opt-in flag (Pass 156z9ed, GUI-ARCH-0b). M3 is automated by [measure_baseline.py](measure_baseline.py). M4 is a static disk-size estimate already filled in §3 below.

**Operator workflow (one-pass, fills M1/M2a/M2b/M5):**

```powershell
# Three cold launches. Kill the GUI between each via the window's X button.
# Each run prints [BASELINE] M1_..., [BASELINE] M2_..., [BASELINE] M5_... lines.
python run.py --gui --baseline
python run.py --gui --baseline
python run.py --gui --baseline
```

For each run:
1. **M1** — Read the `[BASELINE] M1_cold_start_s=<value>` line printed once the window is idle.
2. **M2a / M2b** — Click sidebar CONFIG, then FORGE. Each click prints one `[BASELINE] M2_switch from=<from> to=<to> ms=<value>` line.
3. **M5** — Start a smoke-data training run from FORGE (epochs=1, ~30 s). Watch the `[BASELINE] M5_max_stall_ms_so_far=<value>` checkpoint lines (one every ~5 s). Record the final value after the run.

Take the **median** of 3 cold launches per metric and write it into §3.

For **M3** (idle RAM, GUI must be running):
```powershell
# In a separate terminal, with the GUI idle on CORE / HOME:
python information/gui/measure_baseline.py --m3 --pid <PID> --settle 60
```
Get `<PID>` from Task Manager or `Get-Process python`.

Run each measurement **3 times**, record min/median/max, use **median** in the table. Close all non-essential apps first. Disable Windows Game Bar / Discord overlay / any FPS overlay before measuring.

### M1 — Cold start (shell ready to accept input)

**What:** Time from `python run.py --gui --baseline` invocation to the moment the main window is rendered AND tk's event loop reaches its first idle cycle (i.e. ready to receive input). Captured as `[BASELINE] M1_cold_start_s=<seconds>`.

**Mechanism:** `_PROCESS_START = time.perf_counter()` is captured at the very top of `run.py` (right after imports). When `--baseline` is set, that value is forwarded to `BaselineMonitor`. `EnigmaGUI.__init__` schedules `self.after(0, monitor.emit_m1)`, which fires once the tk mainloop is idle — i.e. after the full cold-start window has elapsed.

**Record:** median of 3 cold starts (kill the process between runs; do not measure warm starts).

### M2 — Page-switch latency

**What:** Time from sidebar button press on CONFIG → page fully rendered. Repeat for FORGE.

**Mechanism:** `_switch_page` captures `time.perf_counter()` before any grid reflow and emits `[BASELINE] M2_switch from=<old> to=<new> ms=<value>` after the new page is gridded and the SEND-button safety check runs. The first auto-switch to CORE during `__init__` prints `from=<initial>`; ignore that line and use the operator-triggered transitions.

**Record:** median of 3 switches each. Switch from CORE (the boot landing page) to CONFIG, then to FORGE.

### M3 — Idle RAM, shell process only

**What:** Resident set size (RSS) of the GUI Python process after the window has been idle for 60 seconds, **before any model is loaded** (no checkpoint selected in MODELS page).

**How:** PowerShell helper:
```powershell
python information/gui/measure_baseline.py --m3 --pid <PID> --settle 60
```
**Record:** median of 3 cold launches.

**Why "before model loaded":** Per rubric (§5 row 3), torch + model weights are excluded from the shell-only number. Loading a 742M model adds ~3 GB and dominates the measurement, which would make every framework look identical.

### M4 — Packaged install size

**Filled by agent (May 13, 2026, GUI-ARCH-0b).** The current GUI ships as a Python source tree, not a packaged installer. The number below is the **UI-surface estimate** (GUI source + CustomTkinter + Pillow) excluding the Python interpreter (~30 MB) and torch (~2 GB) per rubric §5 row 3.

| Component | Size (MB) |
|---|---|
| `enigma_engine/gui/` source | 2.4 |
| CustomTkinter | 1.4 |
| Pillow | 15.1 |
| **Sum** | **19.0** |

Reproduce with:
```powershell
python information/gui/measure_baseline.py --m4
```

### M5 — Frame stall during 30-second training step (Gate G3)

**What:** While a synthetic 30-second training run is executing on a worker thread, log the longest interval between `after(16, ...)` callbacks. A 16 ms interval = 60 FPS. Anything above ~50 ms is visible jank. The number we record is the **median of the max-stall-per-run** over 3 runs.

**Mechanism:** When `--baseline` is set, `EnigmaGUI` schedules `self.after(16, self._baseline_frame_tick)` once the mainloop starts. Each tick calls `BaselineMonitor.frame_tick()` which tracks the rolling max. Every ~5 s the GUI prints `[BASELINE] M5_max_stall_ms_so_far=<ms>` so the operator does not need a debugger. Record the final printed value after the 30 s training step completes.

**Record:** median of 3 max-stall values. This is the number Phase 1 POCs must not exceed on the same workload (primary G3) or improve to < 100 ms (stretch G3).

### M6 — Dev-velocity proxy (for rubric row 5)

**Soft signal only.** Not measured here. Phase 1 records dev-time-to-parity per track during the POC.

## 3. Results table (fill in after measurement)

| Metric | Cold (1) | Cold (2) | Cold (3) | Median |
|---|---|---|---|---|
| M1 — Cold start (s) | | | | |
| M2a — Page-switch HOME → CONFIG (ms) | | | | |
| M2b — Page-switch HOME → FORGE (ms) | | | | |
| M3 — Idle RAM (MB) | | | | |
| M4 — Estimated bundle size (MB) | 19.0 (one-shot) | — | — | **19.0** |
| M5 — Max frame stall during 30 s training (ms) | | | | |

## 4. Notes captured during measurement

Operator: free-form notes about anything unusual — disk I/O during cold start, GPU driver pop-ups, debounced widgets behaving oddly, anything that would invalidate a number. Document everything; future Phase 1 / Phase 2 will refer back here.

```
(empty until first run)
```

## 5. Acceptance for Phase 0b close

- [x] M4 estimate captured (agent, May 13, 2026 — 19.0 MB).
- [x] Helper script [measure_baseline.py](measure_baseline.py) automates M3 + M4.
- [x] M1/M2/M5 splice shipped behind `--baseline` opt-in (agent, Pass 156z9ed — `python run.py --gui --baseline`).
- [ ] §1 Environment filled in (operator).
- [ ] §3 Results table has medians for M1, M2a, M2b, M3, M5 (operator).
- [ ] §4 Notes captured (or explicit "no anomalies observed") (operator).
- [ ] Numbers cross-referenced in [ARCH_DECISION.md](ARCH_DECISION.md) §9 sign-off checklist as "filled in" (operator).

Once §5 is complete, Phase 0b is closed and the Phase 1 rubric has real targets to score against.
