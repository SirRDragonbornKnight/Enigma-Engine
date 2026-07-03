# GUI Architecture Decision Doc — Phase 0a

**Status:** Draft (Phase 0 deliverable). Author: AA code maker. Date: May 13, 2026.
**Pass anchor:** GUI-ARCH-0a (see [SUGGESTIONS.md](../../SUGGESTIONS.md) Next-Actions row 7, GUI Modernization Planning section).

This document captures the **constraints, non-goals, current pain, and decision gates** that govern the GUI modernization effort. It is the contract every later phase (POCs, decision, migration, cutover) refers back to. No framework is picked here. No code is changed yet.

---

## 1. Non-negotiable constraints

These come from the project goal in [AA code maker.md](../../AA%20code%20maker.md) §Project Goal → Constraints. Any candidate stack that cannot meet all four is eliminated, regardless of other strengths.

| # | Constraint | Why |
|---|---|---|
| C1 | **Fully local / offline-capable** | All training and inference run on the user's PC. No cloud dependency, no external data leakage. |
| C2 | **Black-box deployment posture** | The model is a single artifact; users do not edit weights by hand. No telemetry, no remote auto-update, no analytics SDKs. |
| C3 | **Personality from training, not the user** | The AI's voice, mood, and style are learned. Any GUI control that exposes per-session personality knobs that conflict with this rule is removed or gated before cutover (see PAGE_INVENTORY drift-check). |
| C4 | **One model, one daemon (Enigma AI) + one GUI client (Enigma GUI)** | Sibling package layout, decided May 6, 2026. The GUI rewrite must not collapse this boundary or re-introduce engine code into the GUI process. |

## 2. Non-goals (explicit)

- No cloud sync, telemetry, remote update server, analytics SDKs, or share-mode web UI as the default surface.
- No rewrite before the Phase 1 bake-off POC is complete and reviewed.
- No partial cutover without a rollback flag.
- No changes to `enigma_engine/core/*` driven by GUI work — the service contract adapts to core, not the other way around.
- No new dependency added to `core/*` because the GUI POC wanted it.
- No web UI framework (NiceGUI / Flet / Gradio / Open WebUI) as the **default** shell. Option E is reserved for an opt-in operator console at most.

## 3. Current GUI pain points

Captured from direct reading of `enigma_engine/gui/` modules and from §4 Learned Principles in [AA code maker.md](../../AA%20code%20maker.md).

| # | Pain | Evidence | Phase-0 status |
|---|---|---|---|
| P1 | **Single-threaded Tcl event loop blocks on any long work in a handler.** Long-running tasks must run off the event handler (timer slicing or a worker thread); button callbacks that do real work freeze the UI. | Python docs, [Threading model](https://docs.python.org/3/library/tkinter.html#threading-model). Confirmed in current code by repeated use of `after(0, callback)` + `_model_op_busy()` guards. | Documented. Cannot be removed inside CustomTkinter — only mitigated. |
| P2 | **Hand-rolled theming via `themes.py` is non-standard and brittle.** Custom palette propagation per widget, `CTkEntry.configure(fg_color="transparent")` crashes (Learned Principle), `<Configure>` recursion if not debounced. | `enigma_engine/gui/themes.py`, multiple §4 entries about CTk quirks. | Documented. New shell should use a first-class theming system (Qt stylesheet / Tauri CSS). |
| P3 | **Per-widget rebuild cost is high.** Tooltip dismissal needs watchdog timers, cursor-per-tag binding needs single `<Motion>` handler workaround, fixed-width labels clip text and require `grid_columnconfigure minsize`. | Multiple §4 entries. | Documented. |
| P4 | **30+ direct `from enigma_engine.core.X` imports scattered across `enigma_engine/gui/*.py`** (all but one are deferred / lazy inside functions). This is the surface the Phase 0c service skeleton replaces. | Verified Pass 156z9df by `grep_search "^from enigma_engine\.core" enigma_engine/gui/*.py` (1 module-level) and `grep_search "from enigma_engine\.core"` recursively (200+ deferred). | Documented. Service skeleton lands in Phase 0c. |
| P5 | **No clear engine ↔ GUI process boundary today.** GUI imports core directly and runs the engine in-process. Sibling-package decision (Enigma AI daemon + Enigma GUI client) implies a future IPC seam that does not exist yet. | [ARCH-1 in SUGGESTIONS.md](../../SUGGESTIONS.md). | Documented. The service contract is the precursor — once every GUI/API surface routes through `enigma_engine/services/` `[DELETED dbc19ea, May 25 2026]`, swapping the in-process implementation for an IPC client is a single layer's worth of work. |

## 4. Decision gates (binary pass/fail, scored separately from the rubric)

A track that fails any gate is eliminated regardless of rubric score.

- **G1 — Offline-by-default.** Packet capture during a 10-minute idle + one full FORGE workflow shows zero outbound connections other than loopback. Tool: `pktmon` (built into Windows 10/11) or Wireshark.
- **G2 — No remote update / telemetry by default.** Updater disabled or pinned to local-only feed; no analytics SDK in the dependency tree (verified by `pip list` and the framework's update mechanism docs).
- **G3 — UI does not freeze under model load.** During a synthetic 30-second training step on a worker thread, UI records no frame stall longer than the CustomTkinter baseline measured in Phase 0b on the same workload (primary). Stretch absolute target: < 100 ms. Measured against baseline so "beat baseline" is a real comparison, not an absolute guess.

## 5. Decision criteria (rubric, weighted)

Phase 1 scores each track against this rubric. Weights sum to 100%. Gates G1/G2/G3 are separate and binary.

| Metric | Weight | Target |
|---|---|---|
| Cold start (shell ready to accept input) | 15% | primary: ≤ baseline; stretch: < 2 s absolute |
| Page-switch latency (to CONFIG; FORGE if reached) | 15% | primary: ≤ baseline; stretch: < 200 ms absolute |
| Idle RAM, shell process only (excluding torch + model weights) | 15% | primary: ≤ baseline × 1.5; stretch: ≤ 300 MB absolute |
| Packaged install size, shell + UI runtime only (excludes torch/CUDA/model weights) | 10% | primary: ≤ baseline × 1.5; stretch: ≤ 200 MB compressed absolute |
| Dev velocity (hours to parity on one page) — *soft signal only; familiarity bias disclosed* | 10% | lower is better |
| Packaging/update story on Windows (reproducible build, no remote calls) | 15% | documented + reproducible |
| Theming & layout headroom for future pages | 10% | subjective, recorded with examples |
| User preference (side-by-side use of both prototypes) | 10% | user picks after hands-on |

## 6. Option matrix (candidates entering Phase 1)

| Option | Verdict | Source confidence |
|---|---|---|
| **A. PySide6/Qt desktop rewrite** | **Primary candidate** | medium — [Qt for Python](https://doc.qt.io/qtforpython-6/) broad landing doc |
| **B. Tauri v2 + local Python backend as sidecar** | **Secondary candidate for maximum hardening** | high — [Tauri v2 Security](https://v2.tauri.app/security/), [Sidecar](https://v2.tauri.app/develop/sidecar/) |
| **C. Keep current CustomTkinter and refactor** | **Fallback if migration risk is unacceptable** | high — [Python tkinter threading model](https://docs.python.org/3/library/tkinter.html#threading-model) |
| **D. Electron rewrite** | **Not preferred unless web stack dominates** | high — [Electron Security](https://www.electronjs.org/docs/latest/tutorial/security) |
| **E. Local web UIs (NiceGUI/Flet/Gradio/Open WebUI)** | **Optional operator console only, never default shell** | medium — vendor docs |

## 7. Architecture implications (decision-relevant)

- **[high]** Tauri enforces a core/frontend trust boundary via IPC and capability configuration; frontend only reaches system resources through explicitly exposed commands ([Tauri v2 Security](https://v2.tauri.app/security/)).
- **[high]** Electron can be secured, but default power model requires continuous hardening discipline (context isolation, sandboxing, IPC sender validation, navigation limits, no untrusted remote content) ([Electron Security](https://www.electronjs.org/docs/latest/tutorial/security)).
- **[high]** Tkinter calls from any Python thread are dispatched via the interpreter's event queue (thread-safe by design), but the event loop is single-threaded: any long work in a handler blocks every other event until it returns ([Python threading model](https://docs.python.org/3/library/tkinter.html#threading-model)).
- **[medium]** Web-first local UIs typically run a localhost server; black-box posture depends on bind/auth/network policy rather than framework default alone.

## 8. What this doc does NOT decide

- Which framework wins (Phase 2 decision after Phase 1 bake-off).
- Packaging tool (PyInstaller vs Nuitka for Track A; Tauri bundler config for Track B — decided in Phase 3c).
- Cutover order beyond "low-risk pages first" (driven by [PAGE_INVENTORY.md](PAGE_INVENTORY.md) classification).
- Whether the engine eventually moves to a separate process (ARCH-1) — that is a different slice, but the service contract is its prerequisite.

## 9. Sign-off checklist (before Phase 0 closes)

- [x] Constraints C1–C4 enumerated and tied to project goal.
- [x] Non-goals listed explicitly so future passes do not drift.
- [x] Pain points captured with evidence per row.
- [x] Gates G1/G2/G3 written and scoped.
- [x] Rubric weights sum to 100% and metric targets reference baseline (not arbitrary absolutes).
- [x] Option matrix lists all 5 with confidence-tagged sources.
- [ ] [BASELINE.md](BASELINE.md) measured and numbers filled in (Phase 0b — measurement protocol shipped; numbers pending operator run).
- [ ] [PAGE_INVENTORY.md](PAGE_INVENTORY.md) classification rows complete with drift-check appendix (Phase 0d).
- [ ] `enigma_engine/services/` skeleton merged (Phase 0c). `[OBSOLETE: services/ deleted dbc19ea, May 25 2026; Strategy Reset May 26 2026 chose Gradio over CTk/Tauri tracks]`

When all 9 boxes are checked, Phase 0 is closed and Phase 1 (POC bake-off) is unblocked.
