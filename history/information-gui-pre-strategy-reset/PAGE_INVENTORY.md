# GUI Page Inventory — Phase 0d

**Status:** Inventory shipped. Classification + drift-check pending operator review. Author: AA code maker. Date: May 13, 2026.
**Pass anchor:** GUI-ARCH-0d.

Enumerates every module under `enigma_engine/gui/`. Split into **user-facing pages** (classified per cutover risk) and **support modules** (port strategy per row, no classification). Includes a **project-goal drift-check** appendix per the constraint in [ARCH_DECISION.md](ARCH_DECISION.md) §1.

> **Why the split.** Pages have classification because they cut over individually (Phase 4 PR-per-page). Support modules migrate as infrastructure once the chosen stack is in place — there is no "FORGE-page-equivalent" decision for `themes.py`.

---

## 1. User-facing pages (17 modules)

Each row gets a classification:
- **v1** = must ship in initial cutover (Phase 4 first wave). User-blocking on day one.
- **v2** = can ship after cutover. Useful but not blocking.
- **drop** = does not survive the rewrite. Either replaced by something better or no longer needed.

Classification is **proposed** by audit; operator must confirm/adjust before Phase 4 starts.

| Module | Entry function / class | LOC | Widget complexity | Direct `core.*` imports (deferred, count) | Proposed | Notes |
|---|---|---|---|---|---|---|
| `desktop.py` | `EnigmaGUI` class, sidebar + page host | ~3400 | High — root window, navigation, theme load, model lifecycle, all page wiring | 1 | **v1** | The shell. Must exist on day one. The biggest single module. |
| `gui_pages.py` | `_build_*_page` builders (HOME, MODELS, CONFIG, etc.) | ~2000 | High — most-used pages live here | ~6 | **v1** | Contains CONFIG (the Phase 1 bake-off page). |
| `gui_pages_config.py` | CONFIG-page details | ~600 | Medium — numeric entries, dropdowns, save/load | 0 | **v1** | Phase 1 bake-off target. |
| `gui_pages_forge.py` | FORGE page wiring (top-level radio cards, layout) | ~800 | High — paned layout, 12 modes | 0 | **v1** | FORGE is the most-used training surface. |
| `gui_forge.py` | FORGE shared logic + data prep | ~2600 | High — pre-training pipelines, smoke data, tokenizer build | ~8 | **v1** | Heavy logic file; ports with FORGE page. |
| `gui_forge_training.py` | SFT / pre-train / vision-train launchers | ~2200 | High — multiple training dispatchers | ~22 | **v1** | Critical for any training path. |
| `gui_forge_new_modes.py` | DPO/SimPO/KTO/ORPO/Reasoning/Personality dispatch | ~3400 | High — 10+ alignment modes | ~37 | **v1** | Carries Personality-5 + reasoning GRPO. |
| `gui_forge_adaptive.py` | Adaptive (TRAINER-evaluates-STUDENT) loop GUI | ~1100 | Medium-high | ~11 | **v2** | Less-used path; can ship after initial cutover. |
| `gui_forge_advanced.py` | Advanced pipeline launcher | ~900 | Medium-high | ~9 | **v2** | Lower frequency. |
| `gui_forge_models.py` | MODELS-page extras: merge, download, GGUF export | ~1200 | High | ~22 | **v1** | Merge + GGUF export are first-class flows. |
| `gui_forge_queue.py` | FORGE queue dispatcher | ~500 | Medium — list view, run button | ~7 | **v1** | Queue is integral to FORGE. |
| `gui_forge_tools.py` | FORGE tools (coherence benchmark, web utils, etc.) | ~1500 | Medium — many small tool buttons | ~7 | **v2** | Bench/tool surface, not blocking. |
| `gui_forge_teacher.py` | Teacher-config card for distillation | ~300 | Low — entries + dropdown | 0 | **v1** | Required for distill mode. |
| `gui_cmd_page.py` | Commands / Cmd page (registry-driven) | ~1300 | Medium — command list, output text | ~5 | **v1** | Power-user surface. |
| `gui_docs_page.py` | Docs / RAG ingestion page | ~1000 | Medium — file picker, ingest progress | ~3 | **v1** | Required for RAG. |
| `gui_mods.py` | Mods sidebar / mod manager | ~700 | Medium — mod list, enable/disable | 0 | **v1** | Mods are core to the system. |
| `gui_mod_page.py` | Individual mod-page renderer | ~500 | Medium — per-mod UI builder | 0 | **v1** | Same. |

**Page-count total:** 17 user-facing modules. v1 count: **13**. v2 count: **4**. drop count: 0.

**LOC + import counts** are coarse estimates (`Get-ChildItem ... | Measure-Object Lines`-grade) and don't need to be precise — they exist to flag the biggest cutover PRs. Operator can replace them with exact numbers during review.

## 2. Support modules (7 modules)

Not classified by cutover wave because they are infrastructure. Each row gets a **port strategy**: `direct port` (line-for-line equivalent in new stack), `rewrite` (rebuild from scratch in the new stack's idioms), or `drop` (no longer needed).

| Module | Purpose | Port strategy | Notes |
|---|---|---|---|
| `widgets.py` | Custom widgets (CTk extensions, factory helpers) | **rewrite** | Qt/Tauri have first-class widget systems; hand-rolled CTk wrappers do not map 1-to-1. |
| `themes.py` | Hand-rolled palette + atomic theme persistence | **rewrite** | Replaced by Qt stylesheet / Tauri CSS. Theme **persistence** logic (atomic write) is retained but moves to `enigma_engine/services/` `[DELETED dbc19ea, May 25 2026]`. |
| `scanners.py` | Filesystem scanners (model registry, mods) | **direct port** | Pure logic. Already mostly framework-agnostic; lives behind a service. |
| `media.py` | Image / media helpers for chat surface | **rewrite** | Qt and Tauri have native image-handling pipelines; the current PIL+CTk shim becomes redundant. |
| `gui_logic.py` | Chat dispatch, command runner, RAG wiring (mixin-style) | **direct port → service** | Already mixin-style. Port the logic, replace direct `core.*` imports with service calls. |
| `gui_logic_chat.py` | Chat send/stream/render mixin | **direct port → service** | Same. Highest-frequency direct-import file (model_context, sentiment, memory, auto_research, reasoning). |
| `gui_logic_media.py` | Media attach / save / preview mixin | **direct port** | Small; lives behind a service. |

## 3. Project-goal drift check (binary pass/fail)

Per [ARCH_DECISION.md](ARCH_DECISION.md) §1 constraint C3: *"Personality from training, not the user"*. The check below flags any GUI surface that lets the user hand-author personality, identity, or internal state, AND any surface that exposes raw model internals beyond debugging/training need.

| Page / surface | Widget(s) | Risk | Disposition |
|---|---|---|---|
| Profile editor (if any) | Profile JSON editor on CONFIG page | **C3 risk** if it exposes free-text personality fields. Verify against actual widget code before cutover. | **Investigate before v1 cutover.** Pass 156z9dd already removed `AIProfile.personality` config layer (no widget should write to it); confirm the profile JSON editor either does not exist or only edits the allowed fields (`name`, `system_prompt`, `task` overlays — NOT personality, NOT identity). |
| `gui_pages.py` emotional_state debug readout | Live `_EMOTIONAL_RANGES` display (line ~629) | **OK** — emotional_state is AI-computed runtime state, not user config. Read-only display is in scope (the AI knowing itself). | **Keep.** Display only, no writeback. Re-verify "no writeback" before cutover. |
| Identity / "tell the AI who it is" widgets | None known | — | **None to flag** (Pass 156z9dd cleanup). Re-verify by grep against `personality` after Phase 0 closes. |
| Raw model internals (weights, logits, hidden states) | None exposed in standard pages | — | **None to flag.** Forge tools surface metrics/benchmarks (coherence score) but not raw tensors. |

**Drift-check status:** No active violations identified at Phase 0d. **Operator must re-verify before Phase 4 v1 cutover** — any v1 page that introduces a personality-authoring widget between now and cutover fails C3 and must be removed or gated.

## 4. Cutover-order proposal (Phase 4 wave 1)

Driven by P5 (engine ↔ GUI boundary doesn't exist yet) + risk classification above. Lower-risk pages first so the service contract is exercised under low blast radius before FORGE.

1. CONFIG (Phase 1 bake-off page — already migrated by end of Phase 1 in POC form).
2. HOME / MODELS list pages (read-mostly, simple state).
3. CMD page (registry-driven, mostly display).
4. DOCS / RAG page.
5. MODS page + individual mod pages.
6. Chat surface (gui_logic_chat + media).
7. FORGE — biggest cutover, last. Pages: `gui_pages_forge.py`, `gui_forge.py`, `gui_forge_training.py`, `gui_forge_new_modes.py`, `gui_forge_models.py`, `gui_forge_queue.py`, `gui_forge_teacher.py`.
8. v2 pages: `gui_forge_adaptive.py`, `gui_forge_advanced.py`, `gui_forge_tools.py`.

After step 7 finishes AND `api/server.py` is migrated (per GUI-ARCH-4-api), legacy code path is removed (GUI-ARCH-4-final).

## 5. Acceptance for Phase 0d close

- [ ] §1 page classification confirmed by operator (any v2 → v1 or v1 → v2 disagreements logged).
- [ ] §2 support-module port strategies confirmed.
- [ ] §3 drift check re-run AFTER Phase 0 closes (in case any new widget snuck in between then and Phase 4 start).
- [ ] §4 cutover order accepted or adjusted.
- [ ] Final doc cross-referenced in [ARCH_DECISION.md](ARCH_DECISION.md) §9 sign-off checklist.
