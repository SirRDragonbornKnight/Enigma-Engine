"""
Training Diagnostic — exercises the same code path as the GUI.

Runs a minimal pre-train + solo fine-tune loop on real data and
reports exactly what happens at each stage.  Output goes to console
so issues are visible without a GUI.

Usage:
    python test_training_diagnostic.py                  # Quick (2 epochs, nano)
    python test_training_diagnostic.py --full            # Longer (5 epochs, small)
    python test_training_diagnostic.py -v                # Verbose — show every callback
    python test_training_diagnostic.py -f pretrain       # Only run pretrain step
    python test_training_diagnostic.py -f pretrain,callbacks  # Combine filters
    python test_training_diagnostic.py --report          # Save report to logs/
    python test_training_diagnostic.py -v --report       # Verbose + report

Filters: imports, hardware, data, tokenizer, pretrain, finetune, callbacks
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
#  Event log — captures every callback/event with timestamps
# ---------------------------------------------------------------------------
class EventLog:
    """Records timestamped events during the diagnostic run."""

    CATEGORIES = ("system", "progress", "loss", "epoch", "throughput", "data", "error", "result")

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.events: list[dict] = []
        self._start = time.monotonic()

    def log(self, category: str, message: str, **data):
        elapsed = time.monotonic() - self._start
        entry = {
            "t": round(elapsed, 3),
            "cat": category,
            "msg": message,
            **data,
        }
        self.events.append(entry)
        if self.verbose:
            tag = f"[{category:>10s}]"
            detail = ""
            for k, v in data.items():
                if isinstance(v, float):
                    detail += f" {k}={v:.4f}"
                else:
                    detail += f" {k}={v}"
            print(f"  {elapsed:7.2f}s {tag} {message}{detail}")

    def get_events(self, category: str | None = None) -> list[dict]:
        if category is None:
            return list(self.events)
        return [e for e in self.events if e["cat"] == category]

    def to_dict(self) -> dict:
        return {
            "event_count": len(self.events),
            "categories": {cat: len(self.get_events(cat)) for cat in self.CATEGORIES},
            "events": self.events,
        }


# Global log — set by main()
_log: EventLog = EventLog()


def _hr(title: str = ""):
    if title:
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print(f"{'=' * 50}")
    else:
        print("-" * 50)


def _check_imports():
    """Verify core imports work."""
    _hr("STEP 1: Import Check")
    errors = []
    modules = [
        ("torch", "import torch"),
        ("model", "from enigma_engine.core.model import Enigma"),
        ("presets", "from enigma_engine.core.model_presets import get_preset, ForgeConfig"),
        ("training", "from enigma_engine.training.training import Trainer, TrainingConfig"),
        ("tokenizer", "from enigma_engine.core.tokenizer import get_tokenizer"),
        ("dataset", "from enigma_engine.core.dataset import process_text_corpus, estimate_token_count"),
        ("safe_save", "from enigma_engine.core.safe_save import atomic_torch_save"),
        ("bpe_tokenizer", "from enigma_engine.core.bpe_tokenizer import BPETokenizer"),
    ]
    for name, stmt in modules:
        try:
            exec(stmt)  # noqa: S102  -- stmt comes from the hardcoded import list above, not user input
            print(f"  [OK] {name}")
            _log.log("system", f"import {name}", status="ok")
        except Exception as exc:
            print(f"  [FAIL] {name}: {exc}")
            _log.log("error", f"import {name} failed", error=str(exc))
            errors.append(name)

    if errors:
        print(f"\n  BLOCKED: {len(errors)} import(s) failed. Cannot continue.")
        return False
    print(f"\n  All {len(modules)} imports OK")
    return True


def _check_hardware():
    """Report hardware info."""
    _hr("STEP 2: Hardware")
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device   : {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / 1e9
        free = torch.cuda.mem_get_info(0)
        free_gb = free[0] / 1e9
        print(f"  GPU      : {gpu_name}")
        print(f"  VRAM     : {total_gb:.1f} GB")
        print(f"  Free     : {free_gb:.1f} GB")
        _log.log(
            "system",
            "hardware detected",
            device=device,
            gpu=gpu_name,
            vram_gb=round(total_gb, 1),
            free_gb=round(free_gb, 1),
        )
    else:
        import psutil

        ram = psutil.virtual_memory()
        print(f"  RAM      : {ram.total / 1e9:.1f} GB ({ram.available / 1e9:.1f} GB free)")
        _log.log("system", "hardware detected", device="cpu", ram_gb=round(ram.total / 1e9, 1))
    return device


def _check_data():
    """Find usable training data."""
    _hr("STEP 3: Data Check")
    data_dir = Path("data")
    pretrain_dir = data_dir / "pretrain"

    # Fine-tune data
    txt_files = sorted(data_dir.glob("*.txt"), key=lambda p: p.stat().st_size, reverse=True)
    if txt_files:
        biggest = txt_files[0]
        print(f"  Fine-tune data: {biggest.name} ({biggest.stat().st_size / 1024:.1f} KB)")
    else:
        biggest = None
        print("  Fine-tune data: NONE FOUND")

    # Pre-train data
    if pretrain_dir.exists():
        pretrain_files = list(pretrain_dir.rglob("*.txt"))
        total_size = sum(f.stat().st_size for f in pretrain_files)
        print(f"  Pre-train data: {len(pretrain_files)} files ({total_size / 1e6:.1f} MB)")
    else:
        pretrain_files = []
        print("  Pre-train data: NONE (data/pretrain/ not found)")

    return biggest, pretrain_files


def _test_tokenizer():
    """Test tokenizer loading and encoding."""
    _hr("STEP 4: Tokenizer")
    from enigma_engine.core.tokenizer import get_tokenizer

    tokenizer = get_tokenizer("auto")
    ttype = type(tokenizer).__name__
    vocab = tokenizer.vocab_size
    print(f"  Type     : {ttype}")
    print(f"  Vocab    : {vocab:,}")

    test = "Hello, how are you today? The quick brown fox jumps."
    tokens = tokenizer.encode(test)
    decoded = tokenizer.decode(tokens)
    print(f"  Test encode: {len(test)} chars -> {len(tokens)} tokens")
    print(f"  Tokens   : {tokens[:15]}{'...' if len(tokens) > 15 else ''}")
    print(f"  Decoded  : '{decoded[:60]}'")

    if len(tokens) == 0:
        print("  [FAIL] Tokenizer produced 0 tokens!")
        return None
    if vocab < 100:
        print(f"  [WARN] Very small vocab ({vocab}) — training may be slow")

    return tokenizer


def _test_pretrain(device: str, pretrain_files: list[Path], tokenizer, epochs: int, preset_name: str):
    """Test pre-training from scratch."""
    _hr("STEP 5: Pre-Train Test")
    import torch
    from enigma_engine.core.model import Enigma
    from enigma_engine.core.model_presets import get_preset
    from enigma_engine.training.training import Trainer, TrainingConfig

    # Use a small sample of pretrain data — cap at ~200K chars for speed
    MAX_CHARS = 200_000
    print(f"  Sampling up to {MAX_CHARS:,} chars from pretrain files...")

    text = ""
    files_used = 0
    for f in pretrain_files:
        try:
            t = f.read_text(encoding="utf-8", errors="replace")
            text += t + "\n\n"
            files_used += 1
        except Exception as exc:
            print(f"  [WARN] Skip {f.name}: {exc}")
        if len(text) >= MAX_CHARS:
            text = text[:MAX_CHARS]
            break
    print(f"  Used {files_used} files")

    print(f"  Data     : {len(text):,} chars")
    if len(text) < 100:
        print("  [FAIL] Not enough data. Need at least 100 chars.")
        return False

    # Create model from preset
    print(f"  Preset   : {preset_name}")
    config = get_preset(preset_name, vocab_size=tokenizer.vocab_size)
    model = Enigma(config=config).to(device)
    pc = sum(p.numel() for p in model.parameters())
    print(f"  Params   : {pc:,}")
    print(f"  Seq len  : {config.max_seq_len}")

    # Chunk text
    max_seq = config.max_seq_len
    chars_per_chunk = max(1000, max_seq * 3)
    chunks = []
    pos = 0
    while pos < len(text):
        end = min(pos + chars_per_chunk, len(text))
        if end < len(text):
            boundary = text.rfind(" ", pos, end)
            nl = text.rfind("\n", pos, end)
            boundary = max(boundary, nl)
            if boundary > pos:
                end = boundary + 1
        chunk = text[pos:end].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        pos = end
    text = "\n\n".join(chunks)
    print(f"  Chunks   : {len(chunks)}")

    # Training config
    train_config = TrainingConfig(
        epochs=epochs,
        batch_size=0,  # auto
        learning_rate=3e-4,
        general_mix_ratio=0.0,
        val_split=0.1,
        save_every=0,  # don't checkpoint during diagnostic
        checkpoint_dir="models/checkpoints",
        use_amp=torch.cuda.is_available(),
        run_evaluation=False,  # skip eval for speed
    )

    trainer = Trainer(model, tokenizer, train_config)
    print(f"  Batch sz : {trainer.config.batch_size} (auto-estimated)")

    # Callbacks — track everything
    results = {
        "steps": 0,
        "losses": [],
        "epoch_losses": [],
        "tokens_total": 0,
        "errors": [],
        "progress_calls": 0,
        "loss_calls": 0,
    }
    t0 = time.monotonic()

    def on_progress(pct, msg):
        results["progress_calls"] += 1
        _log.log("progress", f"pretrain {pct:.0f}%", pct=pct, detail=str(msg))

    def on_loss(loss):
        results["loss_calls"] += 1
        results["steps"] += 1
        results["losses"].append(loss)
        _log.log("loss", f"step {results['steps']}", loss=loss)

    def on_epoch(epoch, loss):
        results["epoch_losses"].append(loss)
        elapsed = time.monotonic() - t0
        print(f"    Epoch {epoch}/{epochs} | loss {loss:.4f} | {elapsed:.1f}s")
        _log.log("epoch", f"pretrain epoch {epoch}/{epochs}", loss=loss, elapsed=round(elapsed, 1))

    def on_throughput(tokens, step_time):
        results["tokens_total"] += tokens
        tps = tokens / step_time if step_time > 0 else 0
        _log.log("throughput", "batch done", tokens=tokens, step_time=round(step_time, 4), tokens_per_sec=round(tps))

    trainer.on_progress = on_progress
    trainer.on_loss = on_loss
    trainer.on_epoch_complete = on_epoch
    trainer.on_throughput = on_throughput

    print(f"\n  Training ({epochs} epochs)...")
    _log.log("system", "pretrain started", epochs=epochs, batch_size=trainer.config.batch_size, chunks=len(chunks))
    try:
        state = trainer.train(text)
    except Exception as exc:
        print(f"\n  [FAIL] Training crashed: {exc}")
        traceback.print_exc()
        _log.log("error", "pretrain crashed", error=str(exc), traceback=traceback.format_exc())
        return False

    elapsed = time.monotonic() - t0
    speed = state.total_tokens / elapsed if elapsed > 0 else 0
    print("\n  --- RESULTS ---")
    print(f"  Time     : {elapsed:.1f}s")
    print(f"  Steps    : {state.step}")
    print(f"  Tokens   : {state.total_tokens:,}")
    if elapsed > 0:
        print(f"  Speed    : {speed:,.0f} tok/s")
    print(f"  Best loss: {state.best_loss:.4f}")
    print(f"  Epoch losses: {[f'{l:.4f}' for l in results['epoch_losses']]}")
    print(f"  Callbacks: {results['progress_calls']} progress, {results['loss_calls']} loss")

    _log.log(
        "result",
        "pretrain finished",
        elapsed=round(elapsed, 1),
        steps=state.step,
        tokens=state.total_tokens,
        speed=round(speed),
        best_loss=round(state.best_loss, 4),
        epoch_losses=[round(l, 4) for l in results["epoch_losses"]],
        progress_calls=results["progress_calls"],
        loss_calls=results["loss_calls"],
    )

    # Sanity checks
    ok = True
    if math.isinf(state.best_loss):
        print("  [FAIL] best_loss is inf — training didn't produce any valid loss")
        ok = False
    if state.step == 0:
        print("  [FAIL] 0 steps completed — training loop didn't execute")
        ok = False
    if results["epoch_losses"] and results["epoch_losses"][-1] > 100:
        print(f"  [WARN] Final loss very high ({results['epoch_losses'][-1]:.1f})")
    if len(results["epoch_losses"]) >= 2:
        if results["epoch_losses"][-1] < results["epoch_losses"][0]:
            print(f"  [OK] Loss decreased: {results['epoch_losses'][0]:.4f} -> {results['epoch_losses'][-1]:.4f}")
        else:
            print("  [WARN] Loss did not decrease — model may not be learning")

    if ok:
        print("\n  PRE-TRAIN: PASS")
    else:
        print("\n  PRE-TRAIN: FAIL")

    # Cleanup
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ok


def _test_finetune(device: str, data_file: Path, tokenizer, epochs: int, preset_name: str):
    """Test fine-tuning on existing data."""
    _hr("STEP 6: Fine-Tune Test")
    import torch
    from enigma_engine.core.model import Enigma
    from enigma_engine.core.model_presets import get_preset
    from enigma_engine.training.training import Trainer, TrainingConfig

    text = data_file.read_text(encoding="utf-8")
    if len(text) > 200_000:
        text = text[:200_000]
    print(f"  Data     : {data_file.name} ({len(text):,} chars)")

    if len(text) < 100:
        print("  [SKIP] Data too small for fine-tune test")
        return True

    config = get_preset(preset_name, vocab_size=tokenizer.vocab_size)
    model = Enigma(config=config).to(device)
    pc = sum(p.numel() for p in model.parameters())
    print(f"  Preset   : {preset_name} ({pc:,} params)")

    train_config = TrainingConfig(
        epochs=epochs,
        batch_size=0,
        learning_rate=5e-5,
        general_mix_ratio=0.0,
        val_split=0.1,
        save_every=0,
        checkpoint_dir="models/checkpoints",
        use_amp=torch.cuda.is_available(),
        run_evaluation=False,
    )

    trainer = Trainer(model, tokenizer, train_config)
    print(f"  Batch sz : {trainer.config.batch_size} (auto)")

    results = {"epoch_losses": [], "progress_calls": 0, "loss_calls": 0, "losses": []}
    t0 = time.monotonic()

    def on_progress(pct, msg):
        results["progress_calls"] += 1
        _log.log("progress", f"finetune {pct:.0f}%", pct=pct, detail=str(msg))

    def on_loss(loss):
        results["loss_calls"] += 1
        results["losses"].append(loss)
        _log.log("loss", f"finetune step {results['loss_calls']}", loss=loss)

    def on_epoch(epoch, loss):
        results["epoch_losses"].append(loss)
        elapsed = time.monotonic() - t0
        print(f"    Epoch {epoch}/{epochs} | loss {loss:.4f} | {elapsed:.1f}s")
        _log.log("epoch", f"finetune epoch {epoch}/{epochs}", loss=loss, elapsed=round(elapsed, 1))

    def on_throughput(tokens, step_time):
        tps = tokens / step_time if step_time > 0 else 0
        _log.log(
            "throughput", "finetune batch", tokens=tokens, step_time=round(step_time, 4), tokens_per_sec=round(tps)
        )

    trainer.on_progress = on_progress
    trainer.on_loss = on_loss
    trainer.on_epoch_complete = on_epoch
    trainer.on_throughput = on_throughput

    print(f"\n  Training ({epochs} epochs)...")
    _log.log("system", "finetune started", epochs=epochs, batch_size=trainer.config.batch_size, data_chars=len(text))
    try:
        state = trainer.train(text)
    except Exception as exc:
        print(f"\n  [FAIL] Fine-tune crashed: {exc}")
        traceback.print_exc()
        _log.log("error", "finetune crashed", error=str(exc), traceback=traceback.format_exc())
        return False

    elapsed = time.monotonic() - t0
    print("\n  --- RESULTS ---")
    print(f"  Time     : {elapsed:.1f}s")
    print(f"  Steps    : {state.step}")
    print(f"  Best loss: {state.best_loss:.4f}")
    print(f"  Epoch losses: {[f'{l:.4f}' for l in results['epoch_losses']]}")
    print(f"  Callbacks: {results['progress_calls']} progress, {results['loss_calls']} loss")

    _log.log(
        "result",
        "finetune finished",
        elapsed=round(elapsed, 1),
        steps=state.step,
        best_loss=round(state.best_loss, 4),
        epoch_losses=[round(l, 4) for l in results["epoch_losses"]],
        progress_calls=results["progress_calls"],
        loss_calls=results["loss_calls"],
    )

    ok = True
    if math.isinf(state.best_loss):
        print("  [FAIL] best_loss is inf")
        ok = False
    if state.step == 0:
        print("  [FAIL] 0 steps")
        ok = False

    if ok:
        print("\n  FINE-TUNE: PASS")
    else:
        print("\n  FINE-TUNE: FAIL")

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ok


def _test_gui_callbacks():
    """Test that GUI callback wiring works (no actual GUI needed)."""
    _hr("STEP 7: Callback Wiring")
    import torch
    from enigma_engine.core.model import Enigma
    from enigma_engine.core.model_presets import get_preset
    from enigma_engine.training.training import Trainer, TrainingConfig
    from enigma_engine.core.tokenizer import get_tokenizer

    tokenizer = get_tokenizer("auto")
    config = get_preset("nano", vocab_size=tokenizer.vocab_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Enigma(config=config).to(device)

    train_config = TrainingConfig(
        epochs=1,
        batch_size=2,
        learning_rate=1e-4,
        save_every=0,
        checkpoint_dir="models/checkpoints",
        use_amp=False,
        run_evaluation=False,
    )

    trainer = Trainer(model, tokenizer, train_config)

    counts = {"progress": 0, "loss": 0, "epoch": 0, "throughput": 0}

    def _cb_progress(p, m):
        counts["progress"] += 1
        _log.log("progress", f"callback-test {p:.0f}%", pct=p)

    def _cb_loss(l):
        counts["loss"] += 1
        _log.log("loss", f"callback-test step {counts['loss']}", loss=l)

    def _cb_epoch(e, l):
        counts["epoch"] += 1
        _log.log("epoch", f"callback-test epoch {e}", loss=l)

    def _cb_throughput(t, s):
        counts["throughput"] += 1
        tps = t / s if s > 0 else 0
        _log.log("throughput", "callback-test batch", tokens=t, tokens_per_sec=round(tps))

    trainer.on_progress = _cb_progress
    trainer.on_loss = _cb_loss
    trainer.on_epoch_complete = _cb_epoch
    trainer.on_throughput = _cb_throughput

    # Minimal training data
    data = "The quick brown fox jumps over the lazy dog.\n" * 50
    data += "User: Hello\nAssistant: Hi there!\n" * 20

    try:
        state = trainer.train(data)
        print(f"  Progress calls : {counts['progress']}")
        print(f"  Loss calls     : {counts['loss']}")
        print(f"  Epoch calls    : {counts['epoch']}")
        print(f"  Throughput calls: {counts['throughput']}")
        print(f"  Steps completed: {state.step}")

        ok = True
        if counts["epoch"] != 1:
            print(f"  [FAIL] Expected 1 epoch callback, got {counts['epoch']}")
            ok = False
        if counts["progress"] == 0:
            print("  [FAIL] No progress callbacks fired")
            ok = False
        if counts["loss"] == 0:
            print("  [WARN] No loss callbacks fired (may be < log_every steps)")
        if counts["throughput"] == 0:
            print("  [FAIL] No throughput callbacks fired")
            ok = False
        if ok:
            print("\n  CALLBACKS: PASS")
        else:
            print("\n  CALLBACKS: FAIL")
        return ok

    except Exception as exc:
        print(f"  [FAIL] Callback test crashed: {exc}")
        traceback.print_exc()
        return False
    finally:
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _write_report(report_data: dict, report_path: Path):
    """Write diagnostic report to file."""
    lines = []
    lines.append("=" * 60)
    lines.append("  TRAINING DIAGNOSTIC REPORT")
    lines.append(f"  Generated: {report_data['timestamp']}")
    lines.append("=" * 60)
    lines.append("")

    # Config
    cfg = report_data["config"]
    lines.append(f"  Mode   : {cfg['mode']}")
    lines.append(f"  Preset : {cfg['preset']}")
    lines.append(f"  Epochs : {cfg['epochs']}")
    lines.append(f"  Filter : {cfg.get('filter', 'all')}")
    lines.append(f"  Verbose: {cfg.get('verbose', False)}")
    lines.append("")

    # Hardware
    if "hardware" in report_data:
        hw = report_data["hardware"]
        lines.append("  HARDWARE")
        lines.append(f"    Device : {hw.get('device', '?')}")
        lines.append(f"    GPU    : {hw.get('gpu', 'N/A')}")
        lines.append(f"    VRAM   : {hw.get('vram_gb', '?')} GB")
        lines.append(f"    Free   : {hw.get('free_gb', '?')} GB")
        lines.append("")

    # Results summary
    if "results" in report_data:
        lines.append("  RESULTS")
        for name, result in report_data["results"].items():
            if result is None:
                status = "SKIP"
            elif result:
                status = "PASS"
            else:
                status = "FAIL"
            lines.append(f"    {name:15s} : {status}")
        lines.append("")

    # Event log
    events = report_data.get("events", {})
    if events.get("event_count", 0) > 0:
        lines.append(f"  EVENT LOG ({events['event_count']} events)")
        cats = events.get("categories", {})
        for cat, count in cats.items():
            if count > 0:
                lines.append(f"    {cat:>12s} : {count}")
        lines.append("")

        lines.append("  DETAILED EVENTS")
        lines.append(f"  {'Time':>8s}  {'Category':>10s}  Message")
        lines.append("  " + "-" * 56)
        for ev in events.get("events", []):
            detail_parts = []
            for k, v in ev.items():
                if k in ("t", "cat", "msg"):
                    continue
                if isinstance(v, float):
                    detail_parts.append(f"{k}={v:.4f}")
                elif isinstance(v, str) and len(v) > 80:
                    detail_parts.append(f"{k}=<{len(v)} chars>")
                else:
                    detail_parts.append(f"{k}={v}")
            detail = " | " + ", ".join(detail_parts) if detail_parts else ""
            lines.append(f"  {ev['t']:7.2f}s  [{ev['cat']:>10s}]  {ev['msg']}{detail}")
        lines.append("")

    lines.append(f"  Total time: {report_data.get('total_time', '?')}s")
    lines.append("")

    report_text = "\n".join(lines)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n  Report saved: {report_path}")

    # Also save raw JSON for programmatic use
    json_path = report_path.with_suffix(".json")
    json_path.write_text(json.dumps(report_data, indent=2, default=str), encoding="utf-8")
    print(f"  JSON data : {json_path}")


def main():
    global _log

    parser = argparse.ArgumentParser(
        description="Training diagnostic — exercises training code paths",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Filters: imports, hardware, data, tokenizer, pretrain, finetune, callbacks\n"
        "Examples:\n"
        "  python test_training_diagnostic.py -v -f pretrain\n"
        "  python test_training_diagnostic.py --report --full\n"
        "  python test_training_diagnostic.py -v -f pretrain,callbacks --report\n",
    )
    parser.add_argument("--full", action="store_true", help="Run longer test (5 epochs, small preset)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show every callback trigger in real-time")
    parser.add_argument(
        "-f",
        "--filter",
        type=str,
        default=None,
        help="Comma-separated list of steps to run (imports,hardware,data,tokenizer,pretrain,finetune,callbacks)",
    )
    parser.add_argument(
        "--report",
        nargs="?",
        const="auto",
        default=None,
        help="Save report to file (default: logs/diagnostic_<timestamp>.txt)",
    )
    # Keep old flags for compat
    parser.add_argument("--pretrain-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--finetune-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Resolve filter
    ALL_STEPS = ["imports", "hardware", "data", "tokenizer", "pretrain", "finetune", "callbacks"]
    if args.filter:
        active = [s.strip().lower() for s in args.filter.split(",")]
        invalid = [s for s in active if s not in ALL_STEPS]
        if invalid:
            print(f"Unknown filter(s): {', '.join(invalid)}")
            print(f"Valid: {', '.join(ALL_STEPS)}")
            sys.exit(1)
    elif args.pretrain_only:
        active = ["imports", "hardware", "data", "tokenizer", "pretrain"]
    elif args.finetune_only:
        active = ["imports", "hardware", "data", "tokenizer", "finetune"]
    else:
        active = list(ALL_STEPS)

    if args.full:
        epochs = 5
        preset = "small"
    else:
        epochs = 2
        preset = "nano"

    # Init event log
    _log = EventLog(verbose=args.verbose)

    _hr("TRAINING DIAGNOSTIC")
    mode_str = "full" if args.full else "quick"
    filter_str = ",".join(active) if args.filter else "all"
    print(f"  Mode   : {mode_str}")
    print(f"  Preset : {preset}")
    print(f"  Epochs : {epochs}")
    print(f"  Verbose: {args.verbose}")
    print(f"  Filter : {filter_str}")
    if args.report:
        print(f"  Report : {'auto' if args.report == 'auto' else args.report}")
    t_start = time.monotonic()

    _log.log(
        "system",
        "diagnostic started",
        mode=mode_str,
        preset=preset,
        epochs=epochs,
        filter=filter_str,
        verbose=args.verbose,
    )

    device = "cpu"
    tokenizer = None
    finetune_data = None
    pretrain_files = []

    # Step 1: Imports (always run — needed for everything)
    if "imports" in active:
        if not _check_imports():
            sys.exit(1)

    # Step 2: Hardware
    if "hardware" in active:
        device = _check_hardware()
    else:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 3: Data
    if "data" in active:
        finetune_data, pretrain_files = _check_data()
    else:
        # Still need data paths for training steps
        data_dir = Path("data")
        pretrain_dir = data_dir / "pretrain"
        txt_files = sorted(data_dir.glob("*.txt"), key=lambda p: p.stat().st_size, reverse=True)
        finetune_data = txt_files[0] if txt_files else None
        pretrain_files = list(pretrain_dir.rglob("*.txt")) if pretrain_dir.exists() else []

    # Step 4: Tokenizer
    if "tokenizer" in active:
        tokenizer = _test_tokenizer()
        if tokenizer is None:
            sys.exit(1)
    else:
        from enigma_engine.core.tokenizer import get_tokenizer

        tokenizer = get_tokenizer("auto")

    results = {}

    # Step 5: Pre-train
    if "pretrain" in active:
        if pretrain_files:
            results["pretrain"] = _test_pretrain(device, pretrain_files, tokenizer, epochs, preset)
        else:
            print("\n  [SKIP] No pretrain data available")
            results["pretrain"] = None

    # Step 6: Fine-tune
    if "finetune" in active:
        if finetune_data and finetune_data.stat().st_size > 100:
            results["finetune"] = _test_finetune(device, finetune_data, tokenizer, epochs, preset)
        else:
            print("\n  [SKIP] No fine-tune data available")
            results["finetune"] = None

    # Step 7: Callbacks
    if "callbacks" in active:
        results["callbacks"] = _test_gui_callbacks()

    # Summary
    total_time = round(time.monotonic() - t_start, 1)
    _log.log("system", "diagnostic finished", total_time=total_time)

    _hr("SUMMARY")
    for name, result in results.items():
        if result is None:
            status = "SKIP"
        elif result:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  {name:15s} : {status}")
    print(f"\n  Total time: {total_time}s")

    # Event summary
    event_counts = {cat: len(_log.get_events(cat)) for cat in EventLog.CATEGORIES}
    active_cats = {k: v for k, v in event_counts.items() if v > 0}
    if active_cats:
        print(f"\n  Events recorded: {len(_log.events)} total")
        for cat, cnt in active_cats.items():
            print(f"    {cat:>12s} : {cnt}")

    # Report
    if args.report is not None:
        if args.report == "auto":
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = Path("logs") / f"diagnostic_{ts}.txt"
        else:
            report_path = Path(args.report)

        # Gather hardware info from events
        hw_events = [e for e in _log.events if e["cat"] == "system" and "device" in e]
        hw_data = hw_events[0] if hw_events else {}

        report_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "mode": mode_str,
                "preset": preset,
                "epochs": epochs,
                "filter": filter_str,
                "verbose": args.verbose,
            },
            "hardware": hw_data,
            "results": {k: v for k, v in results.items()},
            "events": _log.to_dict(),
            "total_time": total_time,
        }
        _write_report(report_data, report_path)

    if any(v is False for v in results.values()):
        print("\n  RESULT: SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("\n  RESULT: ALL TESTS PASSED")


if __name__ == "__main__":
    main()
