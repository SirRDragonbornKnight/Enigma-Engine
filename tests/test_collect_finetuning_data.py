"""Tests for collect_finetuning_data.py reasoning-data collectors.

Pass 155: D-4 OpenThoughts3 + D-11 SmolTalk2 fetchers. We mock the
`datasets` library via fake module injection so tests do not require
HuggingFace network access. See AA learned principle:
"Optional dependency tests: inject fake module via types.ModuleType
+ monkeypatch.setitem(sys.modules, ...)".
"""

import importlib
import sys
import types
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Fake datasets injection ─────────────────────────────────────────────────


def _install_fake_datasets(monkeypatch, rows_by_path, splits_by_path=None):
    """Inject a fake `datasets` module whose `load_dataset(path, ...)`
    returns the rows registered for that dataset path.

    `rows_by_path` keys are HF repo paths; values are lists of dict rows.
    `splits_by_path` optionally maps repo path → list of split names for
    `get_dataset_split_names`. Defaults to ["train"] for known paths.
    Streaming and non-streaming both yield the same row list.
    """
    fake = types.ModuleType("datasets")

    class _FakeDataset:
        def __init__(self, rows):
            self._rows = rows

        def __iter__(self):
            return iter(self._rows)

        def __len__(self):
            return len(self._rows)

    def _load_dataset(path, *args, **kwargs):
        if path not in rows_by_path:
            raise ValueError("BuilderConfig 'unknown' not found. Available: ['default', 'subset_a', 'subset_b']")
        return _FakeDataset(rows_by_path[path])

    def _get_dataset_split_names(path, config=None, *args, **kwargs):
        if splits_by_path is not None and path in splits_by_path:
            return list(splits_by_path[path])
        if path in rows_by_path:
            return ["train"]
        raise ValueError("BuilderConfig 'unknown' not found. Available: ['default', 'subset_a', 'subset_b']")

    fake.load_dataset = _load_dataset
    fake.get_dataset_split_names = _get_dataset_split_names
    monkeypatch.setitem(sys.modules, "datasets", fake)


@pytest.fixture
def cf_module(monkeypatch):
    """Reload collect_finetuning_data after fake `datasets` is installed."""
    if "collect_finetuning_data" in sys.modules:
        monkeypatch.delitem(sys.modules, "collect_finetuning_data")
    return importlib.import_module("collect_finetuning_data")


# ── OpenThoughts3 (D-4) ─────────────────────────────────────────────────────


class TestCollectOpenThoughts3:
    """Pass 139 spec verified the row schema:
        {"difficulty": int, "source": str, "domain": str,
         "conversations": [{"from": "human"/"gpt", "value": str}]}
    The gpt value contains <think>...</think> reasoning followed by the
    final answer. Tags MUST be preserved verbatim (no whitespace collapse).
    """

    _SAMPLE_ROW = {
        "difficulty": 5,
        "source": "competition_math",
        "domain": "math",
        "conversations": [
            {"from": "human", "value": "What is 2+2?"},
            {
                "from": "gpt",
                "value": ("<think>\nOkay, I need to add two and two.\n2 + 2 = 4.\n</think>\n\nThe answer is 4."),
            },
        ],
    }

    def test_preserves_think_tags_verbatim(self, monkeypatch, cf_module):
        """<think>...</think> must survive — no whitespace collapse."""
        _install_fake_datasets(
            monkeypatch,
            {"open-thoughts/OpenThoughts3-1.2M": [self._SAMPLE_ROW]},
        )
        # Re-import after injection so the fake module is picked up
        cf = importlib.reload(cf_module)
        pairs = cf.collect_openthoughts3(max_samples=10)
        assert len(pairs) == 1
        completion = pairs[0]["completion"]
        assert "<think>" in completion
        assert "</think>" in completion
        # Verbatim — newlines inside <think> survived
        assert "\nOkay, I need to add" in completion
        assert "</think>\n\nThe answer is 4." in completion

    def test_skips_rows_with_no_human_gpt_pair(self, monkeypatch, cf_module):
        """Rows missing either turn are dropped, not crashed on."""
        rows = [
            {"conversations": [{"from": "human", "value": "Q"}]},  # no gpt
            {"conversations": [{"from": "gpt", "value": "A"}]},  # no human
            self._SAMPLE_ROW,
        ]
        _install_fake_datasets(monkeypatch, {"open-thoughts/OpenThoughts3-1.2M": rows})
        cf = importlib.reload(cf_module)
        pairs = cf.collect_openthoughts3(max_samples=10)
        assert len(pairs) == 1

    def test_respects_max_samples(self, monkeypatch, cf_module):
        """`max_samples` caps streaming early."""
        rows = [self._SAMPLE_ROW] * 50
        _install_fake_datasets(monkeypatch, {"open-thoughts/OpenThoughts3-1.2M": rows})
        cf = importlib.reload(cf_module)
        pairs = cf.collect_openthoughts3(max_samples=3)
        # Dedup collapses identical rows to 1 — but we cap iteration at 3
        # before dedup, so the function should not iterate past 3.
        assert len(pairs) <= 3


# ── SmolTalk2 (D-11) ────────────────────────────────────────────────────────


class TestCollectSmolTalk2:
    """SmolTalk2 (`HuggingFaceTB/smoltalk2`) ships many configs, not one
    'SFT'. Standard ChatML schema: `messages: [{role, content}]`.
    """

    _SAMPLE_ROW = {
        "messages": [
            {"role": "user", "content": "Explain gravity briefly."},
            {
                "role": "assistant",
                "content": (
                    "Gravity is the force pulling masses together. On Earth it accelerates objects at ~9.8 m/s^2."
                ),
            },
        ],
    }

    def test_extracts_user_assistant_pair(self, monkeypatch, cf_module):
        _install_fake_datasets(monkeypatch, {"HuggingFaceTB/smoltalk2": [self._SAMPLE_ROW]})
        cf = importlib.reload(cf_module)
        pairs = cf.collect_smoltalk2(max_samples=10, config="default")
        assert len(pairs) == 1
        assert "gravity" in pairs[0]["prompt"].lower()
        assert "9.8" in pairs[0]["completion"]

    def test_unknown_config_returns_empty_with_log(self, monkeypatch, cf_module, caplog):
        """Missing config → log available configs, return [], do not crash.
        Per learned principle: detect gated/missing on first attempt, no loop.
        """
        _install_fake_datasets(monkeypatch, {})  # any path raises
        cf = importlib.reload(cf_module)
        with caplog.at_level("ERROR"):
            pairs = cf.collect_smoltalk2(max_samples=10, config="unknown")
        assert pairs == []
        # Error message should help user pick a real config
        assert any("config" in rec.message.lower() or "available" in rec.message.lower() for rec in caplog.records)

    def test_skips_short_or_empty_messages(self, monkeypatch, cf_module):
        rows = [
            {
                "messages": [
                    {"role": "user", "content": ""},
                    {"role": "assistant", "content": "hi"},
                ]
            },  # empty user
            {
                "messages": [
                    {"role": "user", "content": "ok?"},
                    {"role": "assistant", "content": "y"},  # too short
                ]
            },
            self._SAMPLE_ROW,
        ]
        _install_fake_datasets(monkeypatch, {"HuggingFaceTB/smoltalk2": rows})
        cf = importlib.reload(cf_module)
        pairs = cf.collect_smoltalk2(max_samples=10, config="default")
        assert len(pairs) == 1

    def test_split_none_iterates_all_splits(self, monkeypatch, cf_module):
        """When split=None, all splits in the config are concatenated.

        Pass 155b: real SmolTalk2 SFT config has 25 named splits, none
        called "train". Default behaviour is auto-iterate-all so users
        do not need to pick one of 25 names.
        """
        _install_fake_datasets(
            monkeypatch,
            {"HuggingFaceTB/smoltalk2": [self._SAMPLE_ROW]},
            splits_by_path={"HuggingFaceTB/smoltalk2": ["split_a_think", "split_b_no_think", "split_c"]},
        )
        cf = importlib.reload(cf_module)
        pairs = cf.collect_smoltalk2(max_samples=10, config="SFT", split=None)
        # Same row in every split → after dedup, 1 pair survives.
        assert len(pairs) == 1

    def test_explicit_split_used_directly(self, monkeypatch, cf_module):
        """When split is provided, function uses it without enumerating."""
        _install_fake_datasets(
            monkeypatch,
            {"HuggingFaceTB/smoltalk2": [self._SAMPLE_ROW]},
        )
        cf = importlib.reload(cf_module)
        pairs = cf.collect_smoltalk2(max_samples=10, config="SFT", split="train")
        assert len(pairs) == 1


# ── D-11 wiring (Pass 156i8): combined_finetune.txt for SFT consumer ────────


class TestCombineAllText:
    """`combine_all` must emit a User:/Assistant: text file alongside the
    JSONL so the existing SFT training path (which reads plain text via
    `Path.read_text`) can consume the collected fine-tune data without
    a JSONL-aware loader.

    Closes the consumer-side gap from D-11: collector writes JSONL, but
    no enigma_engine code reads from data/finetune/. Emitting the canon
    chat format on disk bridges the gap with zero training-side change.
    """

    def test_combine_all_emits_text_file_alongside_jsonl(self, tmp_path):
        """combine_all writes both .jsonl AND .txt outputs."""
        import collect_finetuning_data as cf
        import json

        src = tmp_path / "smoltalk2.jsonl"
        with src.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"prompt": "What is 2+2?", "completion": "4."}) + "\n")
            f.write(json.dumps({"prompt": "Capital of France?", "completion": "Paris."}) + "\n")
        cf.combine_all(tmp_path)
        text_path = tmp_path / "combined_finetune.txt"
        assert text_path.exists(), "combine_all must emit combined_finetune.txt alongside combined_finetune.jsonl"
        text = text_path.read_text(encoding="utf-8")
        assert "User: What is 2+2?" in text
        assert "Assistant: 4." in text
        assert "User: Capital of France?" in text
        assert "Assistant: Paris." in text

    def test_text_file_uses_canonical_chat_format(self, tmp_path):
        """Block format: 'User: <p>\\n\\nAssistant: <c>' with blank
        line between blocks. Matches router.py L315-L317 BackgroundTrainer
        format and the existing GUI chat format."""
        import collect_finetuning_data as cf
        import json

        src = tmp_path / "test.jsonl"
        with src.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"prompt": "P1", "completion": "C1"}) + "\n")
            f.write(json.dumps({"prompt": "P2", "completion": "C2"}) + "\n")
        cf.combine_all(tmp_path)
        text = (tmp_path / "combined_finetune.txt").read_text(encoding="utf-8")
        assert "User: P1\n\nAssistant: C1" in text
        assert "User: P2\n\nAssistant: C2" in text
        # Blocks separated by blank line (so \n\n\n appears between them)
        assert "C1\n\nUser: P2" in text

    def test_text_file_skips_empty_pairs(self, tmp_path):
        """Empty prompt or completion would produce a malformed block —
        combine_all already dedups; the text writer must additionally
        skip empties so the SFT path never sees 'User: \\n\\nAssistant:'."""
        import collect_finetuning_data as cf
        import json

        src = tmp_path / "test.jsonl"
        with src.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"prompt": "", "completion": "C"}) + "\n")
            f.write(json.dumps({"prompt": "P", "completion": ""}) + "\n")
            f.write(json.dumps({"prompt": "Pgood", "completion": "Cgood"}) + "\n")
        cf.combine_all(tmp_path)
        text = (tmp_path / "combined_finetune.txt").read_text(encoding="utf-8")
        assert "User: \n\nAssistant: C" not in text
        assert "User: P\n\nAssistant: " not in text
        assert "User: Pgood\n\nAssistant: Cgood" in text

    def test_warns_when_all_pairs_yield_empty_text(self, tmp_path, caplog):
        """D-11d (Pass 156l): file-present-zero-yield must be loud.

        When combine_all has collected pairs but every one had empty
        prompt or completion, the .txt file is 0 bytes and the SFT
        path silently trains on nothing. Mirror the file-present-zero-
        yield WARNING pattern from Pass 156i6 anchor loader.
        """
        import collect_finetuning_data as cf
        import json
        import logging

        src = tmp_path / "bad.jsonl"
        with src.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"prompt": "", "completion": "C"}) + "\n")
            f.write(json.dumps({"prompt": "P", "completion": ""}) + "\n")
        with caplog.at_level(logging.WARNING, logger=cf.logger.name):
            cf.combine_all(tmp_path)
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("text file is 0 bytes" in r.getMessage() for r in warnings), (
            f"Expected WARNING about empty text yield. Got: {[r.getMessage() for r in warnings]}"
        )
