"""TC-9/TC-10: Tests for curated_dataset.py and dataset.py — JSONL workflow, text cleaning, corpus processing."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from enigma_engine.core.curated_dataset import CuratedDataset, DatasetEntry
from enigma_engine.core.dataset import (
    KNOWN_DATASETS,
    clean_text,
    download_dataset,
    estimate_token_count,
    iter_text_chunks,
    process_text_corpus,
)


# ================================================================
# DatasetEntry
# ================================================================


class TestDatasetEntry:
    """Test DatasetEntry dataclass."""

    def test_defaults(self):
        e = DatasetEntry()
        assert e.text == ""
        assert e.status == "pending"
        assert e.timestamp != ""  # auto-set by __post_init__

    def test_explicit_timestamp_preserved(self):
        e = DatasetEntry(timestamp="2026-01-01T00:00:00")
        assert e.timestamp == "2026-01-01T00:00:00"

    def test_to_dict(self):
        e = DatasetEntry(text="hello", source="test")
        d = e.to_dict()
        assert d["text"] == "hello"
        assert d["source"] == "test"
        assert "timestamp" in d

    def test_from_dict(self):
        data = {
            "text": "hello",
            "source": "test",
            "status": "approved",
            "timestamp": "2026-01-01",
            "stage": "basics",
            "metadata": {},
        }
        e = DatasetEntry.from_dict(data)
        assert e.text == "hello"
        assert e.status == "approved"

    def test_from_dict_ignores_unknown_keys(self):
        data = {"text": "hi", "unknown_field": 42}
        e = DatasetEntry.from_dict(data)
        assert e.text == "hi"

    def test_roundtrip(self):
        e = DatasetEntry(text="test", source="chat", stage="basics")
        d = e.to_dict()
        e2 = DatasetEntry.from_dict(d)
        assert e2.text == e.text
        assert e2.source == e.source
        assert e2.stage == e.stage


# ================================================================
# CuratedDataset — Add / Query
# ================================================================


class TestCuratedDatasetAdd:
    """Test adding entries."""

    def test_add_returns_index(self, tmp_path):
        ds = CuratedDataset(tmp_path / "ds.jsonl")
        idx = ds.add("Hello world", source="test")
        assert idx == 0

    def test_add_empty_returns_minus_one(self, tmp_path):
        ds = CuratedDataset(tmp_path / "ds.jsonl")
        assert ds.add("") == -1
        assert ds.add("   ") == -1
        assert ds.count == 0

    def test_add_multiple(self, tmp_path):
        ds = CuratedDataset(tmp_path / "ds.jsonl")
        ds.add("first", source="a")
        ds.add("second", source="b")
        assert ds.count == 2

    def test_add_batch_deduplicates(self, tmp_path):
        ds = CuratedDataset(tmp_path / "ds.jsonl")
        count = ds.add_batch(["a", "b", "a", "c"])
        assert count == 3  # "a" deduplicated

    def test_add_batch_deduplicates_against_existing(self, tmp_path):
        ds = CuratedDataset(tmp_path / "ds.jsonl")
        ds.add("existing")
        count = ds.add_batch(["existing", "new"])
        assert count == 1  # only "new" added

    def test_add_with_metadata(self, tmp_path):
        ds = CuratedDataset(tmp_path / "ds.jsonl")
        ds.add("text", metadata={"key": "val"})
        assert ds.entries[0].metadata == {"key": "val"}


# ================================================================
# CuratedDataset — Approval Workflow
# ================================================================


class TestApprovalWorkflow:
    """Test approve/reject/remove."""

    @pytest.fixture()
    def ds(self, tmp_path):
        d = CuratedDataset(tmp_path / "ds.jsonl")
        d.add("entry0", source="test")
        d.add("entry1", source="test")
        d.add("entry2", source="test")
        return d

    def test_approve(self, ds):
        assert ds.approve(0)
        assert ds.entries[0].status == "approved"

    def test_reject(self, ds):
        assert ds.reject(1)
        assert ds.entries[1].status == "rejected"

    def test_approve_invalid_index(self, ds):
        assert not ds.approve(99)
        assert not ds.approve(-1)

    def test_remove(self, ds):
        assert ds.remove(1)
        assert ds.count == 2
        assert ds.entries[0].text == "entry0"
        assert ds.entries[1].text == "entry2"

    def test_remove_invalid_index(self, ds):
        assert not ds.remove(99)

    def test_approve_all_pending(self, ds):
        ds.approve(0)  # already approve one
        count = ds.approve_all_pending()
        assert count == 2  # the other two
        assert ds.approved_count == 3

    def test_reject_all_pending(self, ds):
        count = ds.reject_all_pending()
        assert count == 3
        assert ds.rejected_count == 3
        assert ds.pending_count == 0


# ================================================================
# CuratedDataset — Queries
# ================================================================


class TestCuratedDatasetQueries:
    """Test query methods."""

    @pytest.fixture()
    def ds(self, tmp_path):
        d = CuratedDataset(tmp_path / "ds.jsonl")
        d.add("a", source="guided", stage="basics", status="approved")
        d.add("b", source="chat", stage="basics", status="pending")
        d.add("c", source="guided", stage="commands", status="rejected")
        d.add("d", source="chat", stage="commands", status="approved")
        return d

    def test_get_approved(self, ds):
        approved = ds.get_approved()
        assert len(approved) == 2
        texts = {e.text for e in approved}
        assert texts == {"a", "d"}

    def test_get_pending(self, ds):
        assert len(ds.get_pending()) == 1

    def test_get_by_source(self, ds):
        guided = ds.get_by_source("guided")
        assert len(guided) == 2

    def test_get_by_stage(self, ds):
        commands = ds.get_by_stage("commands")
        assert len(commands) == 2

    def test_get_approved_text(self, ds):
        texts = ds.get_approved_text()
        assert texts == ["a", "d"]

    def test_get_training_data(self, ds):
        data = ds.get_training_data()
        assert "a" in data
        assert "d" in data
        assert "b" not in data  # pending

    def test_get_training_data_by_stage(self, ds):
        data = ds.get_training_data(stage="commands")
        assert "d" in data
        assert "a" not in data  # different stage

    def test_count_properties(self, ds):
        assert ds.count == 4
        assert ds.approved_count == 2
        assert ds.pending_count == 1
        assert ds.rejected_count == 1


# ================================================================
# CuratedDataset — Persistence (JSONL)
# ================================================================


class TestCuratedDatasetPersistence:
    """Test save/load JSONL roundtrip."""

    def test_save_load_roundtrip(self, tmp_path):
        path = tmp_path / "ds.jsonl"
        ds1 = CuratedDataset(path)
        ds1.add("hello", source="test", stage="basics")
        ds1.approve(0)
        ds1.add("world", source="chat")
        ds1.save()

        ds2 = CuratedDataset(path)
        assert ds2.count == 2
        assert ds2.entries[0].text == "hello"
        assert ds2.entries[0].status == "approved"
        assert ds2.entries[1].text == "world"
        assert ds2.entries[1].status == "pending"

    def test_save_creates_valid_jsonl(self, tmp_path):
        path = tmp_path / "ds.jsonl"
        ds = CuratedDataset(path)
        ds.add("line1")
        ds.add("line2")
        ds.save()

        # Read raw JSONL and verify each line is valid JSON
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert "text" in obj
            assert "status" in obj

    def test_load_nonexistent_file(self, tmp_path):
        ds = CuratedDataset(tmp_path / "nonexistent.jsonl")
        assert ds.count == 0

    def test_load_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        ds = CuratedDataset(path)
        assert ds.count == 0

    def test_load_handles_malformed_lines(self, tmp_path):
        """One corrupt line is skipped-and-counted; BOTH good lines survive.

        The old version asserted `count >= 0` -- a tautology that stayed
        green while describing a fail-fast design the implementation never
        had (it does per-line skip, curated_dataset._load_entries). A
        regression that loads 0 entries from a file with 2 good lines --
        total data loss, then save() clobbers the file -- must fail here
        (test-suite audit 2026-07-17).
        """
        path = tmp_path / "bad.jsonl"
        path.write_text(
            '{"text": "good", "status": "pending"}\nnot valid json\n{"text": "also good", "status": "approved"}\n'
        )
        ds = CuratedDataset(path)
        assert ds.count == 2
        texts = {e.text for e in ds.entries}
        assert texts == {"good", "also good"}


# ================================================================
# CuratedDataset — Thread Safety
# ================================================================


class TestCuratedDatasetThreadSafety:
    """Test that add_batch holds lock for entire operation (S669 fix)."""

    def test_concurrent_add_batch_no_duplicates(self, tmp_path):
        """Concurrent add_batch calls must not produce duplicate entries."""
        import threading

        ds = CuratedDataset(tmp_path / "ds.jsonl")
        barrier = threading.Barrier(4)

        def add_shared():
            barrier.wait()
            ds.add_batch(["shared_a", "shared_b", "shared_c"])

        threads = [threading.Thread(target=add_shared) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        texts = [e.text for e in ds.entries]
        assert len(texts) == len(set(texts)), f"Duplicates found: {[t for t in texts if texts.count(t) > 1]}"
        assert ds.count == 3  # only 3 unique texts


# ================================================================
# TC-10: clean_text (from test_dataset.py)
# ================================================================


class TestCleanText:
    """Test text cleaning for pre-training."""

    def test_removes_null_bytes(self):
        assert "\x00" not in clean_text("hello\x00world")
        assert clean_text("hello\x00world") == "helloworld"

    def test_removes_control_chars(self):
        result = clean_text("hello\x01\x02\x03world")
        assert result == "helloworld"

    def test_preserves_newline_tab(self):
        result = clean_text("hello\n\tworld")
        assert "\n" in result
        assert "\t" in result

    def test_normalizes_crlf(self):
        assert "\r" not in clean_text("hello\r\nworld")
        assert clean_text("hello\r\nworld") == "hello\nworld"

    def test_normalizes_bare_cr(self):
        assert clean_text("a\rb") == "a\nb"

    def test_strips_trailing_whitespace(self):
        result = clean_text("line1   \nline2  \n")
        assert result == "line1\nline2"

    def test_collapses_excessive_newlines(self):
        result = clean_text("a\n\n\n\n\nb")
        assert result == "a\n\nb"

    def test_two_newlines_preserved(self):
        result = clean_text("a\n\nb")
        assert result == "a\n\nb"

    def test_strips_outer_whitespace(self):
        result = clean_text("  hello  ")
        assert result == "hello"

    def test_empty_string(self):
        assert clean_text("") == ""


# ================================================================
# TC-10: estimate_token_count
# ================================================================


class TestEstimateTokenCount:
    """Test token count estimation."""

    def test_empty_string(self):
        assert estimate_token_count("") == 0

    def test_short_string(self):
        # "hi" = 2 chars / 4.0 = 0.5 → max(1, 0) = 1
        assert estimate_token_count("hi") >= 1

    def test_known_length(self):
        text = "a" * 400  # 400 chars / 4.0 = 100 tokens
        assert estimate_token_count(text) == 100

    def test_custom_chars_per_token(self):
        text = "a" * 200  # 200 chars / 2.0 = 100
        assert estimate_token_count(text, chars_per_token=2.0) == 100


# ================================================================
# TC-10: process_text_corpus
# ================================================================


class TestProcessTextCorpus:
    """Test loading and cleaning text from files."""

    def test_txt_file(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("Hello world\nThis is a test\n", encoding="utf-8")
        result = process_text_corpus(f)
        assert "Hello world" in result
        assert "This is a test" in result

    def test_txt_file_cleaned(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("Hello\x00world\r\ntest  \n", encoding="utf-8")
        result = process_text_corpus(f)
        assert "\x00" not in result
        assert "\r" not in result

    def test_jsonl_file(self, tmp_path):
        f = tmp_path / "data.jsonl"
        lines = [
            json.dumps({"text": "line one"}),
            json.dumps({"text": "line two"}),
        ]
        f.write_text("\n".join(lines), encoding="utf-8")
        result = process_text_corpus(f)
        assert "line one" in result
        assert "line two" in result

    def test_jsonl_custom_key(self, tmp_path):
        f = tmp_path / "data.jsonl"
        lines = [json.dumps({"content": "hello"})]
        f.write_text("\n".join(lines), encoding="utf-8")
        result = process_text_corpus(f, text_key="content")
        assert "hello" in result

    def test_jsonl_skips_malformed(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text(
            '{"text": "good"}\nnot json\n{"text": "also good"}\n',
            encoding="utf-8",
        )
        result = process_text_corpus(f)
        assert "good" in result

    def test_json_array(self, tmp_path):
        f = tmp_path / "data.json"
        data = [{"text": "first"}, {"text": "second"}]
        f.write_text(json.dumps(data), encoding="utf-8")
        result = process_text_corpus(f)
        assert "first" in result
        assert "second" in result

    def test_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
        (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
        result = process_text_corpus(tmp_path)
        assert "alpha" in result
        assert "beta" in result

    def test_directory_ignores_non_text(self, tmp_path):
        (tmp_path / "data.txt").write_text("hello", encoding="utf-8")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        result = process_text_corpus(tmp_path)
        assert "hello" in result

    def test_nonexistent_path(self, tmp_path):
        result = process_text_corpus(tmp_path / "nope.txt")
        assert result == ""

    def test_directory_recursive(self, tmp_path):
        """S788: process_text_corpus should recurse into subdirectories."""
        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "top.txt").write_text("top level", encoding="utf-8")
        (sub / "nested.txt").write_text("nested file", encoding="utf-8")
        result = process_text_corpus(tmp_path)
        assert "top level" in result
        assert "nested file" in result, "_process_directory should recurse into subdirectories"

    def test_string_path(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("hello", encoding="utf-8")
        result = process_text_corpus(str(f))
        assert "hello" in result


# ================================================================
# TC-10: KNOWN_DATASETS registry
# ================================================================


class TestKnownDatasets:
    """Test dataset registry structure."""

    def test_tinystories_exists(self):
        assert "tinystories" in KNOWN_DATASETS

    def test_required_fields(self):
        for name, info in KNOWN_DATASETS.items():
            assert "name" in info, f"{name} missing 'name'"
            assert "repo_id" in info, f"{name} missing 'repo_id'"
            assert "description" in info, f"{name} missing 'description'"


# ================================================================
# TC-10: download_dataset — validation only (no actual downloads)
# ================================================================


class TestDownloadDataset:
    """Test download_dataset validation (mocked network)."""

    def test_unknown_dataset_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown dataset"):
            download_dataset("nonexistent", tmp_path)

    def test_creates_dest_dir(self, tmp_path):
        dest = tmp_path / "sub" / "dir"
        with (
            patch("huggingface_hub.snapshot_download", side_effect=RuntimeError("mocked")),
            pytest.raises(RuntimeError),
        ):
            download_dataset("tinystories", dest)
        # dest dir should exist even if download fails
        assert dest.exists()

    def test_progress_fn_called(self, tmp_path):
        msgs = []
        with patch("huggingface_hub.snapshot_download", return_value=str(tmp_path / "tinystories")):
            download_dataset("tinystories", tmp_path, progress_fn=msgs.append)
        assert len(msgs) >= 1
        assert "TinyStories" in msgs[0]


# ================================================================
# TC-10: iter_text_chunks — streaming generator
# ================================================================


class TestIterTextChunks:
    """Test streaming text chunk generator."""

    def test_yields_from_txt_file(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("Hello world\nThis is a test\n", encoding="utf-8")
        chunks = list(iter_text_chunks(f))
        assert len(chunks) >= 1
        combined = "\n\n".join(chunks)
        assert "Hello world" in combined

    def test_returns_generator(self, tmp_path):
        """iter_text_chunks must return a generator, not a list."""
        f = tmp_path / "data.txt"
        f.write_text("Hello", encoding="utf-8")
        result = iter_text_chunks(f)
        import types

        assert isinstance(result, types.GeneratorType)

    def test_yields_from_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
        (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
        chunks = list(iter_text_chunks(tmp_path))
        combined = "\n\n".join(chunks)
        assert "alpha" in combined
        assert "beta" in combined

    def test_recurses_subdirectories(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "deep.txt").write_text("deep text", encoding="utf-8")
        (tmp_path / "top.txt").write_text("top text", encoding="utf-8")
        chunks = list(iter_text_chunks(tmp_path))
        combined = "\n\n".join(chunks)
        assert "deep text" in combined
        assert "top text" in combined

    def test_empty_file_yields_nothing(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        chunks = list(iter_text_chunks(f))
        assert chunks == []

    def test_nonexistent_path_yields_nothing(self, tmp_path):
        chunks = list(iter_text_chunks(tmp_path / "nope.txt"))
        assert chunks == []

    def test_jsonl_file(self, tmp_path):
        f = tmp_path / "data.jsonl"
        lines = [
            json.dumps({"text": "line one"}),
            json.dumps({"text": "line two"}),
        ]
        f.write_text("\n".join(lines), encoding="utf-8")
        chunks = list(iter_text_chunks(f))
        assert len(chunks) >= 1
        combined = "\n\n".join(chunks)
        assert "line one" in combined

    def test_on_progress_called(self, tmp_path):
        """The callback must actually fire on the chunked-read path.

        Small files never reach _chunked_read_text (>500 MB gate), so the
        old version's `isinstance(calls, list)` on its own list could not
        fail -- iter_text_chunks silently dropping on_progress stayed green
        (test-suite audit 2026-07-17). Drive the chunked reader directly:
        one small file = one chunk = one callback at 100%.
        """
        from enigma_engine.core.dataset import _chunked_read_text

        f = tmp_path / "data.txt"
        # newline-terminated: progress fires per completed-line chunk (text
        # with no newline at all rides the remainder path, no callback)
        f.write_text("Hello world test data\nsecond line\n", encoding="utf-8")
        calls = []
        chunks = _chunked_read_text(f, on_progress=lambda p, m: calls.append((p, m)))
        assert "Hello world" in "".join(chunks)
        assert calls, "on_progress never fired on the chunked-read path"
        pct, msg = calls[-1]
        assert pct == 100
        assert "data.txt" in msg
        # and the small-file fast path still tolerates the kwarg silently
        small_calls = []
        list(iter_text_chunks(f, on_progress=lambda p, m: small_calls.append(p)))
        assert small_calls == []  # below the streaming threshold: no progress

    def test_string_path(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("hello", encoding="utf-8")
        chunks = list(iter_text_chunks(str(f)))
        assert len(chunks) == 1
        assert "hello" in chunks[0]

    def test_skips_non_text_files_in_dir(self, tmp_path):
        (tmp_path / "data.txt").write_text("hello", encoding="utf-8")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        chunks = list(iter_text_chunks(tmp_path))
        combined = "\n\n".join(chunks)
        assert "hello" in combined
        assert len(chunks) == 1
