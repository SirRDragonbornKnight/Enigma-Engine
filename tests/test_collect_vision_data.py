"""Tests for collect_vision_data.py vision-data collectors (V-5).

Pass 156c: LLaVA-Pretrain fetcher. We mock the `datasets` library via fake
module injection so tests do not require HuggingFace network access. See AA
learned principle:
"Optional dependency tests: inject fake module via types.ModuleType
+ monkeypatch.setitem(sys.modules, ...)".

Fake images-dir is a tmp_path with empty .jpg files — the fetcher must only
verify the file exists (it does not read pixel bytes).
"""

import importlib
import json
import sys
import types
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Fake datasets injection ─────────────────────────────────────────────────


def _install_fake_datasets(monkeypatch, rows_by_path):
    """Inject a fake `datasets` module whose `load_dataset(path, ...)`
    returns the rows registered for that dataset path.

    Streaming and non-streaming both yield the same row list.
    Unknown paths raise ValueError with the available-choices message
    so the fetcher's first-attempt detection fires (per Pass 155 lesson).
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
            raise ValueError(f"Repo '{path}' not found. Available: ['liuhaotian/LLaVA-Pretrain']")
        return _FakeDataset(rows_by_path[path])

    fake.load_dataset = _load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake)


@pytest.fixture
def cv_module(monkeypatch):
    """Reload collect_vision_data after fake `datasets` is installed."""
    if "collect_vision_data" in sys.modules:
        monkeypatch.delitem(sys.modules, "collect_vision_data")
    return importlib.import_module("collect_vision_data")


def _make_image_files(images_dir: Path, names: list[str]) -> None:
    """Create empty stub .jpg files at the given relative paths."""
    for n in names:
        p = images_dir / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")


# ── LLaVA-Pretrain (V-5) ────────────────────────────────────────────────────


class TestCollectLLaVAPretrain:
    """`liuhaotian/LLaVA-Pretrain` row schema (HF dataset card):

        {
          "id": "00453_004534029",
          "image": "00453/004534029.jpg",
          "conversations": [
            {"from": "human", "value": "<image>\\nWhat does this depict?"},
            {"from": "gpt", "value": "A photo of a cat sitting on a mat."}
          ]
        }

    Caption text = the gpt turn's value. The image filename is relative
    to an `images.zip` archive that the user extracts manually and points
    to via `--images-dir`. The fetcher does not download images itself.
    """

    _SAMPLE_ROWS = [
        {
            "id": "00000_000000001",
            "image": "00000/000000001.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\nDescribe."},
                {"from": "gpt", "value": "A photo of a cat sitting on a mat."},
            ],
        },
        {
            "id": "00000_000000002",
            "image": "00000/000000002.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\nWhat is this?"},
                {"from": "gpt", "value": "A red car parked beside a tree."},
            ],
        },
        {
            "id": "00000_000000003",
            "image": "00000/000000003.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\nCaption please."},
                {"from": "gpt", "value": "A close-up of a yellow flower."},
            ],
        },
    ]

    def test_writes_jsonl_with_absolute_image_paths(
        self,
        monkeypatch,
        cv_module,
        tmp_path,
    ):
        """All present images → JSONL with abs-path image + gpt-turn text."""
        images_dir = tmp_path / "llava_images"
        _make_image_files(
            images_dir,
            ["00000/000000001.jpg", "00000/000000002.jpg", "00000/000000003.jpg"],
        )
        _install_fake_datasets(
            monkeypatch,
            {"liuhaotian/LLaVA-Pretrain": list(self._SAMPLE_ROWS)},
        )
        # Reload after datasets is installed
        if "collect_vision_data" in sys.modules:
            monkeypatch.delitem(sys.modules, "collect_vision_data")
        cv = importlib.import_module("collect_vision_data")

        # RELATIVE images_dir on purpose: with an absolute tmp_path the
        # is_absolute() assert below passed even without the collector's
        # .resolve() -- a production `--images-dir data/vision` would then
        # emit cwd-relative paths (test-suite audit 2026-07-17).
        monkeypatch.chdir(tmp_path)
        pairs = cv.collect_llava_pretrain(
            max_samples=10,
            images_dir=Path("llava_images"),
        )

        assert len(pairs) == 3
        for p in pairs:
            assert "image" in p and "text" in p
            # Absolute path so trainer/scanner can open without cwd tricks
            assert Path(p["image"]).is_absolute()
            assert Path(p["image"]).exists()
        # Text must be the gpt turn (not the human turn)
        captions = {p["text"] for p in pairs}
        assert "A photo of a cat sitting on a mat." in captions
        assert "A red car parked beside a tree." in captions
        assert "A close-up of a yellow flower." in captions

    def test_skips_missing_image_with_warning(
        self,
        monkeypatch,
        cv_module,
        tmp_path,
        caplog,
    ):
        """Missing-on-disk images are skipped; one warning per skip."""
        images_dir = tmp_path / "llava_images"
        # Only image #1 and #3 exist; #2 missing.
        _make_image_files(
            images_dir,
            ["00000/000000001.jpg", "00000/000000003.jpg"],
        )
        _install_fake_datasets(
            monkeypatch,
            {"liuhaotian/LLaVA-Pretrain": list(self._SAMPLE_ROWS)},
        )
        if "collect_vision_data" in sys.modules:
            monkeypatch.delitem(sys.modules, "collect_vision_data")
        cv = importlib.import_module("collect_vision_data")

        import logging

        with caplog.at_level(logging.WARNING, logger=cv.logger.name):
            pairs = cv.collect_llava_pretrain(
                max_samples=10,
                images_dir=images_dir,
            )

        assert len(pairs) == 2
        assert any("000000002.jpg" in r.message for r in caplog.records)

    def test_caps_at_max_samples(
        self,
        monkeypatch,
        cv_module,
        tmp_path,
    ):
        """max_samples bounds the number of rows iterated."""
        images_dir = tmp_path / "llava_images"
        _make_image_files(
            images_dir,
            ["00000/000000001.jpg", "00000/000000002.jpg", "00000/000000003.jpg"],
        )
        _install_fake_datasets(
            monkeypatch,
            {"liuhaotian/LLaVA-Pretrain": list(self._SAMPLE_ROWS)},
        )
        if "collect_vision_data" in sys.modules:
            monkeypatch.delitem(sys.modules, "collect_vision_data")
        cv = importlib.import_module("collect_vision_data")

        pairs = cv.collect_llava_pretrain(
            max_samples=2,
            images_dir=images_dir,
        )

        assert len(pairs) == 2

    def test_dedups_identical_image_caption_pairs(
        self,
        monkeypatch,
        cv_module,
        tmp_path,
    ):
        """Two rows with same image+caption collapse to one."""
        images_dir = tmp_path / "llava_images"
        _make_image_files(images_dir, ["00000/000000001.jpg"])
        dup_row = self._SAMPLE_ROWS[0]
        _install_fake_datasets(
            monkeypatch,
            {"liuhaotian/LLaVA-Pretrain": [dup_row, dup_row, dup_row]},
        )
        if "collect_vision_data" in sys.modules:
            monkeypatch.delitem(sys.modules, "collect_vision_data")
        cv = importlib.import_module("collect_vision_data")

        pairs = cv.collect_llava_pretrain(
            max_samples=10,
            images_dir=images_dir,
        )

        assert len(pairs) == 1

    def test_skips_rows_missing_gpt_turn(
        self,
        monkeypatch,
        cv_module,
        tmp_path,
    ):
        """Rows without a gpt turn are silently skipped (malformed data)."""
        images_dir = tmp_path / "llava_images"
        _make_image_files(images_dir, ["00000/000000001.jpg"])
        bad_row = {
            "id": "bad",
            "image": "00000/000000001.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\nQ?"},
                # No gpt turn.
            ],
        }
        _install_fake_datasets(
            monkeypatch,
            {"liuhaotian/LLaVA-Pretrain": [bad_row]},
        )
        if "collect_vision_data" in sys.modules:
            monkeypatch.delitem(sys.modules, "collect_vision_data")
        cv = importlib.import_module("collect_vision_data")

        pairs = cv.collect_llava_pretrain(
            max_samples=10,
            images_dir=images_dir,
        )

        assert pairs == []

    def test_unknown_repo_returns_empty_and_logs(
        self,
        monkeypatch,
        cv_module,
        tmp_path,
        caplog,
    ):
        """Auth/availability failures → return [] (per Pass 155 lesson)."""
        images_dir = tmp_path / "llava_images"
        images_dir.mkdir()
        _install_fake_datasets(monkeypatch, {})  # repo not registered

        if "collect_vision_data" in sys.modules:
            monkeypatch.delitem(sys.modules, "collect_vision_data")
        cv = importlib.import_module("collect_vision_data")

        import logging

        with caplog.at_level(logging.ERROR, logger=cv.logger.name):
            pairs = cv.collect_llava_pretrain(
                max_samples=10,
                images_dir=images_dir,
            )

        assert pairs == []
        assert any("LLaVA-Pretrain" in r.message for r in caplog.records)

    def test_missing_images_dir_raises(
        self,
        monkeypatch,
        cv_module,
        tmp_path,
    ):
        """A non-existent images-dir is a hard error — fail fast, don't
        silently produce a JSONL referencing files that aren't there.
        """
        _install_fake_datasets(
            monkeypatch,
            {"liuhaotian/LLaVA-Pretrain": list(self._SAMPLE_ROWS)},
        )
        if "collect_vision_data" in sys.modules:
            monkeypatch.delitem(sys.modules, "collect_vision_data")
        cv = importlib.import_module("collect_vision_data")

        with pytest.raises(FileNotFoundError):
            cv.collect_llava_pretrain(
                max_samples=10,
                images_dir=tmp_path / "does_not_exist",
            )


class TestWriteJsonl:
    """Round-trip the JSONL writer used by the CLI driver."""

    def test_writes_one_record_per_line(self, cv_module, tmp_path):
        out = tmp_path / "out.jsonl"
        cv_module._write_jsonl(
            [
                {"image": "/abs/a.jpg", "text": "alpha"},
                {"image": "/abs/b.jpg", "text": "bravo"},
            ],
            out,
        )
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["text"] == "alpha"
        assert json.loads(lines[1])["image"] == "/abs/b.jpg"
