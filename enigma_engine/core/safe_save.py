"""
Atomic save helpers for model weights and data files.

Writes to a temporary file first, then atomically replaces
the target. This prevents corruption if the process is killed,
the system loses power, or the user closes the window mid-save.

Usage:
    from enigma_engine.core.safe_save import atomic_torch_save, atomic_safetensors_save
    atomic_torch_save({"model_state_dict": sd, "config": cfg}, path)
    atomic_safetensors_save(state_dict, path)    # preferred — no pickle

    from enigma_engine.core.safe_save import atomic_write_text, atomic_write_json
    atomic_write_text(path, content)             # plain text files
    atomic_write_json(path, data)                # JSON files
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def refuse_existing_artifact(out_dir: str | Path) -> None:
    """Refuse to aim a trainer at a dir that already holds a checkpoint.

    Model artifacts are VERSIONED, never rebuilt in place (pretokenize's
    corpus rule, applied to the checkpoint writers 2026-08-13 after the DPO
    trainer's --out default was found pointing at the v8 ROLLBACK dir). All
    three rotation names count: an interrupted run leaves latest.pth or
    prev.pth with no model.pth, and those are receipts too. Callers decide
    the sanctioned exceptions (--resume, --sanity) BEFORE calling.
    """
    out_dir = Path(out_dir)
    # --out naming a FILE is the model.pth-typo class: <file>/model.pth
    # never exists, so the dir check alone passed and a DPO run trained to
    # completion before dying at the post-training mkdir (audit 2026-08-13).
    # A .pth name is the same typo even when nothing exists there yet.
    if out_dir.is_file() or out_dir.suffix == ".pth":
        raise SystemExit(
            f"REFUSED: --out {out_dir} names a checkpoint FILE, not a run "
            f"directory. Pass a NEW run dir instead."
        )
    for name in ("model.pth", "latest.pth", "prev.pth"):
        target = out_dir / name
        if target.exists():
            raise SystemExit(
                f"REFUSED: {target} already exists -- model artifacts are "
                f"versioned, never rebuilt in place. Name a NEW --out dir, "
                f"or delete the old artifact deliberately first."
            )


def refuse_existing_patterns(out_dir, patterns) -> None:
    """refuse_existing_artifact for writers whose artifact names are
    run-derived (the encoder aligns write <stem>_<modality>_best.pt and
    friends, never model.pth). Same contract: versioned, never rebuilt
    in place; callers decide sanctioned exceptions BEFORE calling."""
    out_dir = Path(out_dir)
    if out_dir.is_file() or out_dir.suffix == ".pth" or out_dir.suffix == ".pt":
        raise SystemExit(f"--out {out_dir} names a file; pass a directory")
    if not out_dir.is_dir():
        return
    for pat in patterns:
        for hit in sorted(out_dir.glob(pat)):
            raise SystemExit(
                f"{out_dir} already holds {hit.name} -- model artifacts are "
                "versioned, never rebuilt in place. Pass a NEW --out."
            )


def atomic_torch_save(data: dict, path: str | Path, rotate_to: str | Path | None = None) -> None:
    """Save a PyTorch checkpoint atomically.

    Writes to ``<path>.tmp`` first, then replaces the target via
    ``os.replace`` (atomic on both Windows NTFS and Linux ext4).
    On failure BEFORE rotation the temp file is cleaned up and the
    original is left untouched.

    When *rotate_to* is given and *path* already exists, the current
    file is renamed to *rotate_to* after the new data is fully written
    and immediately before promotion — keeping one previous generation
    as a fallback. A failure AFTER rotation (the crash window between
    the two renames, or a promotion that raises) leaves *rotate_to*
    (old data) plus the complete ``.tmp`` — nothing at *path* itself;
    recovery is renaming the ``.tmp`` into place by hand.
    """
    import torch

    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    rotated = False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # fsync before the renames: without it a power loss can commit the
        # rename metadata ahead of the data blocks, leaving a complete-looking
        # but truncated file (process kills were already safe; power was not)
        with open(tmp_path, "wb") as f:
            torch.save(data, f)
            f.flush()
            os.fsync(f.fileno())
        if rotate_to is not None and path.exists():
            os.replace(path, Path(rotate_to))
            rotated = True
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up the partial temp file -- UNLESS the rotation already
        # happened, at which point the .tmp is the ONLY complete copy of the
        # new data and deleting it would leave nothing at the target at all
        # (review 2026-08-13; the docstring's crash-window promise now holds
        # for exceptions too, not just hard kills).
        if not rotated:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def atomic_safetensors_save(
    tensors: dict[str, "torch.Tensor"],  # noqa: F821
    path: str | Path,
    metadata: dict[str, str] | None = None,
) -> None:
    """Save tensors in safetensors format atomically.

    Preferred over ``atomic_torch_save`` for pure state dicts —
    safetensors uses no pickle so loading cannot execute code.

    Args:
        tensors: Flat dict of ``{name: Tensor}``.
        path: Target ``.safetensors`` file.
        metadata: Optional string-to-string metadata header.
    """
    from safetensors.torch import save_file

    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(tmp_path), metadata=metadata)
        # fsync before the rename (see atomic_torch_save for the
        # power-loss rationale). save_file exposes no file handle, so
        # reopen the completed temp file to force its data to disk.
        with open(tmp_path, "rb+") as f:
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_text(path: str | Path, content: str, backup: bool = True) -> None:
    """Write a text file atomically with fsync and backup rotation.

    1. Backs up the existing file to ``<path>.bak`` (if it exists and
       *backup* is True).
    2. Writes *content* to ``<path>.tmp`` with ``fsync`` for durability.
    3. Atomically replaces the target via ``os.replace``.

    On failure the temp file is cleaned up and the original is
    left untouched.

    Args:
        path: Target file path.
        content: Text content to write.
        backup: Rotate the previous version to ``.bak``. Pass False for
            stores with DELETION semantics (the memory store): keeping a
            pre-delete copy would silently defeat the delete.
    """
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        # Preserve original permissions when the file already exists
        original_mode: int | None = None
        if path.exists():
            try:
                original_mode = path.stat().st_mode
            except OSError:
                pass

            # Backup existing file before overwriting
            if backup:
                bak_path = path.with_suffix(path.suffix + ".bak")
                try:
                    import shutil

                    shutil.copy2(str(path), str(bak_path))
                except OSError as exc:
                    logger.warning("Backup copy failed for %s: %s", path, exc)

        # Write to temp file with fsync for durability
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())

        # Atomic rename
        os.replace(tmp_path, path)

        # Restore original permissions (S551)
        if original_mode is not None:
            try:
                os.chmod(path, original_mode)
            except OSError:
                pass  # Best-effort on Windows
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_json(
    path: str | Path,
    data: object,
    indent: int = 2,
    ensure_ascii: bool = False,
    default: object = None,
) -> None:
    """Write a JSON file atomically with fsync and backup rotation.

    Convenience wrapper around :func:`atomic_write_text` that handles
    ``json.dumps`` serialization.

    Args:
        path: Target file path.
        data: JSON-serializable data.
        indent: JSON indentation (default 2).
        ensure_ascii: Whether to escape non-ASCII (default False).
        default: Fallback serializer for non-standard types (e.g., ``str``).
    """
    import json

    content = json.dumps(
        data,
        indent=indent,
        ensure_ascii=ensure_ascii,
        default=default,
    )
    atomic_write_text(path, content)


def unlink_with_backup(path: str | Path) -> None:
    """Delete *path* and its ``<path>.bak`` sibling together.

    :func:`atomic_write_text` and :func:`atomic_write_json` rotate the
    previous file contents to ``<path>.bak`` on every write. When the
    caller later deletes ``<path>`` directly (``Path.unlink``), the
    ``.bak`` sibling is orphaned and lingers on disk forever — the user
    sees the artifact disappear from the UI but the file is still there.

    This helper is the safe deletion counterpart: removes both the
    target and its backup in one call. Both removals are
    ``missing_ok=True`` — the helper is idempotent and safe to call
    even if only one of the two exists. Directories are not handled
    here; callers deleting directories should use ``shutil.rmtree``.

    Args:
        path: Target file path (file, not directory).

    Raises:
        OSError: If unlinking the primary path fails for a reason other
            than ``FileNotFoundError``. ``.bak`` failures are logged but
            never re-raised — losing a backup never blocks deletion.
    """
    p = Path(path)
    bak = p.with_suffix(p.suffix + ".bak")
    # Primary first — its failure is fatal to the caller.
    p.unlink(missing_ok=True)
    # Backup second — best-effort, never blocks the delete contract.
    try:
        bak.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Backup cleanup failed for %s: %s", bak, exc)
