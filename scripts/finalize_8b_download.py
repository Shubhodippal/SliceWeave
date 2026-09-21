"""Finalize the interrupted Qwen3-8B hub download (2026-09-21).

All five safetensors shards had finished transferring (byte-exact sizes) but
were left named ``<lfs-sha256>.<etag>.incomplete`` and never linked into the
snapshot, so ``from_pretrained`` would re-download ~15.25 GB.  This script:

1. SHA256-verifies every ``.incomplete`` blob against the hub's LFS hash,
2. renames each verified blob to ``blobs/<lfs-sha256>`` (rename-in-place),
3. hard-links them into the pinned snapshot
   ``snapshots/b968826d9c46dd6066d109eabc6255188de91218/<shard>.safetensors``,
4. removes the stale per-shard ``.lock`` files.

Every step is idempotent and refuses to touch anything unless the hash matches.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

CACHE = Path(r"e:\Project\SliceWeave\Qwen3-8B_Results\model_weights")
REPO_DIR = CACHE / "models--Qwen--Qwen3-8B"
BLOBS = REPO_DIR / "blobs"
SNAP = REPO_DIR / "snapshots" / "b968826d9c46dd6066d109eabc6255188de91218"

# shard name -> (LFS sha256, exact size in bytes) as reported by the hub API
SHARDS = {
    "model-00001-of-00005.safetensors": (
        "31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f",
        3996250744,
    ),
    "model-00002-of-00005.safetensors": (
        "5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282",
        3993160032,
    ),
    "model-00003-of-00005.safetensors": (
        "c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836",
        3959604768,
    ),
    "model-00004-of-00005.safetensors": (
        "b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a",
        3187841392,
    ),
    "model-00005-of-00005.safetensors": (
        "20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff",
        1244659840,
    ),
}

CHUNK = 1 << 24


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    SNAP.mkdir(parents=True, exist_ok=True)
    for shard, (hash_, size) in sorted(SHARDS.items()):
        dest_blob = BLOBS / hash_
        snapshot_link = SNAP / shard
        if snapshot_link.exists() and dest_blob.exists():
            print(f"{shard}: already finalized")
            continue
        candidates = list(BLOBS.glob(f"{hash_}.*.incomplete")) or list(
            BLOBS.glob(f"{hash_}.incomplete")
        )
        assert candidates, f"{shard}: no .incomplete blob found for {hash_[:12]}"
        src = candidates[0]
        assert src.stat().st_size == size, (
            f"{shard}: size {src.stat().st_size} != expected {size}; refusing"
        )
        digest = sha256(src)
        assert digest == hash_, (
            f"{shard}: sha256 {digest} != expected {hash_}; refusing to finalize"
        )
        if not dest_blob.exists():
            os.rename(src, dest_blob)  # same volume: instant, no data copy
        try:
            os.link(dest_blob, snapshot_link)  # hard link; fall back to copy
        except OSError:
            import shutil

            shutil.copy2(dest_blob, snapshot_link)
        lock = CACHE / ".locks" / "models--Qwen--Qwen3-8B" / f"{hash_}.lock"
        if lock.exists():
            lock.unlink()
        print(f"{shard}: finalized ({size} bytes, hash verified)")


if __name__ == "__main__":
    main()
