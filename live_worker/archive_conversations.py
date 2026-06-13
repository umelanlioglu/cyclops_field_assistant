#!/usr/bin/env python3
"""
archive_conversations.py

Archives old live session folders so live_ai_worker.py does not re-process stale
data/gelen_json/conv_* folders on startup.

Default:
  data/gelen_json/conv_* -> data/archive/<timestamp>/gelen_json/
  data/giden_json/conv_* -> data/archive/<timestamp>/giden_json/
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def log(msg: str) -> None:
    print(msg, flush=True)


def is_archivable_dir(path: Path, conv_id: Optional[str] = None) -> bool:
    if not path.is_dir():
        return False
    if path.name.startswith("."):
        return False
    if path.name in {"__pycache__", "processed", "archive"}:
        return False
    if conv_id is not None:
        return path.name == conv_id
    return path.name.startswith("conv_")


def dir_age_sec(path: Path) -> float:
    try:
        return time.time() - path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def unique_destination(base: Path) -> Path:
    if not base.exists():
        return base
    i = 1
    while True:
        candidate = base.with_name(f"{base.name}__{i}")
        if not candidate.exists():
            return candidate
        i += 1


def collect_dirs(root: Path, conv_id: Optional[str], min_age_sec: float) -> List[Path]:
    if not root.exists():
        return []
    out = []
    for p in sorted(root.iterdir(), key=lambda x: x.name):
        if not is_archivable_dir(p, conv_id=conv_id):
            continue
        if min_age_sec > 0 and dir_age_sec(p) < min_age_sec:
            continue
        out.append(p)
    return out


def move_dirs(
    source_root: Path,
    archive_subdir: Path,
    label: str,
    conv_id: Optional[str],
    min_age_sec: float,
    dry_run: bool,
) -> List[Dict[str, str]]:
    moved = []
    dirs = collect_dirs(source_root, conv_id=conv_id, min_age_sec=min_age_sec)

    if not dirs:
        log(f"[{label}] no folders to archive.")
        return moved

    if not dry_run:
        archive_subdir.mkdir(parents=True, exist_ok=True)

    for src in dirs:
        dst = unique_destination(archive_subdir / src.name)
        log(f"[{label}] {src} -> {dst}")

        moved.append({
            "label": label,
            "source": str(src),
            "destination": str(dst),
            "folder": src.name,
        })

        if not dry_run:
            shutil.move(str(src), str(dst))

    return moved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gelen-dir", default="data/gelen_json")
    parser.add_argument("--giden-dir", default="data/giden_json")
    parser.add_argument("--archive-root", default="data/archive")
    parser.add_argument("--conv-id", default=None, help="Archive only this conversation id.")
    parser.add_argument("--min-age-sec", type=float, default=0.0, help="Only archive folders older than this.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gelen_dir = Path(args.gelen_dir).expanduser()
    giden_dir = Path(args.giden_dir).expanduser()
    archive_root = Path(args.archive_root).expanduser()

    archive_session_dir = archive_root / now_stamp()
    archive_gelen_dir = archive_session_dir / "gelen_json"
    archive_giden_dir = archive_session_dir / "giden_json"

    log("Archive conversations")
    log(f"  gelen_dir     : {gelen_dir}")
    log(f"  giden_dir     : {giden_dir}")
    log(f"  archive_batch : {archive_session_dir}")
    log(f"  conv_id       : {args.conv_id}")
    log(f"  min_age_sec   : {args.min_age_sec}")
    log(f"  dry_run       : {args.dry_run}")
    log("")

    moved = []
    moved.extend(move_dirs(gelen_dir, archive_gelen_dir, "gelen", args.conv_id, args.min_age_sec, args.dry_run))
    moved.extend(move_dirs(giden_dir, archive_giden_dir, "giden", args.conv_id, args.min_age_sec, args.dry_run))

    manifest = {
        "created_at_unix": time.time(),
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "archive_batch": str(archive_session_dir),
        "gelen_dir": str(gelen_dir),
        "giden_dir": str(giden_dir),
        "conv_id": args.conv_id,
        "min_age_sec": args.min_age_sec,
        "dry_run": args.dry_run,
        "moved": moved,
    }

    if not args.dry_run and moved:
        archive_session_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = archive_session_dir / "archive_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        log("")
        log(f"Wrote manifest: {manifest_path}")

    log("")
    log(f"Archived folders: {len(moved)}")
    if args.dry_run:
        log("Dry run only; no folders were moved.")


if __name__ == "__main__":
    main()
