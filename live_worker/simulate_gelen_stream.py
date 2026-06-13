#!/usr/bin/env python3
"""
simulate_gelen_stream.py

Simulates the Android/WebSocket backend by writing frames and optional audio into:

    data/gelen_json/<conv_id>/

The live_ai_worker.py watches that folder and writes outputs to data/giden_json/<conv_id>/.

Example:

python live_worker/simulate_gelen_stream.py \
  --frame-dir data/live_test/frames_720p_10fps \
  --gelen-dir data/gelen_json \
  --giden-dir data/giden_json \
  --conv-id conv_sim_test \
  --fps 10 \
  --clear \
  --audio data/gelen_json/conv_a57587baf98c/a_03808729_2e4b.m4a \
  --audio-at-frame 60 \
  --audio-duration-ms 6000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import List, Optional


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def now() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def frame_id_from_name(path: Path) -> Optional[int]:
    m = re.search(r"(?:f|frame)_(\d+)", path.stem)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", path.stem)
    return int(nums[-1]) if nums else None


def collect_frames(frame_dir: Path, glob_pattern: str) -> List[Path]:
    if not frame_dir.exists():
        raise FileNotFoundError(f"Frame dir not found: {frame_dir}")

    if glob_pattern:
        paths = [p for p in frame_dir.glob(glob_pattern) if p.suffix.lower() in IMAGE_EXTS]
    else:
        paths = [p for p in frame_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

    paths = sorted(
        paths,
        key=lambda p: (
            frame_id_from_name(p) if frame_id_from_name(p) is not None else 10**12,
            p.name,
        ),
    )

    if not paths:
        raise FileNotFoundError(f"No image frames found in {frame_dir} with glob={glob_pattern!r}")

    return paths


def clear_conversation(conv_dir: Path) -> None:
    if conv_dir.exists():
        shutil.rmtree(conv_dir)
    conv_dir.mkdir(parents=True, exist_ok=True)


def maybe_clear_giden(giden_dir: Optional[Path], conv_id: str) -> None:
    if giden_dir is None:
        return
    conv_out = giden_dir / conv_id
    if conv_out.exists():
        shutil.rmtree(conv_out)


def copy_audio_event(
    audio_path: Path,
    conv_dir: Path,
    audio_id: str,
    started_at_frame: int,
    ended_at_frame: int,
    duration_ms: int,
) -> None:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    out_audio = conv_dir / f"{audio_id}{audio_path.suffix.lower()}"
    out_meta = conv_dir / f"{audio_id}.json"

    meta = {
        "v": 1,
        "audio_id": audio_id,
        "audio_file": out_audio.name,
        "duration_ms": duration_ms,
        "started_at_frame": started_at_frame,
        "ended_at_frame": ended_at_frame,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    atomic_write_bytes(out_audio, audio_path.read_bytes())
    atomic_write_json(out_meta, meta)

    log(f"Wrote audio event: {out_audio.name}")
    log(f"Wrote audio meta : {out_meta.name}")
    log(f"Audio linked to frames {started_at_frame} -> {ended_at_frame}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--frame-dir", required=True, help="Folder containing source frames.")
    parser.add_argument("--frame-glob", default="", help="Optional glob, e.g. 'frame_*.jpg' or 'f_*.jpg'. Empty means all images.")
    parser.add_argument("--gelen-dir", default="data/gelen_json")
    parser.add_argument("--giden-dir", default=None, help="Optional. If passed with --clear, clears matching giden conv too.")
    parser.add_argument("--conv-id", default=None, help="Conversation folder name. Default: conv_sim_<timestamp>")

    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=None, help="Max number of frames to send.")
    parser.add_argument("--loop", action="store_true", help="Loop frames forever.")
    parser.add_argument("--clear", action="store_true", help="Clear gelen conv folder before starting.")

    parser.add_argument("--start-output-frame-id", type=int, default=1, help="First output frame id written as f_XXXXXX.jpg.")
    parser.add_argument("--preserve-source-ids", action="store_true", help="Use source frame ids instead of sequential output ids when possible.")

    parser.add_argument("--audio", default=None, help="Optional audio file to inject, e.g. .m4a.")
    parser.add_argument("--audio-id", default=None, help="Default: a_sim_<timestamp>")
    parser.add_argument("--audio-at-frame", type=int, default=None, help="Output frame id after which audio event is written.")
    parser.add_argument("--audio-start-offset", type=int, default=20, help="started_at_frame = audio_at_frame - offset.")
    parser.add_argument("--audio-duration-ms", type=int, default=3000)

    parser.add_argument("--write-session-json", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    frame_dir = Path(args.frame_dir).expanduser()
    gelen_dir = Path(args.gelen_dir).expanduser()
    giden_dir = Path(args.giden_dir).expanduser() if args.giden_dir else None

    conv_id = args.conv_id or f"conv_sim_{int(time.time())}"
    conv_dir = gelen_dir / conv_id

    frames = collect_frames(frame_dir, args.frame_glob)
    if args.limit is not None:
        frames = frames[:args.limit]

    if args.clear:
        clear_conversation(conv_dir)
        maybe_clear_giden(giden_dir, conv_id)
    else:
        conv_dir.mkdir(parents=True, exist_ok=True)

    if args.write_session_json and not args.dry_run:
        atomic_write_json(
            conv_dir / "session.json",
            {
                "v": 1,
                "type": "session.start",
                "conv_id": conv_id,
                "created_at_unix": time.time(),
                "simulated": True,
            },
        )

    log(f"Source frame dir: {frame_dir}")
    log(f"Frames found     : {len(frames)}")
    log(f"Gelen conv dir   : {conv_dir}")
    log(f"FPS              : {args.fps}")
    log(f"Clear            : {args.clear}")

    audio_path = Path(args.audio).expanduser() if args.audio else None
    audio_id = args.audio_id or f"a_sim_{int(time.time())}"

    if audio_path is not None:
        if args.audio_at_frame is None:
            args.audio_at_frame = args.start_output_frame_id + max(0, len(frames) // 2)
        log(f"Audio            : {audio_path}")
        log(f"Audio id         : {audio_id}")
        log(f"Audio at frame   : {args.audio_at_frame}")

    if args.dry_run:
        log("Dry run enabled. No files will be written.")
        return

    sleep_sec = 0.0 if args.fps <= 0 else 1.0 / args.fps
    audio_written = False
    total_written = 0
    loop_idx = 0

    try:
        while True:
            for src in frames:
                src_id = frame_id_from_name(src)

                if args.preserve_source_ids and src_id is not None:
                    out_id = src_id
                else:
                    out_id = args.start_output_frame_id + total_written

                out_frame = conv_dir / f"f_{out_id:06d}.jpg"

                atomic_write_bytes(out_frame, src.read_bytes())
                total_written += 1

                if total_written % 10 == 1 or total_written <= 5:
                    log(f"Wrote frame {out_frame.name} from {src.name}")

                if audio_path is not None and not audio_written and out_id >= args.audio_at_frame:
                    ended = out_id
                    started = max(args.start_output_frame_id, ended - args.audio_start_offset)
                    copy_audio_event(
                        audio_path=audio_path,
                        conv_dir=conv_dir,
                        audio_id=audio_id,
                        started_at_frame=started,
                        ended_at_frame=ended,
                        duration_ms=args.audio_duration_ms,
                    )
                    audio_written = True

                if sleep_sec > 0:
                    time.sleep(sleep_sec)

            loop_idx += 1
            if not args.loop:
                break

            log(f"Loop iteration complete: {loop_idx}")

    except KeyboardInterrupt:
        log("Stopped by user.")

    log(f"Done. Total frames written: {total_written}")
    log(f"Conversation id: {conv_id}")
    log(f"Input folder: {conv_dir}")


if __name__ == "__main__":
    main()
