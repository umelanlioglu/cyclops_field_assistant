#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
from pathlib import Path
from typing import List, Optional


def now() -> str:
    return time.strftime("%H:%M:%S")


def frame_id_from_name(name: str) -> Optional[int]:
    m = re.search(r"(?:f|frame)_(\d+)", name)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else None


def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def mtime(path: Path) -> Optional[float]:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def latest_conv(gelen_dir: Path, giden_dir: Path) -> Optional[str]:
    candidates = []
    for root in [gelen_dir, giden_dir]:
        if not root.exists():
            continue
        for p in root.iterdir():
            if p.is_dir() and p.name.startswith("conv_"):
                try:
                    candidates.append((p.stat().st_mtime, p.name))
                except FileNotFoundError:
                    pass
    if not candidates:
        return None
    return sorted(candidates)[-1][1]


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    idx = (len(values) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1f} ms"


def summary(values: List[float]) -> str:
    values = [v for v in values if v is not None and v >= 0]
    if not values:
        return "n/a"
    return (
        f"count={len(values)} "
        f"mean={statistics.mean(values):.1f}ms "
        f"p50={percentile(values, 0.50):.1f}ms "
        f"p90={percentile(values, 0.90):.1f}ms "
        f"p95={percentile(values, 0.95):.1f}ms "
        f"max={max(values):.1f}ms"
    )


def collect_frame_rows(gelen_conv: Path, giden_conv: Path, limit: Optional[int]) -> List[dict]:
    out_jsons = sorted(
        giden_conv.glob("r_frame_*.json"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
    )
    if limit is not None:
        out_jsons = out_jsons[-limit:]

    rows = []
    for out_json in out_jsons:
        meta = read_json(out_json)
        if not meta:
            continue

        source_name = meta.get("source_frame_file")
        image_file = meta.get("image_file")
        frame_id = meta.get("frame_id")

        if not source_name:
            fid = frame_id_from_name(out_json.name)
            if fid is None:
                continue
            source_name = f"f_{fid:06d}.jpg"
            frame_id = fid

        in_jpg = gelen_conv / source_name
        out_jpg = giden_conv / image_file if image_file else out_json.with_suffix(".jpg")

        in_t = mtime(in_jpg)
        out_json_t = mtime(out_json)
        out_jpg_t = mtime(out_jpg)
        created_t = meta.get("created_at_unix")

        rows.append({
            "frame_id": frame_id,
            "source_frame_file": source_name,
            "output_json": out_json.name,
            "output_jpg": out_jpg.name,
            "visible_objects": ",".join(meta.get("visible_objects", [])),
            "has_annotations": meta.get("has_annotations"),
            "input_to_output_json_mtime_ms": (out_json_t - in_t) * 1000 if in_t and out_json_t else None,
            "input_to_output_json_created_ms": (float(created_t) - in_t) * 1000 if in_t and isinstance(created_t, (int, float)) else None,
            "input_to_output_jpg_mtime_ms": (out_jpg_t - in_t) * 1000 if in_t and out_jpg_t else None,
            "input_mtime": in_t,
            "output_json_mtime": out_json_t,
            "output_jpg_mtime": out_jpg_t,
        })
    return rows


def collect_audio_rows(gelen_conv: Path, giden_conv: Path, limit: Optional[int]) -> List[dict]:
    out_jsons = sorted(
        giden_conv.glob("r_a_*.json"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
    )
    if limit is not None:
        out_jsons = out_jsons[-limit:]

    rows = []
    for out_json in out_jsons:
        meta = read_json(out_json)
        if not meta:
            continue

        audio_file = meta.get("audio_file")
        audio_id = meta.get("audio_id")
        if not audio_file and audio_id:
            audio_file = f"{audio_id}.m4a"
        if not audio_file:
            continue

        in_audio = gelen_conv / audio_file
        in_audio_json = in_audio.with_suffix(".json")
        in_audio_t = mtime(in_audio)
        in_audio_json_t = mtime(in_audio_json)
        out_json_t = mtime(out_json)
        created_t = meta.get("created_at_unix")

        rows.append({
            "audio_id": audio_id,
            "output_json": out_json.name,
            "transcript": meta.get("transcript", ""),
            "status": meta.get("status", "ok"),
            "audio_to_output_json_mtime_ms": (out_json_t - in_audio_t) * 1000 if in_audio_t and out_json_t else None,
            "audio_json_to_output_json_mtime_ms": (out_json_t - in_audio_json_t) * 1000 if in_audio_json_t and out_json_t else None,
            "audio_to_output_json_created_ms": (float(created_t) - in_audio_t) * 1000 if in_audio_t and isinstance(created_t, (int, float)) else None,
        })
    return rows


def print_frames(rows: List[dict], n: int) -> None:
    for r in rows[-n:]:
        objs = r["visible_objects"]
        if len(objs) > 60:
            objs = objs[:57] + "..."
        print(
            f"frame={r['frame_id']} "
            f"jpg→json_mtime={fmt(r['input_to_output_json_mtime_ms'])} "
            f"jpg→json_created={fmt(r['input_to_output_json_created_ms'])} "
            f"jpg→jpg_mtime={fmt(r['input_to_output_jpg_mtime_ms'])} "
            f"ann={r['has_annotations']} objs=[{objs}]",
            flush=True,
        )


def print_audios(rows: List[dict], n: int) -> None:
    for r in rows[-n:]:
        transcript = (r["transcript"] or "").replace("\n", " ")
        if len(transcript) > 50:
            transcript = transcript[:47] + "..."
        print(
            f"audio={r['audio_id']} "
            f"m4a→json={fmt(r['audio_to_output_json_mtime_ms'])} "
            f"meta→json={fmt(r['audio_json_to_output_json_mtime_ms'])} "
            f"status={r['status']} transcript={transcript!r}",
            flush=True,
        )


def print_summary(frame_rows: List[dict], audio_rows: List[dict]) -> None:
    print("\n===== SUMMARY =====", flush=True)
    print("frame jpg → output json mtime  :", summary([r["input_to_output_json_mtime_ms"] for r in frame_rows]), flush=True)
    print("frame jpg → output json created:", summary([r["input_to_output_json_created_ms"] for r in frame_rows]), flush=True)
    print("frame jpg → output jpg mtime   :", summary([r["input_to_output_jpg_mtime_ms"] for r in frame_rows]), flush=True)
    if audio_rows:
        print("audio m4a → answer json mtime :", summary([r["audio_to_output_json_mtime_ms"] for r in audio_rows]), flush=True)
        print("audio meta → answer json mtime:", summary([r["audio_json_to_output_json_mtime_ms"] for r in audio_rows]), flush=True)
    print("===================\n", flush=True)


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gelen-dir", default="data/gelen_json")
    parser.add_argument("--giden-dir", default="data/giden_json")
    parser.add_argument("--conv-id", default=None)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--latest", type=int, default=5)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--audio", action="store_true")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    gelen_dir = Path(args.gelen_dir)
    giden_dir = Path(args.giden_dir)
    conv_id = args.conv_id or latest_conv(gelen_dir, giden_dir)

    if not conv_id:
        raise SystemExit("No conv_* folder found. Pass --conv-id.")

    gelen_conv = gelen_dir / conv_id
    giden_conv = giden_dir / conv_id

    print(f"conv_id: {conv_id}", flush=True)
    print(f"gelen  : {gelen_conv}", flush=True)
    print(f"giden  : {giden_conv}", flush=True)

    last_seen = None

    while True:
        frame_rows = collect_frame_rows(gelen_conv, giden_conv, args.limit)
        audio_rows = collect_audio_rows(gelen_conv, giden_conv, args.limit) if args.audio else []

        current = (
            frame_rows[-1]["output_json"] if frame_rows else None,
            audio_rows[-1]["output_json"] if audio_rows else None,
            len(frame_rows),
            len(audio_rows),
        )

        if not args.watch or current != last_seen:
            print("\n" + "-" * 80, flush=True)
            print(f"[{now()}] latest latency", flush=True)
            print_frames(frame_rows, args.latest)
            if args.audio:
                print_audios(audio_rows, args.latest)
            print_summary(frame_rows, audio_rows)
            last_seen = current

            if args.csv and not args.watch:
                write_csv(Path(args.csv), frame_rows)
                print(f"Wrote CSV: {args.csv}", flush=True)

        if not args.watch:
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
