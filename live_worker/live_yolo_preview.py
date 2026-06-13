from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import time
from pathlib import Path
import json

import cv2

def load_visual_targets(out_dir: Path):
    path = out_dir / "visual_targets.json"
    if not path.exists():
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        targets = payload.get("visual_targets", [])
        return targets if isinstance(targets, list) else []
    except Exception:
        return []


def import_package(package_dir: Path, package_name: str = "cr10_live_yolo_preview_pkg") -> str:
    init_file = package_dir / "__init__.py"

    spec = importlib.util.spec_from_file_location(
        package_name,
        init_file,
        submodule_search_locations=[str(package_dir)],
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import package from {package_dir}")

    pkg = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = pkg
    spec.loader.exec_module(pkg)

    return package_name


def atomic_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def atomic_write_image(path: Path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp.jpg")
    cv2.imwrite(str(tmp), frame)
    os.replace(tmp, path)


def draw_debug_overlay(frame, visible_objects, frame_mtime):
    out = frame.copy()

    text_1 = f"Live YOLO preview | {time.strftime('%H:%M:%S')}"
    text_2 = "Visible: " + (", ".join(visible_objects) if visible_objects else "none")

    cv2.rectangle(out, (8, 8), (900, 70), (0, 0, 0), -1)
    cv2.putText(out, text_1, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(out, text_2[:90], (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--rag-package-dir", required=True)
    parser.add_argument("--frame-path", required=True)
    parser.add_argument("--out-image", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--yolo-weights", required=True)
    parser.add_argument("--manual-chunks", required=True)

    parser.add_argument("--semantic-index", action="store_true")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--fps", type=float, default=3.0)

    args = parser.parse_args()

    package_dir = Path(args.rag_package_dir).resolve()
    package_name = import_package(package_dir)

    LiveStreamController = __import__(
        f"{package_name}.live_stream_controller",
        fromlist=["LiveStreamController"],
    ).LiveStreamController

    data_loading = __import__(f"{package_name}.data_loading", fromlist=[""])
    retrieval = __import__(f"{package_name}.retrieval", fromlist=[""])

    manual_chunks = data_loading.load_manual_chunks(Path(args.manual_chunks))

    rag_index = None
    if args.semantic_index:
        print("Building semantic RAG index for live YOLO preview...", flush=True)
        rag_index = retrieval.build_rag_index(manual_chunks)
        print("Semantic RAG index ready.", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    controller = LiveStreamController(
        yolo_weights_path=Path(args.yolo_weights),
        manual_chunks=manual_chunks,
        rag_index=rag_index,
        output_dir=out_dir / "live_yolo_controller",
        yolo_conf=args.yolo_conf,
        yolo_imgsz=args.yolo_imgsz,
    )

    frame_path = Path(args.frame_path)
    out_image = Path(args.out_image)

    last_mtime = None
    delay = 1.0 / max(args.fps, 0.01)

    print("Live YOLO preview started.", flush=True)
    print(f"Input : {frame_path}", flush=True)
    print(f"Output: {out_image}", flush=True)

    while True:
        try:
            if not frame_path.exists():
                print("Waiting for input frame...", flush=True)
                time.sleep(0.5)
                continue

            mtime = frame_path.stat().st_mtime

            if last_mtime == mtime:
                time.sleep(0.05)
                continue

            last_mtime = mtime

            frame = cv2.imread(str(frame_path))
            if frame is None:
                print("Could not read latest frame.", flush=True)
                time.sleep(0.1)
                continue

            controller.active_visual_targets = load_visual_targets(out_dir)

            info = controller.process_frame(frame)
            visible_objects = info.get("visible_objects", []) or []
            annotated_path = info.get("annotated_image_path")

            if annotated_path and Path(annotated_path).exists():
                atomic_copy(Path(annotated_path), out_image)
            else:
                debug_frame = draw_debug_overlay(frame, visible_objects, mtime)
                atomic_write_image(out_image, debug_frame)

            print(
                f"[{time.strftime('%H:%M:%S')}] wrote {out_image.name} | visible={visible_objects}",
                flush=True,
            )

            time.sleep(delay)

        except KeyboardInterrupt:
            print("\nStopped live YOLO preview.", flush=True)
            break

        except Exception as e:
            print(f"Error: {repr(e)}", flush=True)
            time.sleep(0.5)


if __name__ == "__main__":
    main()