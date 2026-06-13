#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def norm_name(x: str) -> str:
    return str(x).strip()


def motion_blur(img, ksize: int, angle: float):
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0
    center = (ksize / 2 - 0.5, ksize / 2 - 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (ksize, ksize))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def safe_visual_aug(img, rng: random.Random):
    out = img.copy()

    # Motion/camera blur: the main thing you asked for.
    if rng.random() < 0.75:
        out = motion_blur(
            out,
            ksize=rng.choice([5, 7, 9, 11, 13]),
            angle=rng.uniform(-30, 30),
        )

    # Focus/defocus blur.
    if rng.random() < 0.30:
        k = rng.choice([3, 5, 7])
        out = cv2.GaussianBlur(out, (k, k), 0)

    # Lighting changes.
    if rng.random() < 0.75:
        alpha = rng.uniform(0.70, 1.30)
        beta = rng.randint(-30, 30)
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    # Sensor noise.
    if rng.random() < 0.45:
        sigma = rng.uniform(4, 16)
        noise = np.random.normal(0, sigma, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Stream/JPEG artifacts.
    if rng.random() < 0.55:
        q = rng.randint(40, 85)
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            decoded = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if decoded is not None:
                out = decoded

    return out


def image_path_for(coco_dir: Path, file_name: str) -> Path | None:
    candidates = [
        coco_dir / file_name,
        coco_dir / "images" / file_name,
        coco_dir / "train" / file_name,
    ]

    for p in candidates:
        if p.exists():
            return p

    return None


def polygon_area(poly: List[float]) -> float:
    if len(poly) < 6:
        return 0.0
    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def normalize_polygon(poly: List[float], width: int, height: int) -> List[float]:
    out: List[float] = []
    for i, value in enumerate(poly):
        if i % 2 == 0:
            out.append(min(max(float(value) / width, 0.0), 1.0))
        else:
            out.append(min(max(float(value) / height, 0.0), 1.0))
    return out


def ann_to_yolo_lines(
    anns: List[Dict[str, Any]],
    cat_id_to_yolo_id: Dict[int, int],
    width: int,
    height: int,
) -> List[str]:
    lines: List[str] = []

    for ann in anns:
        cat_id = int(ann.get("category_id", -1))
        if cat_id not in cat_id_to_yolo_id:
            continue

        seg = ann.get("segmentation")
        if not isinstance(seg, list) or not seg:
            # RLE or invalid segmentation is skipped.
            continue

        # Roboflow COCO segmentation normally has one polygon.
        # If multiple exist, keep the largest valid polygon to avoid duplicated objects.
        polygons = [p for p in seg if isinstance(p, list) and len(p) >= 6]
        if not polygons:
            continue

        poly = max(polygons, key=polygon_area)
        norm_poly = normalize_polygon(poly, width, height)

        if len(norm_poly) < 6:
            continue

        cls = cat_id_to_yolo_id[cat_id]
        values = " ".join(f"{v:.6f}" for v in norm_poly)
        lines.append(f"{cls} {values}")

    return lines


def group_key_for_image(img: Dict[str, Any]) -> str:
    # Keeps Roboflow variants of the same original image in the same split.
    extra = img.get("extra") or {}
    if isinstance(extra, dict) and extra.get("name"):
        return str(extra["name"])

    name = str(img.get("file_name", ""))
    if ".rf." in name:
        return name.split(".rf.")[0]
    return Path(name).stem


def split_groups(groups: Dict[str, List[int]], seed: int):
    rng = random.Random(seed)
    keys = list(groups.keys())
    rng.shuffle(keys)

    n = len(keys)
    n_train = int(n * 0.75)
    n_val = int(n * 0.15)

    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train:n_train + n_val])
    test_keys = set(keys[n_train + n_val:])

    image_id_to_split = {}
    for key, ids in groups.items():
        if key in train_keys:
            split = "train"
        elif key in val_keys:
            split = "val"
        else:
            split = "test"
        for image_id in ids:
            image_id_to_split[image_id] = split

    return image_id_to_split


def copy_original_image(src_img: Path, dst_img: Path):
    dst_img.parent.mkdir(parents=True, exist_ok=True)

    # Re-save as JPG to normalize extensions and avoid webp/png surprises.
    img = cv2.imread(str(src_img))
    if img is None:
        return False

    cv2.imwrite(str(dst_img), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return True


def write_augmented_copy(task):
    src_img, src_lbl, dst_img, dst_lbl, seed = task
    src_img = Path(src_img)
    src_lbl = Path(src_lbl)
    dst_img = Path(dst_img)
    dst_lbl = Path(dst_lbl)

    rng = random.Random(seed)
    np.random.seed(seed % (2**32 - 1))

    img = cv2.imread(str(src_img))
    if img is None:
        return False, str(src_img)

    aug = safe_visual_aug(img, rng)

    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(dst_img), aug, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    shutil.copy2(src_lbl, dst_lbl)

    return True, str(dst_img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", default="data/demoday_train/train")
    parser.add_argument("--coco-json", default=None)
    parser.add_argument("--out-dir", default="data/demoday_train_yolo_aug3")
    parser.add_argument("--aug-copies", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop-categories", nargs="*", default=["3D-Printer-Day-1-1", "prit"])
    parser.add_argument("--keep-empty-labels", action="store_true")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    coco_json = Path(args.coco_json) if args.coco_json else src_dir / "_annotations.coco.json"
    out_dir = Path(args.out_dir)

    if not coco_json.exists():
        alt = src_dir / "_annotations.coco (1).json"
        if alt.exists():
            coco_json = alt
        else:
            raise FileNotFoundError(f"COCO JSON not found: {coco_json}")

    if out_dir.exists():
        raise FileExistsError(f"Output already exists: {out_dir}. Remove it first or choose another --out-dir.")

    coco = json.loads(coco_json.read_text(encoding="utf-8"))

    images = {int(img["id"]): img for img in coco.get("images", [])}
    anns_by_image: Dict[int, List[Dict[str, Any]]] = {img_id: [] for img_id in images}

    used_cat_ids = set()
    for ann in coco.get("annotations", []):
        img_id = int(ann.get("image_id", -1))
        if img_id in anns_by_image:
            anns_by_image[img_id].append(ann)
            used_cat_ids.add(int(ann.get("category_id", -1)))

    drop = {norm_name(x) for x in args.drop_categories}
    categories = []
    for cat in coco.get("categories", []):
        cat_id = int(cat["id"])
        name = norm_name(cat["name"])

        # Skip dataset/root category and categories not present in annotations.
        if cat_id not in used_cat_ids:
            continue
        if name in drop:
            continue

        categories.append((cat_id, name))

    categories = sorted(categories, key=lambda x: x[0])
    cat_id_to_yolo_id = {cat_id: i for i, (cat_id, _) in enumerate(categories)}
    yolo_names = [name for _, name in categories]

    print("Classes:")
    for i, name in enumerate(yolo_names):
        print(f"  {i}: {name}")

    groups: Dict[str, List[int]] = {}
    valid_image_ids = []

    for img_id, img in images.items():
        src_img = image_path_for(src_dir, img["file_name"])
        if src_img is None:
            print(f"Missing image file, skipping: {img['file_name']}")
            continue

        lines = ann_to_yolo_lines(
            anns_by_image.get(img_id, []),
            cat_id_to_yolo_id,
            int(img["width"]),
            int(img["height"]),
        )

        if not lines and not args.keep_empty_labels:
            continue

        valid_image_ids.append(img_id)
        key = group_key_for_image(img)
        groups.setdefault(key, []).append(img_id)

    image_id_to_split = split_groups(groups, args.seed)

    split_counts = {"train": 0, "val": 0, "test": 0}
    train_for_aug: List[Tuple[Path, Path, str]] = []

    for img_id in valid_image_ids:
        img = images[img_id]
        split = image_id_to_split[img_id]
        split_counts[split] += 1

        src_img = image_path_for(src_dir, img["file_name"])
        if src_img is None:
            continue

        stem = Path(img["file_name"]).stem
        dst_img = out_dir / "images" / split / f"{stem}.jpg"
        dst_lbl = out_dir / "labels" / split / f"{stem}.txt"

        ok = copy_original_image(src_img, dst_img)
        if not ok:
            print(f"Unreadable image, skipping: {src_img}")
            continue

        lines = ann_to_yolo_lines(
            anns_by_image.get(img_id, []),
            cat_id_to_yolo_id,
            int(img["width"]),
            int(img["height"]),
        )

        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        if split == "train" and lines:
            train_for_aug.append((dst_img, dst_lbl, stem))

    print("\nSplit image counts before offline augmentation:")
    print(split_counts)

    aug_tasks = []
    for src_img, src_lbl, stem in train_for_aug:
        for i in range(args.aug_copies):
            aug_stem = f"{stem}_aug{i:02d}"
            dst_img = out_dir / "images" / "train" / f"{aug_stem}.jpg"
            dst_lbl = out_dir / "labels" / "train" / f"{aug_stem}.txt"
            aug_seed = args.seed + (abs(hash((stem, i))) % 10_000_000)
            aug_tasks.append((str(src_img), str(src_lbl), str(dst_img), str(dst_lbl), aug_seed))

    if aug_tasks:
        print(f"\nCreating {len(aug_tasks)} augmented train images with {args.workers} workers...")
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(write_augmented_copy, task) for task in aug_tasks]
            for fut in as_completed(futures):
                ok, msg = fut.result()
                done += int(ok)
        print(f"Augmented images written: {done}")

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        "path: " + str(out_dir.resolve()) + "\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n" +
        "".join(f"  {i}: {name}\n" for i, name in enumerate(yolo_names)),
        encoding="utf-8",
    )

    print("\nFinal file counts:")
    for split in ["train", "val", "test"]:
        n_img = len(list((out_dir / "images" / split).glob("*.jpg")))
        n_lbl = len(list((out_dir / "labels" / split).glob("*.txt")))
        print(f"  {split}: images={n_img}, labels={n_lbl}")

    print(f"\nWrote: {data_yaml}")


if __name__ == "__main__":
    main()
