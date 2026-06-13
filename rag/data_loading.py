"""Small JSON and COCO loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(path: str | Path) -> Any:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_manual_chunks(path: str | Path) -> List[Dict[str, Any]]:
    chunks = load_json(path)
    if not isinstance(chunks, list):
        raise ValueError("manual_chunks JSON must be a list of chunk objects.")
    return chunks


def load_coco_dataset(dataset_dir: str | Path) -> Tuple[Dict[str, Any], Path]:
    """Load one COCO JSON and find the referenced image.

    Works with either:
      data/test_image2/annotations.json
      data/test_image2/test3dprinter.jpeg

    or nested folders under dataset_dir.
    """
    dataset_dir = Path(dataset_dir)
    json_files = list(dataset_dir.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON file found under {dataset_dir}")

    coco_path = json_files[0]
    coco = load_json(coco_path)

    if not coco.get("images"):
        raise ValueError("COCO JSON has no images.")

    image_name = coco["images"][0]["file_name"]
    direct_path = dataset_dir / image_name

    if direct_path.exists():
        image_path = direct_path
    else:
        matches = list(dataset_dir.rglob(image_name))
        if not matches:
            raise FileNotFoundError(
                f"Image '{image_name}' referenced in COCO was not found under {dataset_dir}"
            )
        image_path = matches[0]

    return coco, image_path


def coco_category_map(coco: Dict[str, Any]) -> Dict[int, str]:
    return {cat["id"]: cat["name"] for cat in coco.get("categories", [])}
