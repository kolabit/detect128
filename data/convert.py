from __future__ import annotations

import json
from pathlib import Path


def coco_to_yolo(
    coco_json: Path,
    images_dir: Path,
    labels_dir: Path,
    require_images_exist: bool = False,
) -> dict:
    """
    Convert one COCO json (instances style) into YOLO .txt labels.

    Returns a dict with category mapping and simple stats.
    """
    labels_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(coco_json.read_text(encoding="utf-8"))

    # COCO category ids can be non-contiguous; map them to 0..N-1
    # categories = sorted(data.get("categories", []), key=lambda c: c["id"])
    # cat_id2yolo = {c["id"]: i for i, c in enumerate(categories)}

    categories = sorted(data.get("categories", []), key=lambda c: c["id"])

    cat_id2yolo = {c["id"]: i for i, c in enumerate(categories)}

    yolo_to_name = {
        cat_id2yolo[c["id"]]: c.get("name", str(c["id"])) for c in categories
    }

    images = {img["id"]: img for img in data.get("images", [])}

    # Collect annotations per image_id
    anns_by_image = {}
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0) == 1:
            continue  # YOLO training usually ignores crowd boxes
        img_id = ann["image_id"]
        anns_by_image.setdefault(img_id, []).append(ann)

    written = 0
    empty = 0

    for img_id, img in images.items():
        file_name = img["file_name"]
        w_img = img["width"]
        h_img = img["height"]

        img_path = images_dir / file_name
        if require_images_exist and not img_path.exists():
            # Skip if images are not present
            continue

        # Label file name matches image stem
        label_path = labels_dir / (Path(file_name).stem + ".txt")

        anns = anns_by_image.get(img_id, [])
        if not anns:
            # Create empty label file (optional but common)
            label_path.write_text("", encoding="utf-8")
            empty += 1
            continue

        lines = []
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in cat_id2yolo:
                continue

            x, y, bw, bh = ann[
                "bbox"
            ]  # COCO bbox is top-left x,y + width,height (pixels)
            # Clamp to image bounds (helps with slightly out-of-range boxes)
            x = max(0.0, min(float(x), w_img - 1.0))
            y = max(0.0, min(float(y), h_img - 1.0))
            bw = max(0.0, min(float(bw), w_img - x))
            bh = max(0.0, min(float(bh), h_img - y))

            # Convert to YOLO normalized center format
            xc = (x + bw / 2.0) / w_img
            yc = (y + bh / 2.0) / h_img
            wn = bw / w_img
            hn = bh / h_img

            # Filter invalid boxes
            if wn <= 0 or hn <= 0:
                continue

            cls = cat_id2yolo[cat_id]
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        label_path.write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        written += 1

    return {
        "num_images": len(images),
        "num_written": written,
        "num_empty": empty,
        "cat_id2yolo": cat_id2yolo,
        "yolo_to_name": yolo_to_name,
    }


if __name__ == "__main__":
    # Example usage (edit paths):
    # root = Path("/home/kola/sources/github/kolabit/detect128/data/dataset")
    root = Path("dataset")
    stats_train = coco_to_yolo(
        # images/test/_annotations.coco.json
        coco_json=root / "images" / "train" / "_annotations.coco.json",
        images_dir=root / "images" / "train",
        labels_dir=root / "labels" / "train",
    )
    stats_val = coco_to_yolo(
        coco_json=root / "images" / "val" / "_annotations.coco.json",
        images_dir=root / "images" / "val",
        labels_dir=root / "labels" / "val",
    )
    stats_test = coco_to_yolo(
        coco_json=root / "images" / "test" / "_annotations.coco.json",
        images_dir=root / "images" / "test",
        labels_dir=root / "labels" / "test",
    )

    print(
        "Train:",
        stats_train["num_written"],
        "images,",
        stats_train["num_empty"],
        "empty labels",
    )
    print(
        "Val:",
        stats_val["num_written"],
        "images,",
        stats_val["num_empty"],
        "empty labels",
    )
    print(
        "Test:",
        stats_test["num_written"],
        "images,",
        stats_test["num_empty"],
        "empty labels",
    )
    print("Classes:", stats_train["yolo_to_name"])
