#!/usr/bin/env python3
"""Audit the COCO dataset for laboratory work 6."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
TRUDI = ROOT.parent / "datasets/trudi/coco/aerial"
OUT = ROOT / "runs/variant2_audit"


def size_name(area: float) -> str:
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def side_histogram(values: list[float]) -> dict[str, int]:
    bounds=(0,32,64,128,256,512,float("inf"))
    names=("<32","[32,64)","[64,128)","[128,256)","[256,512)","≥512")
    counts,_=np.histogram(values,bins=bounds)
    return dict(zip(names,(int(count) for count in counts)))


def inspect_dataset(kind: str, root: Path) -> dict:
    summary = {"dataset": kind, "root": str(root), "splits": {}, "warnings": []}
    all_names: dict[str, set[str]] = {}
    previews = []
    for split in ("train", "val", "test"):
        folder = root / (split if kind == "TRUDI" else {"train": "train_rgb", "val": "valid_rgb", "test": "test_rgb"}[split])
        coco = json.loads((folder / ("annotations.json" if kind == "TRUDI" else "_annotations.coco.json")).read_text(encoding="utf-8-sig"))
        categories = {c["id"]: c["name"] for c in coco["categories"] if c["id"] != 0 or kind == "TRUDI"}
        image_by_id = {im["id"]: im for im in coco["images"]}
        if len(image_by_id) != len(coco["images"]):
            raise ValueError(f"{kind}/{split}: duplicate image IDs")
        all_names[split] = {im["file_name"] for im in coco["images"]}
        anns = defaultdict(list)
        counts, sizes = Counter(), Counter()
        widths, heights = [], []
        for ann in coco["annotations"]:
            if ann["image_id"] not in image_by_id or ann["category_id"] not in categories:
                raise ValueError(f"{kind}/{split}: invalid annotation reference {ann['id']}")
            im = image_by_id[ann["image_id"]]
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0 or x < -1 or y < -1 or x + w > im["width"] + 1 or y + h > im["height"] + 1:
                raise ValueError(f"{kind}/{split}: invalid box {ann['id']}")
            anns[ann["image_id"]].append(ann)
            counts[categories[ann["category_id"]]] += 1
            sizes[size_name(w*h)] += 1
            widths.append(w)
            heights.append(h)
        missing = [im["file_name"] for im in coco["images"] if not (folder / im["file_name"]).is_file()]
        if missing:
            raise FileNotFoundError(f"{kind}/{split}: {len(missing)} images missing, first: {missing[0]}")
        summary["splits"][split] = {
            "images": len(coco["images"]), "annotations": len(coco["annotations"]),
            "empty_images": len(coco["images"]) - len(anns),
            "classes": dict(sorted(counts.items())), "sizes_coco": dict(sorted(sizes.items())),
            "box_width_px": {"min": min(widths), "median": float(np.median(widths)), "max": max(widths)},
            "box_height_px": {"min": min(heights), "median": float(np.median(heights)), "max": max(heights)},
            "box_width_hist_px": side_histogram(widths),
            "box_height_hist_px": side_histogram(heights),
            "image_dimensions": dict(Counter(f"{im['width']}x{im['height']}" for im in coco["images"]).most_common(5)),
        }
        chosen = [im for im in coco["images"] if anns.get(im["id"])][:6]
        for im in chosen:
            with Image.open(folder / im["file_name"]) as source:
                image = source.convert("RGB")
            scale = min(400/image.width, 225/image.height)
            image = image.resize((round(image.width*scale), round(image.height*scale)))
            draw = ImageDraw.Draw(image)
            for ann in anns[im["id"]][:50]:
                x,y,w,h = ann["bbox"]
                draw.rectangle((x*scale,y*scale,(x+w)*scale,(y+h)*scale),outline="red",width=2)
            previews.append((split, image))
    for a,b in (("train","val"),("train","test"),("val","test")):
        overlap = all_names[a] & all_names[b]
        summary["splits"][f"{a}_{b}_same_filenames"] = len(overlap)
        if overlap:
            summary["warnings"].append(f"{a}/{b}: {len(overlap)} matching filenames")
    canvas = Image.new("RGB",(1200, 6*245),"white")
    label = ImageDraw.Draw(canvas)
    for i,(split,image) in enumerate(previews):
        x=(i%3)*400; y=(i//3)*245
        canvas.paste(image,(x,y+20))
        label.text((x+3,y+3),f"{split}: {i+1}",fill="black")
    OUT.mkdir(parents=True,exist_ok=True)
    canvas.save(OUT / {"ReCoDeWaste":"recodewaste_18_boxes.png","TRUDI":"trudi_18_boxes.png","DroneWaste":"dronewaste_18_boxes.png"}[kind])
    return summary



def main() -> None:
    summary = inspect_dataset("TRUDI", TRUDI)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dataset_audit.json").write_text(json.dumps({"lr6": summary}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["splits"], indent=2, ensure_ascii=True))

if __name__ == "__main__":
    main()
