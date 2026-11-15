"""Convert RDD2022 countries (Pascal VOC XML) to single-class YOLO 'pothole'.

One raw source folder per country, so prep_split.py can keep val/test on the
deployment country (India) while the other countries only add training data:

    datasets/raw/rdd_india/{images,labels}, datasets/raw/rdd_japan/..., ...

D40 = pothole -> class 0. Images with no D40 get an EMPTY label file (pothole-
free negatives). Official test folders have no XML and are skipped. Images
larger than --max-side (Norway's wide high-res frames) are downscaled; YOLO
labels are normalised so they stay valid.

    python prep_rdd2022.py --src downloads/rdd2022 --countries India
    python prep_rdd2022.py --src downloads/rdd2022 --countries all          # skips China_Drone
    python prep_rdd2022.py --src downloads/rdd2022 --countries all --include-drone
"""
import argparse
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2

from classical import load_bgr, resize_max
from data_utils import IMG_EXTS, save_json, write_yolo

POTHOLE_TAGS = {"D40"}
TOP_DOWN = {"China_Drone"}       # drone view: no horizon, breaks the road-trapezoid gate


def parse_voc(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w, h = int(float(size.find("width").text)), int(float(size.find("height").text))
    boxes = []
    for obj in root.iter("object"):
        name = obj.find("name")
        if name is None or name.text.strip() not in POTHOLE_TAGS:
            continue
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
    return root.findtext("filename", default=xml_path.stem), w, h, boxes


def country_of(xml_path: Path) -> str:
    """.../<Country>/train/annotations/xmls/x.xml -> <Country>"""
    parts = xml_path.parts
    return parts[parts.index("train") - 1] if "train" in parts else xml_path.parent.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="downloads/rdd2022")
    ap.add_argument("--raw", default="datasets/raw")
    ap.add_argument("--countries", nargs="+", default=["India"], help="names or 'all'")
    ap.add_argument("--include-drone", action="store_true")
    ap.add_argument("--max-side", type=int, default=1280)
    args = ap.parse_args()

    src = Path(args.src)
    by_country = defaultdict(list)
    for xp in src.rglob("*.xml"):
        by_country[country_of(xp)].append(xp)
    print("found:", {c: len(v) for c, v in sorted(by_country.items())})
    wanted = sorted(by_country) if args.countries == ["all"] else args.countries
    if not args.include_drone:
        wanted = [c for c in wanted if c not in TOP_DOWN or args.countries != ["all"]]
    report = {}

    for country in wanted:
        xmls = sorted(by_country.get(country, []))
        if not xmls:
            print(f"{country}: no XML found under {src}")
            continue
        images = {p.stem: p for p in (src / country).rglob("*") if p.suffix.lower() in IMG_EXTS}
        out = Path(args.raw) / f"rdd_{country.lower()}"
        n_pos = n_neg = n_box = missing = 0
        for xp in xmls:
            fname, w, h, boxes = parse_voc(xp)
            img = images.get(Path(fname).stem) or images.get(xp.stem)
            if img is None:
                missing += 1
                continue
            dst = out / "images" / f"{img.stem}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                if max(w, h) > args.max_side or img.suffix.lower() not in {".jpg", ".jpeg"}:
                    small, _ = resize_max(load_bgr(img), args.max_side)
                    cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(dst))
                else:
                    shutil.copy2(img, dst)
            write_yolo(out / "labels" / f"{img.stem}.txt", boxes, w, h)   # normalised -> size-independent
            n_box += len(boxes)
            n_pos += bool(boxes)
            n_neg += not boxes
        report[country] = {"pothole_images": n_pos, "negative_images": n_neg,
                           "pothole_boxes": n_box, "xml_without_image": missing}
        save_json(out / "summary.json", report[country])
        print(f"{country:>16}: {report[country]} -> {out}")
    save_json(Path(args.raw) / "rdd2022_summary.json", report)


if __name__ == "__main__":
    main()
