"""Copy HRP4K (already YOLO, single class 'pothole') into the raw-source layout.

Images are 3840x2160; they are downscaled to --max-side (default 1280) to save
disk and decode time. YOLO labels are normalised, so they are unchanged.
Images whose label file is missing or empty are kept as pothole-free negatives.

    python prep_hrp4k.py --src downloads/hrp4k --out datasets/raw/hrp4k
"""
import argparse
import shutil
from pathlib import Path

import cv2

from classical import load_bgr, resize_max
from data_utils import label_for, list_images, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="HRP4K root (Zenodo unzip or HF mirror 'data' folder)")
    ap.add_argument("--out", default="datasets/raw/hrp4k")
    ap.add_argument("--max-side", type=int, default=1280)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    n_pos = n_neg = 0
    for img_path in list_images(Path(args.src)):
        try:
            lbl = label_for(img_path)
        except ValueError:                       # no 'images' folder in path
            lbl = img_path.with_suffix(".txt")
        stem = f"hrp4k_{img_path.parent.name}_{img_path.stem}"
        dst_img = out / "images" / f"{stem}.jpg"
        if not dst_img.exists():
            img, _ = resize_max(load_bgr(img_path), args.max_side)
            cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(dst_img))
        dst_lbl = out / "labels" / f"{stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, dst_lbl)
        else:
            dst_lbl.write_text("")
        has = dst_lbl.read_text().strip() != ""
        n_pos += has
        n_neg += not has
    summary = {"pothole_images": n_pos, "negative_images": n_neg}
    save_json(out / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
