"""Command-line inference through ONNX Runtime (no Streamlit).

    python infer.py --image road.jpg --row 7            # H1 (classical + MobileNetV3 INT8)
    python infer.py --image photos\\ --row 9 --out out\\  # H2 on a folder (gate + YOLO11n INT8)

Writes annotated images to --out: green = ground truth (if labelled), red = detections,
yellow = classical proposals (--proposals). Prints boxes and per-stage times.
"""
import argparse
from pathlib import Path

import cv2

from cascade import Cascade, draw_boxes
from classical import load_bgr
from config import row_config
from data_utils import label_for, list_images, read_yolo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="image file or folder")
    ap.add_argument("--row", type=int, default=7, help="ablation row: 5 classical, 7 H1, 9 H2")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--proposals", action="store_true", help="also draw classical proposals (yellow)")
    ap.add_argument("--only-potholes", action="store_true", help="skip images without ground-truth potholes")
    args = ap.parse_args()

    cascade = Cascade(row_config(args.row))
    src = Path(args.image)
    paths = list_images(src) if src.is_dir() else [src]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cascade.run(load_bgr(paths[0]))                      # warm-up
    for p in paths:
        if args.only_potholes:
            try:
                if not label_for(p).exists() or not label_for(p).read_text().strip():
                    continue
            except ValueError:
                pass
        img = load_bgr(p)
        r = cascade.run(img)
        vis = img
        try:                                         # ground truth in green when a label exists
            gt = read_yolo(label_for(p), img.shape[1], img.shape[0])
            vis = draw_boxes(vis, gt, color=(0, 200, 0), thickness=2)
        except ValueError:
            gt = []
        if args.proposals and len(r["proposals"]):
            vis = draw_boxes(vis, r["proposals"], color=(0, 220, 255), thickness=1)
        vis = draw_boxes(vis, r["boxes"], r["scores"])
        cv2.imencode(".jpg", vis)[1].tofile(str(out_dir / f"{p.stem}_det.jpg"))
        stages = "  ".join(f"{k}={v:.1f}" for k, v in r["timings"].items())
        print(f"{p.name}: {len(r['boxes'])} pothole(s), gate={'open' if r['gate_open'] else 'closed'} | ms: {stages}")
        for b, s in zip(r["boxes"], r["scores"]):
            print(f"    box x1={b[0]:.0f} y1={b[1]:.0f} x2={b[2]:.0f} y2={b[3]:.0f}  score={s:.2f}")


if __name__ == "__main__":
    main()
