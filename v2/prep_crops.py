"""Build the H1 verifier dataset from classical proposals (roadmap Goal 1).

For every train/val image: run the classical stage in high-recall mode
(row 7 settings, same as at inference), crop each candidate with +10% margin,
label it 'pothole' if IoU >= 0.5 with a ground-truth box, else 'not'; then add
every ground-truth box as a 'pothole' crop. Output is torchvision ImageFolder:

    datasets/crops/{train,val}/{not,pothole}/*.png

Also prints PROPOSAL RECALL at IoU 0.3 / 0.5 - the upper limit for H1.

    python prep_crops.py --row 7
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from classical import box_iou, load_bgr, run_classical
from config import CROPS, DATASET, gate_path, row_config
from data_utils import label_for, load_split, read_yolo, save_json
from detector import crop_box
from gate import Gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=int, default=7)
    ap.add_argument("--data", default=str(DATASET))
    ap.add_argument("--out", default=str(CROPS))
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--pos-iou", type=float, default=0.5)
    args = ap.parse_args()

    cfg = row_config(args.row)
    gp = gate_path(cfg)
    cfg["tau_abs"] = Gate.load(gp).tau_abs if gp.exists() else None
    print(f"classical config of row {args.row}; tau_abs={cfg['tau_abs']}")
    size, margin = cfg["verifier_size"], cfg["verifier_margin"]
    summary = {}

    for split in args.splits:
        counts = {"pothole": 0, "not": 0}
        n_gt = hit03 = hit05 = 0
        for p in load_split(args.data, split):
            img = load_bgr(p)
            c = run_classical(img, cfg)
            small, s = c["img"], c["scale"]
            gt = read_yolo(label_for(p), img.shape[1], img.shape[0]) * s
            props = c["boxes"]
            iou = box_iou(props, gt)
            if len(gt):
                best = iou.max(axis=0) if len(props) else np.zeros(len(gt))
                n_gt += len(gt)
                hit03 += int((best >= 0.3).sum())
                hit05 += int((best >= 0.5).sum())
            crops = []
            for i, b in enumerate(props):
                is_pos = len(gt) and iou[i].max() >= args.pos_iou
                crops.append(("pothole" if is_pos else "not", b, f"p{i}"))
            crops += [("pothole", b, f"gt{i}") for i, b in enumerate(gt)]
            for cls, b, tag in crops:
                out = Path(args.out) / split / cls / f"{p.stem}_{tag}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imencode(".png", crop_box(small, b, size, margin))[1].tofile(str(out))
                counts[cls] += 1
        rec = {"proposal_recall@0.3": hit03 / max(n_gt, 1), "proposal_recall@0.5": hit05 / max(n_gt, 1),
               "gt_boxes": n_gt, **counts}
        summary[split] = rec
        print(split, rec)
    save_json(Path(args.out) / "summary.json", summary)


if __name__ == "__main__":
    main()
