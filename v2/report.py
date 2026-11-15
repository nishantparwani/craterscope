"""Readable confusion counts for evaluated ablation rows.

Reads results/row<N>_<split>.npz written by evaluate.py and prints:
  * box level (IoU >= 0.5): TP, FP, FN, precision, recall, F1
    (there is no "true negative" box, so box-level accuracy does not exist)
  * image level (does the image contain a pothole?): TP, FP, FN, TN, accuracy,
    precision, recall, specificity

    python report.py --split val                 # every row evaluated on val
    python report.py --rows 7 9 --split test
"""
import argparse

import numpy as np

from config import RESULTS, ROWS


def safe(a, b):
    return a / b if b else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=list(ROWS))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args()

    hdr = (f"{'row':>3} {'config':<44} | {'TP':>5} {'FP':>5} {'FN':>5} {'P':>6} {'R':>6} {'F1':>6} |"
           f" {'imgTP':>5} {'imgFP':>5} {'imgFN':>5} {'imgTN':>5} {'Acc':>6} {'Spec':>6}")
    print(f"split = {args.split}   (box level: IoU >= 0.5;  image level: 'has a pothole?')")
    print(hdr)
    print("-" * len(hdr))
    for r in args.rows:
        f = RESULTS / f"row{r}_{args.split}.npz"
        if not f.exists():
            continue
        a = np.load(f)["per_img"]                       # columns: tp, fp, fn per image
        tp, fp, fn = a.sum(0)
        p, rec = safe(tp, tp + fp), safe(tp, tp + fn)
        f1 = safe(2 * p * rec, p + rec)
        has_gt = (a[:, 0] + a[:, 2]) > 0
        has_pred = (a[:, 0] + a[:, 1]) > 0
        itp = int((has_gt & has_pred).sum())
        ifp = int((~has_gt & has_pred).sum())
        ifn = int((has_gt & ~has_pred).sum())
        itn = int((~has_gt & ~has_pred).sum())
        acc = safe(itp + itn, len(a))
        spec = safe(itn, itn + ifp)
        print(f"{r:>3} {ROWS[r]['label'][:44]:<44} | {tp:>5} {fp:>5} {fn:>5} {p:>6.3f} {rec:>6.3f} {f1:>6.3f} |"
              f" {itp:>5} {ifp:>5} {ifn:>5} {itn:>5} {acc:>6.3f} {spec:>6.3f}")
    print("\nTP = correct pothole box, FP = false alarm, FN = missed pothole, TN = clean image left clean.")


if __name__ == "__main__":
    main()
