"""Fit the shared gate: tau_abs (pixel floor) + logistic presence test (H3).

  tau_abs   = 99th percentile of S on VALIDATION pothole-free images
  presence  = LogisticRegression on [max S, top-k mean S, n_components] of
              TRAIN images with and without potholes
  threshold = chosen on VALIDATION so >= --min-recall of pothole images pass

Rows 6, 7 and 9 share one gate file (same S recipe), so H1 and H2 are gated
identically. Rows 2-5 need their own: python fit_gate.py --row 2 (3, 4, 5).

    python fit_gate.py --row 7
"""
import argparse
import json

import numpy as np

from classical import fused_score, load_bgr, resize_max, road_mask, run_classical, to_intensity
from config import DATASET, ROOT, gate_path, row_config
from data_utils import label_for, load_split
from gate import Gate, fit_presence, fit_tau_abs, pick_threshold


def has_pothole(img_path) -> int:
    lbl = label_for(img_path)
    return int(lbl.exists() and lbl.read_text().strip() != "")


def score_map(img_path, cfg):
    img, _ = resize_max(load_bgr(img_path), cfg["max_side"])
    S = fused_score(to_intensity(img, cfg["normalize"]), cfg)
    return S, road_mask(*S.shape, cfg["road_mask"], cfg["horizon"])


def features(paths, cfg):
    X = np.stack([run_classical(load_bgr(p), cfg)["features"] for p in paths])
    y = np.array([has_pothole(p) for p in paths])
    return X, y


def subsample(paths, n, rng):
    return list(rng.choice(paths, size=min(n, len(paths)), replace=False)) if paths else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=int, default=7)
    ap.add_argument("--data", default=str(DATASET))
    ap.add_argument("--max-images", type=int, default=1500, help="per split, for speed")
    ap.add_argument("--min-recall", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sources", nargs="+", default=["auto"],
                    help="raw sources used to fit the presence test; 'auto' = eval_sources in splits.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cfg = row_config(args.row)
    cfg.update(use_gate=True, tau_abs=None)

    val = load_split(args.data, "val")
    train = load_split(args.data, "train")
    sources = args.sources
    if sources == ["auto"]:
        sj = ROOT / "splits.json"
        sources = json.loads(sj.read_text()).get("rules", {}).get("eval_sources") if sj.exists() else None
    if sources and sources != ["all"]:                 # fit on the deployment domain only
        train = [p for p in train if p.name.split("__")[0] in sources]
        print(f"presence test fitted on train images from {sources}: {len(train)} images")
    val_neg = [p for p in val if not has_pothole(p)]
    if not val_neg:
        raise SystemExit("No pothole-free images in val - add negatives (RDD2022 images without D40).")

    maps = [score_map(p, cfg) for p in subsample(val_neg, args.max_images, rng)]
    tau = fit_tau_abs([m[0] for m in maps], [m[1] for m in maps], q=99.0)
    cfg["tau_abs"] = tau
    print(f"tau_abs (99th pct of S on {len(maps)} val negatives) = {tau:.4f}")

    X_tr, y_tr = features(subsample(train, args.max_images, rng), cfg)
    if len(set(y_tr)) < 2:
        raise SystemExit("Train needs images both with and without potholes.")
    model = fit_presence(X_tr, y_tr)
    X_va, y_va = features(subsample(val, args.max_images, rng), cfg)
    thr = pick_threshold(model, X_va, y_va, args.min_recall)

    gate = Gate(tau, model, thr)
    passed = np.array([gate.decide(x, int(x[2]))[0] for x in X_va])
    rec = passed[y_va == 1].mean() if (y_va == 1).any() else float("nan")
    skip = 1 - passed[y_va == 0].mean() if (y_va == 0).any() else float("nan")
    out = gate_path(cfg)
    gate.save(out, row=args.row, val_image_recall=float(rec), val_negatives_skipped=float(skip))
    print(f"presence threshold={thr:.3f}  val pothole images passed={rec:.3f}  "
          f"val empty images skipped={skip:.3f}\nsaved {out}")


if __name__ == "__main__":
    main()
