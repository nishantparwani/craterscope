"""Tune the classical stage on VALIDATION so its boxes actually land on potholes.

Grid search over how S is built and how proposals are cut, scored against the
labelled validation boxes at IoU 0.5 (roadmap rule 2: tune on validation only).

  --group precision    rows 0-5 (classical-only): maximise F1 of the classical boxes
  --group high_recall  rows 6, 7, 9 (H1 proposals + the gate shared with H2):
                       maximise proposal recall@0.5 with at most --max-boxes per image

Searched (cheap stages only; GrabCut is checked once at the end for high_recall):
  bright_weight  0 / 0.5 / 1   bright term for water-filled potholes
  dark_weight    0.3 / 0.6 / 0.9   share of the blob term vs the edge term in S
  horizon        0.35 / 0.45 / 0.55   top of the road trapezoid
  percentile     60 / 70 / 80 / 85 / 90 / 95
  min_area_frac  0.0002 / 0.0005 / 0.001,  max_area_frac 0.05 / 0.25
  max_candidates 8 / 15
  high_recall only: merge off / IoU>0.3 / touching,  max_box_frac 0.08 / 1.0

    python tune_classical.py --group high_recall
    python tune_classical.py --group precision
Writes the winner to models/tuned.json ("classical" block) and the full table to
results/tune_classical_<group>.json.
"""
import argparse
import itertools
import json
import time

import cv2
import numpy as np

from classical import (box_iou, drop_oversize, fused_score, grabcut_tighten, load_bgr, merge_boxes,
                       nms, resize_max, road_mask, to_intensity)
from config import DATASET, RESULTS, row_config, save_classical
from data_utils import label_for, load_split, read_yolo

S_GRID = {"bright_weight": [0.0, 0.5, 1.0], "dark_weight": [0.3, 0.6, 0.9]}
HORIZONS = [0.35, 0.45, 0.55]
PERCENTILES = [60, 70, 80, 85, 90, 95]
MIN_AREA = [0.0002, 0.0005, 0.001]
MAX_AREA = [0.05, 0.25]
MAX_CAND = [8, 15]
MERGE = ["off", "iou", "touching"]
MAX_BOX = [0.08, 1.0]


def components(S, mask, pct):
    """Threshold S at a percentile inside the mask -> per-blob (box, area, mean S)."""
    sample = S[::4, ::4][mask[::4, ::4]]
    thr = float(np.percentile(sample, pct)) if sample.size else float(S.max())
    binary = ((S > thr) & mask).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.zeros((0, 4), np.float32), np.zeros(0), np.zeros(0)
    sums = np.bincount(labels.ravel(), weights=S.ravel(), minlength=n)
    area = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    bw, bh = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    aspect = bw / np.maximum(bh, 1)
    ok = (aspect > 0.2) & (aspect < 6.0)
    ok[0] = False
    idx = np.flatnonzero(ok)
    x, y = stats[idx, cv2.CC_STAT_LEFT], stats[idx, cv2.CC_STAT_TOP]
    boxes = np.stack([x, y, x + bw[idx], y + bh[idx]], 1).astype(np.float32)
    return boxes, area[idx], (sums / np.maximum(area, 1))[idx]


def score_boxes(boxes, gt):
    """(gt covered @0.5, gt covered @0.3, boxes hitting a gt @0.5)."""
    if not len(gt):
        return 0, 0, 0
    if not len(boxes):
        return 0, 0, 0
    iou = box_iou(boxes, gt)
    best_gt = iou.max(0)
    return int((best_gt >= 0.5).sum()), int((best_gt >= 0.3).sum()), int((iou.max(1) >= 0.5).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["precision", "high_recall"], required=True)
    ap.add_argument("--data", default=str(DATASET))
    ap.add_argument("--images", type=int, default=300, help="validation images with potholes")
    ap.add_argument("--negatives", type=int, default=100, help="pothole-free validation images")
    ap.add_argument("--max-boxes", type=float, default=8.0, help="high_recall: average boxes per image")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    from config import DEFAULTS, HIGH_RECALL, TUNED_JSON
    base = row_config(4 if args.group == "precision" else 7)   # S recipe of the rows being tuned
    # "before" = the untuned roadmap settings, whatever an earlier tuning run saved
    for k in ("bright_weight", "dark_weight", "horizon", "min_area_frac", "max_area_frac",
              "merge_touching", "max_box_frac"):
        base[k] = DEFAULTS[k]
    base.pop("merge", None)
    base["percentile"] = HIGH_RECALL["percentile"] if args.group == "high_recall" else DEFAULTS["percentile"]
    base["max_candidates"] = HIGH_RECALL["max_candidates"] if args.group == "high_recall" else DEFAULTS["max_candidates"]
    if args.group == "high_recall":
        base["grabcut"] = True
    rng = np.random.default_rng(0)
    val = load_split(args.data, "val")
    pos = [p for p in val if label_for(p).exists() and label_for(p).read_text().strip()]
    posset = set(pos)
    neg = [p for p in val if p not in posset]
    pick = list(rng.permutation(pos)[: args.images]) + list(rng.permutation(neg)[: args.negatives])
    print(f"tuning '{args.group}' on {min(len(pos), args.images)} pothole + "
          f"{min(len(neg), args.negatives)} clean validation images")

    merges = MERGE if args.group == "high_recall" else ["off"]
    maxbox = MAX_BOX if args.group == "high_recall" else [1.0]
    post = list(itertools.product(MIN_AREA, MAX_AREA, MAX_CAND, merges, maxbox))
    keys = list(itertools.product(S_GRID["bright_weight"], S_GRID["dark_weight"], HORIZONS, PERCENTILES))
    acc = {}                                    # combo -> [cov05, cov03, box_hits, n_boxes]
    n_gt = 0
    t0 = time.time()
    for i, p in enumerate(pick):
        img0 = load_bgr(p)
        img, s = resize_max(img0, base["max_side"])
        gt = read_yolo(label_for(p), img0.shape[1], img0.shape[0]) * s
        n_gt += len(gt)
        I = to_intensity(img, base["normalize"])
        h, w = I.shape
        for bwt, dwt in itertools.product(S_GRID["bright_weight"], S_GRID["dark_weight"]):
            S = fused_score(I, {**base, "bright_weight": bwt, "dark_weight": dwt})
            for hz in HORIZONS:
                mask = road_mask(h, w, base["road_mask"], hz)
                for pct in PERCENTILES:
                    cb, carea, cmean = components(S, mask, pct)
                    for mn, mx, mc, mg, mb in post:
                        ok = (carea >= mn * h * w) & (carea <= mx * h * w)
                        b, sc = cb[ok], cmean[ok]
                        order = np.argsort(-sc)[:mc]
                        b, sc = b[order], sc[order].astype(np.float32)
                        if mg != "off" and len(b):
                            b, sc = merge_boxes(b, sc, base["merge_iou"], touching=(mg == "touching"))
                        b, sc = drop_oversize(b, sc, h, w, mb)
                        b, sc = nms(b, sc, base["nms_iou"])
                        c5, c3, bh = score_boxes(b, gt)
                        a = acc.setdefault((bwt, dwt, hz, pct, mn, mx, mc, mg, mb), [0, 0, 0, 0])
                        a[0] += c5
                        a[1] += c3
                        a[2] += bh
                        a[3] += len(b)
        if i % 25 == 0:
            print(f"  {i}/{len(pick)} images ({time.time() - t0:.0f}s)")

    n_img = len(pick)
    rows = []
    for k, (c5, c3, bh, nb) in acc.items():
        rec5, rec3 = c5 / max(n_gt, 1), c3 / max(n_gt, 1)
        prec = min(bh, c5) / max(nb, 1)
        f1 = 2 * prec * rec5 / max(prec + rec5, 1e-9)
        rows.append({"bright_weight": k[0], "dark_weight": k[1], "horizon": k[2], "percentile": k[3],
                     "min_area_frac": k[4], "max_area_frac": k[5], "max_candidates": k[6],
                     "merge": k[7], "max_box_frac": k[8], "recall@0.5": round(rec5, 4),
                     "recall@0.3": round(rec3, 4), "precision": round(prec, 4), "f1": round(f1, 4),
                     "boxes_per_image": round(nb / n_img, 2)})
    if args.group == "precision":
        rows.sort(key=lambda r: (-r["f1"], -r["recall@0.5"]))
    else:
        fit = [r for r in rows if r["boxes_per_image"] <= args.max_boxes]
        rows = sorted(fit, key=lambda r: (-r["recall@0.5"], r["boxes_per_image"])) + \
               sorted([r for r in rows if r not in fit], key=lambda r: -r["recall@0.5"])
    best = dict(rows[0])

    base_merge = ("touching" if base.get("merge", base["grabcut"]) and base["merge_touching"] else
                  "iou" if base.get("merge", base["grabcut"]) else "off") if args.group == "high_recall" else "off"
    default = [r for r in rows if r["bright_weight"] == 0 and r["dark_weight"] == 0.3 and
               r["merge"] == base_merge and r["max_box_frac"] == 1.0 and
               r["horizon"] == base["horizon"] and r["percentile"] == base["percentile"] and
               r["min_area_frac"] == base["min_area_frac"] and r["max_area_frac"] == base["max_area_frac"]
               and r["max_candidates"] == min(MAX_CAND, key=lambda m: abs(m - base["max_candidates"]))]
    print(f"\n{len(rows)} settings scored on {n_gt} pothole boxes ({time.time() - t0:.0f}s). Top 8:")
    hdr = ["bright_weight", "dark_weight", "horizon", "percentile", "min_area_frac", "max_area_frac",
           "max_candidates", "merge", "max_box_frac", "recall@0.5", "recall@0.3", "precision", "f1",
           "boxes_per_image"]
    print(" | ".join(h.replace("_frac", "").replace("_weight", "") for h in hdr))
    for r in rows[:8]:
        print(" | ".join(str(r[h]) for h in hdr))
    if default:
        d = default[0]
        print(f"\nbefore (current defaults, no GrabCut): recall@0.5 {d['recall@0.5']}  "
              f"recall@0.3 {d['recall@0.3']}  precision {d['precision']}  boxes/img {d['boxes_per_image']}")
    print(f"after  (best):                           recall@0.5 {best['recall@0.5']}  "
          f"recall@0.3 {best['recall@0.3']}  precision {best['precision']}  boxes/img {best['boxes_per_image']}")

    chosen = {k: best[k] for k in ("bright_weight", "dark_weight", "horizon", "percentile",
                                   "min_area_frac", "max_area_frac", "max_candidates")}
    if args.group == "high_recall":
        chosen.update(merge=best["merge"] != "off", merge_touching=best["merge"] == "touching",
                      max_box_frac=best["max_box_frac"])
        # GrabCut is slow, so it is checked once, on the winner, on 120 images
        check = pick[:120]
        cfg = {**base, **chosen, "use_gate": False, "tau_abs": None}
        res = {}
        for gc in (False, True):
            hit = tot = 0
            for p in check:
                img0 = load_bgr(p)
                img, s = resize_max(img0, cfg["max_side"])
                gt = read_yolo(label_for(p), img0.shape[1], img0.shape[0]) * s
                I = to_intensity(img, cfg["normalize"])
                S = fused_score(I, cfg)
                mask = road_mask(*S.shape, cfg["road_mask"], cfg["horizon"])
                cb, carea, cmean = components(S, mask, cfg["percentile"])
                h, w = S.shape
                ok = (carea >= cfg["min_area_frac"] * h * w) & (carea <= cfg["max_area_frac"] * h * w)
                b, sc = cb[ok], cmean[ok]
                order = np.argsort(-sc)[: cfg["max_candidates"]]
                b, sc = b[order], sc[order].astype(np.float32)
                if gc and len(b):
                    b = np.stack([grabcut_tighten(img, x) for x in b])
                if cfg["merge"] and len(b):
                    b, sc = merge_boxes(b, sc, cfg["merge_iou"], cfg["merge_touching"])
                b, sc = drop_oversize(b, sc, h, w, cfg["max_box_frac"])
                hit += score_boxes(b, gt)[0]
                tot += len(gt)
            res[gc] = hit / max(tot, 1)
        chosen["grabcut"] = bool(res[True] > res[False])
        print(f"GrabCut check on 120 images: recall@0.5 off {res[False]:.3f} vs on {res[True]:.3f} "
              f"-> grabcut={chosen['grabcut']}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"tune_classical_{args.group}.json").write_text(json.dumps(
        {"chosen": chosen, "best": best, "before": default[0] if default else None,
         "n_gt": n_gt, "images": n_img, "top": rows[:30]}, indent=2))
    if not args.no_save:
        save_classical(args.group, chosen)
        print(f"saved classical '{args.group}' settings -> models/tuned.json: {chosen}")


if __name__ == "__main__":
    main()
