"""Fill one ablation row: precision, recall, F1, mAP, FPPI, latency (+95% CIs).

    python evaluate.py --row 7 --split val --tune     # pick verifier threshold on VAL
    python evaluate.py --row 7 --split test           # final number, run ONCE
    python evaluate.py --compare 5 7 --split test     # paired bootstrap: does 7 beat 5?

Rules implemented (roadmap "Experiment protocol"):
  * IoU 0.5 greedy matching by score; P/R/F1 + 95% bootstrap CI (1000 image resamples)
  * FPPI on the pothole-free hold-out (neg_test)
  * proposal recall @0.3/@0.5 (the H1 upper limit) for classical/H1 rows
  * latency: median and p95 of the full cascade after warm-up, batch 1
  * COCO mAP@0.5 and mAP@0.5:0.95 with pycocotools (if installed)
  * every test-split run is logged to results/test_runs.log (count of configs tried)
"""
import argparse
import csv
import importlib.metadata as md
import json
import platform
import time
from pathlib import Path

import numpy as np

from cascade import Cascade
from classical import box_iou, load_bgr
from config import DATASET, RESULTS, TUNED_JSON, row_config, save_tuned
from data_utils import label_for, load_split, read_yolo


def match(pred, scores, gt, thr=0.5):
    """Greedy one-to-one matching. Returns (tp flags per pred sorted by score, n_gt)."""
    order = np.argsort(-scores)
    pred = pred[order]
    iou = box_iou(pred, gt)
    used = np.zeros(len(gt), bool)
    tp = np.zeros(len(pred), bool)
    for i in range(len(pred)):
        if not len(gt):
            break
        cand = np.where(~used, iou[i], -1)
        j = int(cand.argmax())
        if cand[j] >= thr:
            used[j] = tp[i] = True
    return tp, scores[order]


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def bootstrap_ci(per_img, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.array(per_img)                       # (N, 3) tp, fp, fn
    stats = []
    for _ in range(n):
        s = a[rng.integers(0, len(a), len(a))].sum(0)
        stats.append(prf(*s))
    lo, hi = np.percentile(stats, [2.5, 97.5], axis=0)
    return {k: (float(lo[i]), float(hi[i])) for i, k in enumerate(["P", "R", "F1"])}


def coco_map(records):
    """records: list of (gt boxes, pred boxes, scores). Returns (mAP50, mAP50:95) or (None, None)."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        return None, None
    images, anns, dets = [], [], []
    for i, (gt, pb, ps) in enumerate(records):
        images.append({"id": i, "width": 1, "height": 1})
        for b in gt:
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": 1, "iscrowd": 0,
                         "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                         "area": float((b[2] - b[0]) * (b[3] - b[1]))})
        for b, s in zip(pb, ps):
            dets.append({"image_id": i, "category_id": 1, "score": float(s),
                         "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]})
    if not dets or not anns:
        return 0.0, 0.0
    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "pothole"}]}
    gt.createIndex()
    ev = COCOeval(gt, gt.loadRes(dets), "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return float(ev.stats[1]), float(ev.stats[0])


def env_info():
    def v(pkg):
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return "-"
    info = {"cpu": platform.processor() or platform.machine(), "os": platform.platform(),
            "python": platform.python_version(), "opencv": v("opencv-python"),
            "onnxruntime": v("onnxruntime"), "torch": v("torch")}
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except ImportError:
        pass
    return info


def run_split(cascade, paths, warmup):
    per_img, records, lat, stage_lat, prop_hits = [], [], [], {}, [0, 0, 0]
    all_tp, all_scores, n_gt_total = [], [], 0
    for k, p in enumerate(paths):
        img = load_bgr(p)
        r = cascade.run(img)
        gt = read_yolo(label_for(p), img.shape[1], img.shape[0])
        tp, sc = match(r["boxes"], r["scores"], gt)
        per_img.append((int(tp.sum()), int((~tp).sum()), int(len(gt) - tp.sum())))
        records.append((gt, r["boxes"], r["scores"]))
        all_tp.append(tp)
        all_scores.append(sc)
        n_gt_total += len(gt)
        if len(gt) and len(r["proposals"]):
            best = box_iou(r["proposals"], gt).max(0)
            prop_hits[0] += int((best >= 0.3).sum())
            prop_hits[1] += int((best >= 0.5).sum())
        prop_hits[2] += len(gt)
        if k >= warmup:
            lat.append(r["timings"]["total"])
            for s, ms in r["timings"].items():
                stage_lat.setdefault(s, []).append(ms)
    return {"per_img": per_img, "records": records, "lat": lat, "stage_lat": stage_lat,
            "tp": np.concatenate(all_tp) if all_tp else np.zeros(0, bool),
            "scores": np.concatenate(all_scores) if all_scores else np.zeros(0),
            "n_gt": n_gt_total, "prop_recall": (prop_hits[0] / max(prop_hits[2], 1),
                                                prop_hits[1] / max(prop_hits[2], 1))}


def tune_threshold(tp, scores, n_gt, min_recall):
    """Best precision with recall >= min_recall (roadmap: R not below baseline 0.110)."""
    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        keep = scores >= t
        ntp = int(tp[keep].sum())
        p, r, f1 = prf(ntp, int(keep.sum()) - ntp, n_gt - ntp)
        rows.append((t, p, r, f1))
        print(f"  thr {t:.2f}  P {p:.3f}  R {r:.3f}  F1 {f1:.3f}")
    ok = [x for x in rows if x[2] >= min_recall]
    if not ok:
        print(f"  no threshold reaches recall >= {min_recall}; choosing best F1 instead")
        ok = rows
    best = max(ok, key=lambda x: (x[1], x[3]))
    return float(best[0])


def paired_bootstrap(a_file, b_file, n=1000, seed=0):
    a, b = np.load(a_file)["per_img"], np.load(b_file)["per_img"]
    assert len(a) == len(b), "rows must be evaluated on the same images"
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(a), len(a))
        diffs.append(prf(*b[idx].sum(0))[2] - prf(*a[idx].sum(0))[2])
    diffs = np.array(diffs)
    return float(np.mean(diffs)), float((diffs <= 0).mean()), np.percentile(diffs, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=int, default=7)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--data", default=str(DATASET))
    ap.add_argument("--tune", action="store_true", help="tune score threshold on VAL")
    ap.add_argument("--min-recall", type=float, default=0.110)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N images")
    ap.add_argument("--yolo", default=None, help="override YOLO weights (e.g. models/yolo11n_416_int8.onnx)")
    ap.add_argument("--verifier-path", default=None, help="override verifier file (FP32 vs INT8)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the score threshold (verifier for H1 rows, YOLO conf for rows 8-9)")
    ap.add_argument("--tag", default="", help="suffix for diagnostic runs, e.g. int8_nogate")
    ap.add_argument("--compare", nargs=2, type=int, default=None, metavar=("ROW_A", "ROW_B"))
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.compare:
        a, b = (RESULTS / f"row{r}_{args.split}.npz" for r in args.compare)
        mean, p, ci = paired_bootstrap(a, b)
        print(f"F1(row {args.compare[1]}) - F1(row {args.compare[0]}) = {mean:+.4f} "
              f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  one-sided p = {p:.4f}")
        return

    if args.split == "test" and args.tune:
        raise SystemExit("Tuning on the test split is not allowed (roadmap rule 2).")
    cfg = row_config(args.row)
    if args.yolo:
        cfg["yolo_weights"] = args.yolo
    if args.verifier_path:
        cfg["verifier_path"] = args.verifier_path
    if args.threshold is not None:
        cfg["verifier_threshold"] = cfg["yolo_conf"] = args.threshold
    if args.tune:                                  # score everything, threshold afterwards
        cfg["verifier_threshold"] = 0.0
        cfg["yolo_conf"] = 0.01
    cascade = Cascade(cfg)
    paths = load_split(args.data, args.split)
    if args.limit:
        paths = paths[: args.limit]
    print(f"row {args.row}: {cfg['label']}  |  {args.split}: {len(paths)} images")

    res = run_split(cascade, paths, args.warmup)

    if args.tune:
        if cfg["mode"] not in ("h1", "yolo"):
            raise SystemExit("--tune applies to verifier (h1) and YOLO rows")
        thr = tune_threshold(res["tp"], res["scores"], res["n_gt"], args.min_recall)
        key = "verifier_threshold" if cfg["mode"] == "h1" else "yolo_conf"
        save_tuned(args.row, {key: thr})
        print(f"saved {key}={thr} for row {args.row} -> {TUNED_JSON}")
        return

    tp, fp, fn = np.array(res["per_img"]).sum(0)
    P, R, F1 = prf(tp, fp, fn)
    ci = bootstrap_ci(res["per_img"])
    map50, map5095 = coco_map(res["records"])

    negs = load_split(args.data, "neg_test")
    fppi = None
    if negs and args.split == "test":
        n_fp = sum(len(cascade.run(load_bgr(p))["boxes"]) for p in negs)
        fppi = n_fp / len(negs)

    lat = np.array(res["lat"]) if res["lat"] else np.array([np.nan])
    stages = {s: float(np.median(v)) for s, v in res["stage_lat"].items()}
    row = {"row": args.row, "config": cfg["label"] + (f" [{args.tag}]" if args.tag else ""), "split": args.split, "images": len(paths),
           "P": round(P, 4), "R": round(R, 4), "F1": round(F1, 4),
           "P_CI": ci["P"], "R_CI": ci["R"], "F1_CI": ci["F1"],
           "mAP50": map50, "mAP50_95": map5095, "FPPI_neg": fppi,
           "proposal_recall@0.3": round(res["prop_recall"][0], 4),
           "proposal_recall@0.5": round(res["prop_recall"][1], 4),
           "median_ms": round(float(np.median(lat)), 2), "p95_ms": round(float(np.percentile(lat, 95)), 2),
           "timed_images": int(len(res["lat"])), "stage_median_ms": stages,
           "threshold": cfg["verifier_threshold"] if cfg["mode"] == "h1" else cfg["yolo_conf"],
           **env_info(), "time": time.strftime("%Y-%m-%d %H:%M")}
    print(json.dumps(row, indent=2, default=str))
    if len(res["lat"]) < 200:
        print("note: roadmap rule 5 wants >= 200 timed images for latency")

    tag = f"_{args.tag}" if args.tag else ""
    np.savez(RESULTS / f"row{args.row}_{args.split}{tag}.npz", per_img=np.array(res["per_img"]))
    (RESULTS / f"row{args.row}_{args.split}{tag}.json").write_text(json.dumps(row, indent=2, default=str))
    out = RESULTS / "ablation.csv"
    new = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow({k: (json.dumps(v) if isinstance(v, (dict, tuple)) else v) for k, v in row.items()})
    if args.split == "test":
        with (RESULTS / "test_runs.log").open("a") as f:
            f.write(f"{row['time']}\trow {args.row}\t{cfg['label']}\tF1={row['F1']}\n")
    print(f"appended to {out}")


if __name__ == "__main__":
    main()
