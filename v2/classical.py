"""Classical stage: the fast first filter shared by H1 and H2.

Pipeline (each step is switched on by an ablation-row flag in config.py):
    resize (<=640 px) -> grey or LAB-L + shadow flattening + CLAHE
    -> fused score S (Method A, optionally 1/4-res blurs)
    -> road-region mask -> threshold max(percentile, tau_abs)
    -> connected components -> boxes + per-box score (mean S)
    -> GrabCut tightening -> merge touching / IoU>0.3 boxes -> NMS
    -> global gate features (max S, top-k mean S, component count)

If your existing Method A / B code differs, replace `fused_score` and/or
`propose` - everything downstream only needs (boxes, scores, S, features).
"""
from __future__ import annotations

import time

import cv2
import numpy as np

CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ----------------------------------------------------------------- I/O
def load_bgr(path) -> np.ndarray:
    """cv2.imread fails on non-ASCII Windows paths; imdecode does not."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def resize_max(img: np.ndarray, max_side) -> tuple[np.ndarray, float]:
    """Cap the long side at max_side. Returns (image, scale = new/old)."""
    if not max_side:
        return img, 1.0
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1.0:
        return img, 1.0
    size = (max(1, round(w * s)), max(1, round(h * s)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA), s


# ----------------------------------------------------------------- features
def to_intensity(img: np.ndarray, normalize: bool) -> np.ndarray:
    """float32 intensity in [0, 1]. normalize=True is roadmap precision fix 5."""
    if not normalize:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
    h, w = L.shape
    f = 4                                                    # background is low-frequency:
    small = cv2.resize(L, (w // f, h // f), interpolation=cv2.INTER_AREA)   # close at 1/4 res
    k = max(5, (min(small.shape) // 8) | 1)                  # large, odd kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    background = cv2.resize(cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel), (w, h),
                            interpolation=cv2.INTER_LINEAR)
    flat = L.astype(np.float32) / np.maximum(background.astype(np.float32), 1.0)
    flat_u8 = np.clip(flat * 255.0, 0, 255).astype(np.uint8)
    return CLAHE.apply(flat_u8).astype(np.float32) / 255.0


def _blur_terms(I, blurred):
    """Dark term (pothole darker than surroundings) and bright term (water-filled
    pothole reflecting the sky, brighter than surroundings)."""
    dark = sum(np.maximum(0, b - I) for b in blurred) / len(blurred)
    bright = sum(np.maximum(0, I - b) for b in blurred) / len(blurred)
    return dark, bright


def fused_score_fast(I: np.ndarray, f: int = 4, bright_weight: float = 0.0,
                     dark_weight: float = 0.3) -> np.ndarray:
    """Roadmap Goal 3: the three large blurs run at 1/f resolution.
    S = dark_weight * (D_dark + bright_weight * D_bright) + (1 - dark_weight) * E.
    Defaults (bright 0, dark 0.3) reproduce the roadmap's Method A exactly."""
    h, w = I.shape
    small = cv2.resize(I, (w // f, h // f), interpolation=cv2.INTER_AREA)
    blurred = [cv2.resize(cv2.GaussianBlur(small, (0, 0), s / f), (w, h)) for s in (15, 30, 50)]
    dark, bright = _blur_terms(I, blurred)
    D = dark + bright_weight * bright if bright_weight else dark
    M = cv2.magnitude(cv2.Sobel(I, cv2.CV_32F, 1, 0), cv2.Sobel(I, cv2.CV_32F, 0, 1))
    M = cv2.resize(M, (w // f, h // f), interpolation=cv2.INTER_AREA)
    E = cv2.resize(cv2.GaussianBlur(M, (0, 0), 15 / f), (w, h))
    return (dark_weight * D + (1 - dark_weight) * E).astype(np.float32)


def fused_score_full(I: np.ndarray, bright_weight: float = 0.0, dark_weight: float = 0.3) -> np.ndarray:
    """Original Method A (row 0): blurs at full resolution."""
    blurred = [cv2.GaussianBlur(I, (0, 0), s) for s in (15, 30, 50)]
    dark, bright = _blur_terms(I, blurred)
    D = dark + bright_weight * bright if bright_weight else dark
    M = cv2.magnitude(cv2.Sobel(I, cv2.CV_32F, 1, 0), cv2.Sobel(I, cv2.CV_32F, 0, 1))
    E = cv2.GaussianBlur(M, (0, 0), 15)
    return (dark_weight * D + (1 - dark_weight) * E).astype(np.float32)


def fused_score(I: np.ndarray, cfg: dict) -> np.ndarray:
    bw, dw = cfg.get("bright_weight", 0.0), cfg.get("dark_weight", 0.3)
    if cfg["fast_blur"]:
        return fused_score_fast(I, cfg["blur_factor"], bw, dw)
    return fused_score_full(I, bw, dw)


def road_mask(h: int, w: int, enabled: bool, horizon: float = 0.45) -> np.ndarray:
    """Fixed trapezoid below the horizon (precision fix 4, first version)."""
    if not enabled:
        return np.ones((h, w), dtype=bool)
    top = int(h * horizon)
    pts = np.array([[int(0.30 * w), top], [int(0.70 * w), top],
                    [w - 1, h - 1], [0, h - 1]], dtype=np.int32)
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


# ----------------------------------------------------------------- boxes
def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a (N,4) and b (M,4), xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


def propose(S: np.ndarray, mask: np.ndarray, cfg: dict):
    """Threshold S and return (boxes xyxy, scores, n_components, threshold)."""
    h, w = S.shape
    sample = S[::4, ::4][mask[::4, ::4]]
    thr = float(np.percentile(sample, cfg["percentile"])) if sample.size else float(S.max())
    if cfg.get("tau_abs") is not None:
        thr = max(thr, float(cfg["tau_abs"]))         # absolute floor (fix 3)
    binary = ((S > thr) & mask).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), 0, thr

    sums = np.bincount(labels.ravel(), weights=S.ravel(), minlength=n)
    area = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    mean_s = sums / np.maximum(area, 1)
    bw, bh = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    aspect = bw / np.maximum(bh, 1)
    ok = ((area >= cfg["min_area_frac"] * h * w) & (area <= cfg["max_area_frac"] * h * w)
          & (aspect > 0.2) & (aspect < 6.0))
    ok[0] = False                                       # background label
    idx = np.flatnonzero(ok)
    n_comp = int(idx.size)
    idx = idx[np.argsort(-mean_s[idx])][: cfg["max_candidates"]]
    x, y = stats[idx, cv2.CC_STAT_LEFT], stats[idx, cv2.CC_STAT_TOP]
    boxes = np.stack([x, y, x + bw[idx], y + bh[idx]], axis=1).astype(np.float32)
    return boxes, mean_s[idx].astype(np.float32), n_comp, thr


def grabcut_tighten(img: np.ndarray, box: np.ndarray, iters: int = 2) -> np.ndarray:
    """Refit one box to its GrabCut foreground (precision fix 2)."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 8:
        return box
    pad = int(0.25 * max(bw, bh)) + 2
    rx1, ry1, rx2, ry2 = max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)
    roi = img[ry1:ry2, rx1:rx2]
    s = min(1.0, 160.0 / max(roi.shape[:2]))           # keep GrabCut cheap
    if s < 1.0:
        roi = cv2.resize(roi, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    rect = (int((x1 - rx1) * s), int((y1 - ry1) * s), max(2, int(bw * s)), max(2, int(bh * s)))
    if rect[2] >= roi.shape[1] - 1 and rect[3] >= roi.shape[0] - 1:
        return box                                      # no background to learn from
    gc_mask = np.zeros(roi.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(roi, gc_mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return box
    fg = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
    if fg.sum() < 0.1 * rect[2] * rect[3]:
        return box
    ys, xs = np.nonzero(fg)
    return np.array([rx1 + xs.min() / s, ry1 + ys.min() / s,
                     rx1 + (xs.max() + 1) / s, ry1 + (ys.max() + 1) / s], dtype=np.float32)


def merge_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.3, touching: bool = True):
    """Union boxes whose IoU > iou_thr or (touching=True) that touch (fixes Method B fragments)."""
    n = len(boxes)
    if n < 2:
        return boxes, scores
    iou = box_iou(boxes, boxes)
    gap_x = np.maximum(boxes[:, None, 0], boxes[None, :, 0]) - np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    gap_y = np.maximum(boxes[:, None, 1], boxes[None, :, 1]) - np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    link = (iou > iou_thr) | (((gap_x <= 1) & (gap_y <= 1)) if touching else False)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.nonzero(np.triu(link, 1))):
        parent[find(i)] = find(j)
    roots = np.array([find(i) for i in range(n)])
    out_b, out_s = [], []
    for r in np.unique(roots):
        g = roots == r
        b = boxes[g]
        out_b.append([b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()])
        out_s.append(scores[g].max())
    return np.array(out_b, np.float32), np.array(out_s, np.float32)


def drop_oversize(boxes: np.ndarray, scores: np.ndarray, h: int, w: int, max_frac: float):
    """Remove boxes that grew past max_frac of the image (e.g. after merging)."""
    if not len(boxes):
        return boxes, scores
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = area <= max_frac * h * w
    return boxes[keep], scores[keep]


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5):
    """cv2.dnn.NMSBoxes (precision fix 6)."""
    if len(boxes) < 2:
        return boxes, scores
    xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
    keep = cv2.dnn.NMSBoxes(xywh, scores.tolist(), 0.0, iou_thr)
    keep = np.array(keep).reshape(-1)
    return boxes[keep], scores[keep]


def gate_features(S: np.ndarray, mask: np.ndarray, n_comp: int) -> np.ndarray:
    """Global statistics of S for the H3 presence test: max, top-k mean, count."""
    vals = S[::2, ::2][mask[::2, ::2]]
    if vals.size == 0:
        return np.zeros(3, np.float32)
    k = max(1, int(0.005 * vals.size))                  # top 0.5 % of pixels
    topk = np.partition(vals, vals.size - k)[-k:]
    return np.array([vals.max(), topk.mean(), n_comp], dtype=np.float32)


# ----------------------------------------------------------------- driver
def run_classical(img_bgr: np.ndarray, cfg: dict) -> dict:
    """Run the classical stage. Boxes are in the RESIZED image's coordinates;
    divide by out['scale'] to map them back to the original image."""
    t = {}
    t0 = time.perf_counter()
    img, scale = resize_max(img_bgr, cfg["max_side"])
    I = to_intensity(img, cfg["normalize"])
    t["preprocess"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    S = fused_score(I, cfg)
    t["features"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mask = road_mask(*S.shape, cfg["road_mask"], cfg["horizon"])
    boxes, scores, n_comp, thr = propose(S, mask, cfg)
    feats = gate_features(S, mask, n_comp)
    t["proposals"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if cfg["grabcut"] and len(boxes):
        boxes = np.stack([grabcut_tighten(img, b) for b in boxes])
    if cfg.get("merge", cfg["grabcut"]) and len(boxes):
        boxes, scores = merge_boxes(boxes, scores, cfg["merge_iou"], cfg.get("merge_touching", True))
    boxes, scores = drop_oversize(boxes, scores, *img.shape[:2], cfg.get("max_box_frac", 1.0))
    boxes, scores = nms(boxes, scores, cfg["nms_iou"])
    t["refine"] = time.perf_counter() - t0

    return {"img": img, "scale": scale, "S": S, "mask": mask, "boxes": boxes,
            "scores": scores, "features": feats, "n_components": n_comp,
            "threshold": thr, "timings": {k: v * 1000 for k, v in t.items()}}
