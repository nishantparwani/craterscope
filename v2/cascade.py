"""The CraterScope 2.0 hybrid cascade (image path + H3 AI-gated video path).

    frame -> classical proposals (shared gate) -> "strong enough?"
          -> no : 0 boxes, next frame
          -> yes: H1 MobileNetV3 verifier  |  H2 YOLO11n
    video : AI every k-th frame, ByteTrack IDs, boxes carried in between

Kept free of Streamlit so it can be unit-tested and timed from evaluate.py.
"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from classical import resize_max, run_classical
from config import gate_path
from gate import Gate

EMPTY_BOXES = np.zeros((0, 4), np.float32)
EMPTY_SCORES = np.zeros(0, np.float32)


class Cascade:
    """One configured pipeline (one ablation row)."""

    def __init__(self, cfg: dict, require_gate: bool = True):
        self.cfg = dict(cfg)
        self.gate = None
        self.warnings = []
        if self.cfg["use_gate"]:
            p = gate_path(self.cfg)
            if p.exists():
                self.gate = Gate.load(p)
                self.cfg["tau_abs"] = self.gate.tau_abs
            elif require_gate:
                raise FileNotFoundError(f"{p.name} missing - run: python fit_gate.py --row {self.cfg['row']}")
            else:
                self.warnings.append(f"No fitted gate ({p.name}); using 'any proposal' as the gate.")
        self.verifier = None
        self.yolo = None
        if self.cfg["mode"] == "h1":
            from detector import load_verifier
            self.verifier = load_verifier(self.cfg["verifier"], self.cfg)
        if self.cfg["mode"] == "yolo":
            from ai_detector import load_yolo
            self.yolo = load_yolo(self.cfg["yolo_weights"], self.cfg["yolo_imgsz"],
                                  self.cfg["yolo_conf"], self.cfg["yolo_iou"])

    # ------------------------------------------------------------------ image
    def run(self, img_bgr: np.ndarray) -> dict:
        cfg = self.cfg
        t_start = time.perf_counter()
        if cfg["mode"] == "yolo" and not cfg["use_gate"]:
            return self._run_yolo_only(img_bgr, t_start)

        c = run_classical(img_bgr, cfg)
        timings = dict(c["timings"])
        img_s, scale = c["img"], c["scale"]
        proposals, prop_scores = c["boxes"], c["scores"]

        gate_open, presence = True, None
        if cfg["use_gate"]:
            t0 = time.perf_counter()
            if self.gate is not None:
                gate_open, presence = self.gate.decide(c["features"], len(proposals))
            else:
                gate_open = len(proposals) > 0
            timings["gate"] = (time.perf_counter() - t0) * 1000

        boxes, scores = EMPTY_BOXES, EMPTY_SCORES
        if gate_open and cfg["mode"] == "classical":
            boxes, scores = proposals, prop_scores
        elif gate_open and cfg["mode"] == "h1" and len(proposals):
            from detector import verify
            t0 = time.perf_counter()
            boxes, scores, _ = verify(img_s, proposals, self.verifier, cfg["verifier_threshold"])
            timings["verifier"] = (time.perf_counter() - t0) * 1000
        elif gate_open and cfg["mode"] == "yolo":
            t0 = time.perf_counter()
            boxes, scores, _ = self.yolo(img_s)
            timings["yolo"] = (time.perf_counter() - t0) * 1000

        timings["total"] = (time.perf_counter() - t_start) * 1000
        return {"boxes": boxes / scale, "scores": scores, "proposals": proposals / scale,
                "proposal_scores": prop_scores, "gate_open": gate_open, "presence": presence,
                "features": c["features"], "S": c["S"], "mask": c["mask"], "scale": scale,
                "timings": timings}

    def _run_yolo_only(self, img_bgr, t_start) -> dict:
        t0 = time.perf_counter()
        img_s, scale = resize_max(img_bgr, self.cfg["max_side"])
        timings = {"preprocess": (time.perf_counter() - t0) * 1000}
        t0 = time.perf_counter()
        boxes, scores, _ = self.yolo(img_s)
        timings["yolo"] = (time.perf_counter() - t0) * 1000
        timings["total"] = (time.perf_counter() - t_start) * 1000
        return {"boxes": boxes / scale, "scores": scores, "proposals": EMPTY_BOXES,
                "proposal_scores": EMPTY_SCORES, "gate_open": True, "presence": None,
                "features": None, "S": None, "mask": None, "scale": scale, "timings": timings}


# ====================================================================== video
def make_tracker(frame_rate: float):
    """ByteTrack from Roboflow `trackers` (current) or `supervision` (<0.30).
    Returns a function: sv.Detections -> sv.Detections with tracker_id."""
    try:
        from trackers import ByteTrackTracker
        try:
            return ByteTrackTracker(frame_rate=frame_rate).update
        except TypeError:
            return ByteTrackTracker().update
    except ImportError:
        pass
    import supervision as sv
    if not hasattr(sv, "ByteTrack"):
        raise ImportError("Install the tracker: pip install trackers")
    return sv.ByteTrack(frame_rate=max(1, int(round(frame_rate)))).update_with_detections


def to_sv(boxes: np.ndarray, scores: np.ndarray):
    import supervision as sv
    return sv.Detections(xyxy=boxes.astype(np.float32).reshape(-1, 4),
                         confidence=scores.astype(np.float32),
                         class_id=np.zeros(len(boxes), dtype=int))


class VideoCascade:
    """H3: gate every k-th frame, AI only on gated frames, ByteTrack carries IDs.

    On frames between AI runs each track is moved by its last per-frame
    velocity (constant-velocity carry), so no model runs at all on them.
    """

    def __init__(self, cascade: Cascade, k: int = 2, fps: float = 30.0, min_hits: int = 2,
                 tracker_fn=None, sv_converter=to_sv):
        self.cascade, self.k = cascade, max(1, int(k))
        self.track = tracker_fn or make_tracker(fps / self.k)
        self.to_det = sv_converter
        self.tracks = {}                     # id -> {"box", "vel", "hits", "score"}
        self.counted = set()
        self.min_hits = min_hits
        self.stats = {"frames": 0, "ai_frames": 0, "gated_out": 0}

    def step(self, frame: np.ndarray, idx: int) -> dict:
        self.stats["frames"] += 1
        t0 = time.perf_counter()
        if idx % self.k != 0:
            for t in self.tracks.values():
                t["box"] = t["box"] + t["vel"]
            return self._out(t0, ran_ai=False)

        r = self.cascade.run(frame)
        self.stats["ai_frames"] += int(r["gate_open"])
        self.stats["gated_out"] += int(not r["gate_open"])
        dets = self.track(self.to_det(r["boxes"], r["scores"]))
        alive = {}
        confs = dets.confidence if dets.confidence is not None else np.zeros(len(dets.xyxy))
        for box, tid, conf in zip(dets.xyxy, dets.tracker_id, confs):
            tid = int(tid)
            prev = self.tracks.get(tid)
            vel = (box - prev["box"]) / self.k if prev is not None else np.zeros(4, np.float32)
            hits = (prev["hits"] + 1) if prev is not None else 1
            alive[tid] = {"box": box.astype(np.float32), "vel": vel, "hits": hits,
                          "score": float(conf)}
            if hits >= self.min_hits:
                self.counted.add(tid)
        self.tracks = alive
        return self._out(t0, ran_ai=True, result=r)

    def _out(self, t0, ran_ai, result=None) -> dict:
        return {"tracks": {i: t["box"].copy() for i, t in self.tracks.items()},
                "ran_ai": ran_ai, "result": result, "ms": (time.perf_counter() - t0) * 1000,
                "unique_potholes": len(self.counted)}


def threaded_frames(path: Path, maxsize: int = 8):
    """Decode in a separate thread (roadmap speed fix) and yield frames."""
    q: queue.Queue = queue.Queue(maxsize=maxsize)
    cap = cv2.VideoCapture(str(path))

    def reader():
        while True:
            ok, frame = cap.read()
            q.put(frame if ok else None)
            if not ok:
                break
        cap.release()

    threading.Thread(target=reader, daemon=True).start()
    while True:
        frame = q.get()
        if frame is None:
            return
        yield frame


def video_fps(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps if fps > 1 else 30.0


# ====================================================================== drawing
def draw_boxes(img: np.ndarray, boxes, scores=None, color=(0, 0, 255), ids=None,
               thickness: int = 2) -> np.ndarray:
    out = img.copy()
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = []
        if ids is not None:
            label.append(f"#{ids[i]}")
        if scores is not None and len(scores) > i:
            label.append(f"{scores[i]:.2f}")
        if label:
            cv2.putText(out, " ".join(label), (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)
    return out


def heatmap(S: np.ndarray) -> np.ndarray:
    s = cv2.normalize(S, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(s, cv2.COLORMAP_INFERNO)
