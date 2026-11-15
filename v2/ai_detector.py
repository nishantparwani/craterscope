"""H2: YOLO11n end-to-end detector (accuracy reference + deployable fast path).

  .onnx (FP32 or INT8)  -> pure ONNX Runtime + NumPy (no torch / ultralytics
                           import on the hot path; this is what you deploy)
  .pt                   -> Ultralytics PyTorch (ablation row 8, FP32 reference)

Export (run once, see export_models.py / README):
  yolo export model=models/yolo11n.pt format=onnx imgsz=640 simplify=True
  yolo export model=models/yolo11n.pt format=onnx imgsz=640 int8=True data=<data.yaml> fraction=0.25
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def letterbox(img: np.ndarray, size: int = 640, color=(114, 114, 114)):
    """Resize keeping aspect, pad to size x size. Returns (img, ratio, (padx, pady))."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else img
    px, py = (size - nw) / 2, (size - nh) / 2
    top, bottom = round(py - 0.1), round(py + 0.1)
    left, right = round(px - 0.1), round(px + 0.1)
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return out, r, (left, top)


class YoloOnnx:
    def __init__(self, path: Path, imgsz: int = 640, conf: float = 0.25, iou: float = 0.5,
                 threads: int = 0):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        fixed = inp.shape[2] if isinstance(inp.shape[2], int) else None
        self.imgsz = fixed or imgsz           # exported static size wins
        self.conf, self.iou = conf, iou

    def __call__(self, img_bgr: np.ndarray):
        """Returns (boxes xyxy in img coords, scores, class_ids)."""
        lb, r, (px, py) = letterbox(img_bgr, self.imgsz)
        x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.input_name: np.ascontiguousarray(x)})[0]
        boxes, scores, cls = self._decode(out)
        if len(boxes):
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / r
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / r
            h, w = img_bgr.shape[:2]
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        return boxes, scores, cls

    def _decode(self, out: np.ndarray):
        out = out[0]
        # End-to-end / nms=True exports: (N, 6) = x1,y1,x2,y2,score,cls
        if out.ndim == 2 and out.shape[-1] == 6 and out.shape[0] != 6:
            keep = out[:, 4] >= self.conf
            d = out[keep]
            return d[:, :4].astype(np.float32), d[:, 4].astype(np.float32), d[:, 5].astype(int)
        # Standard YOLO11 head: (4 + nc, N) = cx,cy,w,h, class scores
        pred = out.T
        cls_scores = pred[:, 4:]
        cls = cls_scores.argmax(1)
        scores = cls_scores[np.arange(len(cls)), cls]
        keep = scores >= self.conf
        pred, scores, cls = pred[keep], scores[keep], cls[keep]
        if not len(pred):
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1).astype(np.float32)
        idx = cv2.dnn.NMSBoxes(np.stack([boxes[:, 0], boxes[:, 1], w, h], 1).tolist(),
                               scores.tolist(), self.conf, self.iou)
        idx = np.array(idx).reshape(-1)
        return boxes[idx], scores[idx].astype(np.float32), cls[idx]


class YoloTorch:
    """Ultralytics .pt inference - FP32 reference row (row 8)."""

    def __init__(self, path: Path, imgsz: int = 640, conf: float = 0.25, iou: float = 0.5):
        from ultralytics import YOLO
        self.model = YOLO(str(path))
        self.imgsz, self.conf, self.iou = imgsz, conf, iou

    def __call__(self, img_bgr: np.ndarray):
        r = self.model.predict(img_bgr, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                               device="cpu", verbose=False)[0]
        b = r.boxes
        return (b.xyxy.cpu().numpy().astype(np.float32), b.conf.cpu().numpy().astype(np.float32),
                b.cls.cpu().numpy().astype(int))


def load_yolo(path, imgsz: int = 640, conf: float = 0.25, iou: float = 0.5):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - train with train_yolo.py and export with export_models.py")
    if path.suffix == ".onnx":
        return YoloOnnx(path, imgsz, conf, iou)
    return YoloTorch(path, imgsz, conf, iou)
