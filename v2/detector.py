"""H1: AI crop verifier (plugs in after contour filtering).

Each classical candidate box is cropped with a +10% margin, resized to
128x128 and classified pothole / not, so shadows, patches and lane markings
are dropped. All crops of one image go through ONE batched forward pass.

Verifiers:
  mnv3 - torchvision MobileNetV3-small, ImageNet weights, 2-class head.
         Runs from .onnx / _int8.onnx through ONNX Runtime (no torch needed),
         or from the .pt checkpoint through PyTorch.
  svm  - LBP + HOG + linear SVM (ablation row 6, the non-deep baseline).

Class index convention (torchvision ImageFolder, alphabetical):
  0 = "not", 1 = "pothole".
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

CLASSES = ["not", "pothole"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# ----------------------------------------------------------------- crops
def expand_box(box, margin: float, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    dx, dy = margin * (x2 - x1), margin * (y2 - y1)
    return (int(max(0, x1 - dx)), int(max(0, y1 - dy)),
            int(min(w, np.ceil(x2 + dx))), int(min(h, np.ceil(y2 + dy))))


def crop_box(img_bgr: np.ndarray, box, size: int = 128, margin: float = 0.10) -> np.ndarray:
    """One BGR uint8 crop (size x size). Used for BOTH training data and inference."""
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = expand_box(box, margin, w, h)
    crop = img_bgr[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def crops_to_tensor(crops: list[np.ndarray]) -> np.ndarray:
    """BGR uint8 crops -> NCHW float32, RGB, ImageNet-normalised."""
    x = np.stack(crops)[..., ::-1].astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


# ----------------------------------------------------------------- MobileNetV3
def build_mnv3(pretrained: bool = True):
    """torchvision MobileNetV3-small with a 2-class head."""
    import torch.nn as nn
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASSES))
    return model


class MobileNetVerifier:
    """Loads .onnx (ONNX Runtime) or .pt (PyTorch). predict() = one batched call."""

    def __init__(self, path: Path, size: int = 128, margin: float = 0.10, threads: int = 0):
        self.path, self.size, self.margin = Path(path), size, margin
        if self.path.suffix == ".onnx":
            import onnxruntime as ort
            so = ort.SessionOptions()
            if threads:
                so.intra_op_num_threads = threads
            self.sess = ort.InferenceSession(str(self.path), so, providers=["CPUExecutionProvider"])
            self.input_name = self.sess.get_inputs()[0].name
            self.torch_model = None
        else:
            import torch
            self.torch_model = build_mnv3(pretrained=False)
            self.torch_model.load_state_dict(torch.load(self.path, map_location="cpu"))
            self.torch_model.eval()
            self.sess = None

    def predict(self, img_bgr: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """P(pothole) for each box."""
        if len(boxes) == 0:
            return np.zeros(0, np.float32)
        x = crops_to_tensor([crop_box(img_bgr, b, self.size, self.margin) for b in boxes])
        if self.sess is not None:
            logits = self.sess.run(None, {self.input_name: x})[0]
        else:
            import torch
            with torch.inference_mode():
                logits = self.torch_model(torch.from_numpy(x)).numpy()
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return (e[:, 1] / e.sum(axis=1)).astype(np.float32)


# ----------------------------------------------------------------- LBP/HOG SVM
def lbp_hog_features(crop_bgr: np.ndarray) -> np.ndarray:
    from skimage.feature import hog, local_binary_pattern

    g = cv2.cvtColor(cv2.resize(crop_bgr, (64, 64)), cv2.COLOR_BGR2GRAY)
    h = hog(g, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    lbp = local_binary_pattern(g, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    return np.concatenate([h, lbp_hist]).astype(np.float32)


class SvmVerifier:
    def __init__(self, path: Path, size: int = 128, margin: float = 0.10):
        import joblib
        self.model = joblib.load(path)
        self.size, self.margin = size, margin

    def predict(self, img_bgr: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return np.zeros(0, np.float32)
        X = np.stack([lbp_hog_features(crop_box(img_bgr, b, self.size, self.margin)) for b in boxes])
        return self.model.predict_proba(X)[:, 1].astype(np.float32)


# ----------------------------------------------------------------- glue
def load_verifier(kind: str, cfg: dict):
    """Pick the fastest available file: INT8 ONNX > FP32 ONNX > .pt."""
    from config import SVM_JOBLIB, VERIFIER_INT8, VERIFIER_ONNX, VERIFIER_PT

    size, margin = cfg["verifier_size"], cfg["verifier_margin"]
    if kind == "svm":
        if not SVM_JOBLIB.exists():
            raise FileNotFoundError(f"{SVM_JOBLIB} missing - run train_verifier.py --kind svm")
        return SvmVerifier(SVM_JOBLIB, size, margin)
    for p in (cfg.get("verifier_path"), VERIFIER_INT8, VERIFIER_ONNX, VERIFIER_PT):
        if p and Path(p).exists():
            return MobileNetVerifier(Path(p), size, margin)
    raise FileNotFoundError("No MobileNetV3 verifier in models/ - run train_verifier.py")


def verify(img_bgr: np.ndarray, boxes: np.ndarray, verifier, threshold: float):
    """Keep boxes with P(pothole) >= threshold. The verifier probability
    becomes the box confidence (precision fix 6: ranking for PR/mAP)."""
    probs = verifier.predict(img_bgr, boxes)
    keep = probs >= threshold
    return boxes[keep], probs[keep], probs
