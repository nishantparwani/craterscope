"""Shared gate: absolute floor tau_abs + logistic-regression presence test (H3).

Both H1 and H2 call `Gate.decide` with the SAME classical features, so the
ablation compares the AI stages behind an identical gate.

  tau_abs  = 99th percentile of S on validation images WITHOUT potholes
             (pixel-level floor applied inside classical.propose)
  presence = sklearn LogisticRegression on [max S, top-k mean S, n_components]
             trained on train images with and without potholes; its decision
             threshold is chosen on validation for high image-level recall.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = ["s_max", "s_topk_mean", "n_components"]


class Gate:
    def __init__(self, tau_abs: float | None = None, model=None, threshold: float = 0.5):
        self.tau_abs = tau_abs
        self.model = model
        self.threshold = threshold

    @classmethod
    def load(cls, path: Path) -> "Gate":
        d = joblib.load(path)
        return cls(d["tau_abs"], d["model"], d["threshold"])

    def save(self, path: Path, **extra) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"tau_abs": self.tau_abs, "model": self.model,
                     "threshold": self.threshold, "features": FEATURE_NAMES, **extra}, path)

    def presence(self, features: np.ndarray) -> float:
        if self.model is None:
            return 1.0
        return float(self.model.predict_proba(features.reshape(1, -1))[0, 1])

    def decide(self, features: np.ndarray, n_boxes: int) -> tuple[bool, float]:
        """'Strong enough?' diamond of the cascade diagram.
        Returns (run_ai, presence_probability)."""
        p = self.presence(features)
        return (n_boxes > 0 and p >= self.threshold), p


def fit_tau_abs(S_maps: list[np.ndarray], masks: list[np.ndarray], q: float = 99.0) -> float:
    """99th percentile of S pooled over pothole-free validation images."""
    pooled = np.concatenate([S[::4, ::4][m[::4, ::4]] for S, m in zip(S_maps, masks)])
    return float(np.percentile(pooled, q))


def fit_presence(X: np.ndarray, y: np.ndarray):
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(class_weight="balanced", max_iter=1000))
    model.fit(X, y)
    return model


def pick_threshold(model, X_val: np.ndarray, y_val: np.ndarray, min_recall: float = 0.98) -> float:
    """Highest threshold that still passes >= min_recall of pothole images."""
    p = model.predict_proba(X_val)[:, 1]
    pos = np.sort(p[y_val == 1])
    if pos.size == 0:
        return 0.5
    idx = int(np.floor((1.0 - min_recall) * pos.size))
    return float(pos[min(idx, pos.size - 1)])
