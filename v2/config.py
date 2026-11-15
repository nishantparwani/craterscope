"""CraterScope 2.0 - central configuration.

Every ablation row in the roadmap is one preset dict in ROWS. Each row
changes exactly one thing relative to the row above it, so a gain in
precision or latency can be traced to that change. H1 (rows 6-7) and H2
(row 9) share the SAME classical gate settings (BASE_GATE).

Values marked "tune on val" must only be changed from validation results
(roadmap rule 2). Tuned values are written to models/tuned.json by
evaluate.py --tune and override the defaults here at load time.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = Path(os.environ.get("CRATERSCOPE_MODELS", ROOT / "models"))  # env override for smoke tests
DATASET = Path(os.environ.get("CRATERSCOPE_DATA", ROOT / "datasets" / "craterscope"))  # prep_split.py
CROPS = Path(os.environ.get("CRATERSCOPE_CROPS", ROOT / "datasets" / "crops"))          # prep_crops.py
RESULTS = Path(os.environ.get("CRATERSCOPE_RESULTS", ROOT / "results"))
TUNED_JSON = MODELS / "tuned.json"

# --- model file names (models/ folder) ------------------------------------
VERIFIER_PT = MODELS / "verifier_mnv3.pt"
VERIFIER_ONNX = MODELS / "verifier_mnv3.onnx"
VERIFIER_INT8 = MODELS / "verifier_mnv3_int8.onnx"
SVM_JOBLIB = MODELS / "verifier_svm.joblib"
YOLO_PT = MODELS / "yolo11n.pt"
YOLO_ONNX = MODELS / "yolo11n.onnx"
YOLO_INT8 = MODELS / "yolo11n_int8.onnx"

# --- defaults shared by every row -------------------------------------------
DEFAULTS = {
    # classical stage
    "max_side": None,          # None = full resolution (row 0)
    "fast_blur": False,        # 1/4-resolution Gaussian blurs (Goal 3)
    "blur_factor": 4,
    "percentile": 85,          # relative threshold on S (tune on val)
    "min_area_frac": 0.0005,   # drop blobs smaller than this fraction of image
    "max_area_frac": 0.25,
    "max_candidates": 3,
    "road_mask": False,
    "horizon": 0.45,           # trapezoid top as fraction of height (tune on val)
    "normalize": False,        # background flattening + CLAHE on LAB L
    "grabcut": False,
    "merge_iou": 0.3,
    "nms_iou": 0.5,
    "bright_weight": 0.0,      # >0 adds the bright (water-filled pothole) term to S
    "dark_weight": 0.3,        # S = dark_weight * D + (1 - dark_weight) * E (roadmap: 0.3)
    "merge_touching": True,    # also merge boxes that merely touch
    "max_box_frac": 1.0,       # drop final boxes larger than this fraction of the image
    # gate (absolute floor + logistic presence test)
    "use_gate": False,
    "tau_abs": None,           # filled from models/gate_*.joblib
    "gate_threshold": None,    # filled from models/gate_*.joblib
    # AI stages
    "mode": "classical",       # classical | h1 | yolo
    "verifier": None,          # None | "svm" | "mnv3"
    "verifier_threshold": 0.5, # tune on val (evaluate.py --tune)
    "verifier_margin": 0.10,   # +10% crop margin (roadmap)
    "verifier_size": 128,
    "yolo_weights": None,      # path; .pt = FP32 reference, .onnx = ORT
    "yolo_imgsz": 640,
    "yolo_conf": 0.25,         # tune on val
    "yolo_iou": 0.5,
}

# High-recall proposal mode for H1 ("fix the recall ceiling" in Goal 1).
HIGH_RECALL = {"percentile": 70, "max_candidates": 8}

# Shared classical gate used by BOTH H1 (rows 6-7) and H2 (row 9).
# It equals row 5 + HIGH_RECALL; `python config.py` asserts this.
BASE_GATE = {
    "max_side": 640,
    "fast_blur": True,
    "use_gate": True,
    "road_mask": True,
    "normalize": True,
    "grabcut": True,
    **HIGH_RECALL,
}

ROWS = {
    0: {"label": "Method A as it is now"},
    1: {"label": "+ blurs at 1/4 res, 640 px input", "max_side": 640, "fast_blur": True},
    2: {"label": "+ absolute gate and presence test", "use_gate": True},
    3: {"label": "+ road mask", "road_mask": True},
    4: {"label": "+ shadow normalisation", "normalize": True},
    5: {"label": "+ GrabCut tightening and box merging", "grabcut": True},
    6: {"label": "+ LBP/HOG + SVM verifier", **HIGH_RECALL, "mode": "h1", "verifier": "svm"},
    7: {"label": "+ MobileNetV3 verifier instead of row 6 (H1)", **HIGH_RECALL,
        "mode": "h1", "verifier": "mnv3"},
    8: {"label": "YOLO11n, FP32 (reference)", "mode": "yolo", "use_gate": False,
        "yolo_weights": str(YOLO_PT)},
    9: {"label": "Gate + YOLO11n, ONNX INT8 (H2)", "mode": "yolo",
        "yolo_weights": str(YOLO_INT8)},
}


def row_config(row: int) -> dict:
    """Cumulative config for an ablation row (each row adds to the one above).

    Rows 6-7 build on row 5 (classical gate + proposals). Rows 8-9 build on the
    shared gate (BASE_GATE) so H1 and H2 see identical gating.
    """
    cfg = dict(DEFAULTS)
    if row <= 7:
        for r in range(row + 1):
            if r == 6 and row == 7:  # row 7 REPLACES row 6's verifier
                continue
            cfg.update({k: v for k, v in ROWS[r].items() if k != "label"})
    else:
        cfg.update(BASE_GATE)
        cfg.update({k: v for k, v in ROWS[row].items() if k != "label"})
    cfg["row"] = row
    cfg["label"] = ROWS[row]["label"]
    return apply_tuned(cfg)


def preset(name: str) -> dict:
    """Named presets used by the Streamlit app."""
    table = {"Classical only": 5, "H1: classical + MobileNetV3": 7,
             "H1-SVM: classical + LBP/HOG SVM": 6, "H2: gate + YOLO11n (INT8)": 9,
             "YOLO11n alone (FP32 reference)": 8}
    return row_config(table[name])


# Proposal settings found by tune_classical.py are shared by row groups, so every
# row in a group sees the same classical stage (one change per ablation row):
#   "precision"   -> rows 0-5 (classical-only rows)
#   "high_recall" -> rows 6, 7, 9 (H1 proposals + the gate shared with H2)
CLASSICAL_GROUP = {0: "precision", 1: "precision", 2: "precision", 3: "precision",
                   4: "precision", 5: "precision", 6: "high_recall", 7: "high_recall",
                   9: "high_recall"}


def apply_tuned(cfg: dict) -> dict:
    """Overlay validation-tuned values (models/tuned.json) onto a config."""
    if not TUNED_JSON.exists():
        return cfg
    tuned = json.loads(TUNED_JSON.read_text())
    group = CLASSICAL_GROUP.get(cfg["row"])
    if group:
        cfg.update(tuned.get("classical", {}).get(group, {}))
    cfg.update(tuned.get(f"row{cfg['row']}", {}))
    return cfg


def save_classical(group: str, values: dict) -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    tuned = json.loads(TUNED_JSON.read_text()) if TUNED_JSON.exists() else {}
    tuned.setdefault("classical", {})[group] = values
    TUNED_JSON.write_text(json.dumps(tuned, indent=2))


def save_tuned(row: int, values: dict) -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    tuned = json.loads(TUNED_JSON.read_text()) if TUNED_JSON.exists() else {}
    tuned.setdefault(f"row{row}", {}).update(values)
    TUNED_JSON.write_text(json.dumps(tuned, indent=2))


def gate_signature(cfg: dict) -> str:
    """Gate statistics depend on how S and the proposals are computed, so one gate
    file per recipe. Untuned defaults keep the original short name."""
    base = "gate_s{}_f{}_m{}_n{}_p{}".format(
        cfg["max_side"] or "full", int(cfg["fast_blur"]), int(cfg["road_mask"]),
        int(cfg["normalize"]), cfg["percentile"])
    extra = {k: cfg.get(k, DEFAULTS[k]) for k in
             ("bright_weight", "dark_weight", "min_area_frac", "max_area_frac", "horizon")}
    if extra == {k: DEFAULTS[k] for k in extra}:
        return base
    import hashlib
    return base + "_" + hashlib.md5(json.dumps(extra, sort_keys=True).encode()).hexdigest()[:6]


def gate_path(cfg: dict) -> Path:
    return MODELS / f"{gate_signature(cfg)}.joblib"


if __name__ == "__main__":
    # Ablation sanity check: H1 and H2 must share the same classical gate.
    sig = {r: gate_signature(row_config(r)) for r in ROWS}
    assert sig[6] == sig[7] == sig[9], sig
    for r in ROWS:
        c = row_config(r)
        print(f"row {r}: {c['label']:<48} mode={c['mode']:<9} "
              f"verifier={c['verifier']} gate={c['use_gate']} sig={sig[r]}")
    print("OK: rows 6, 7 and 9 share gate", sig[7])
