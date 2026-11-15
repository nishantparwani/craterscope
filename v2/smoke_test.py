"""End-to-end smoke test on a tiny SYNTHETIC dataset (no downloads, ~1-2 min on CPU).

Checks that every script runs in order on your machine before you spend time on
real data. Numbers it prints are meaningless; only "PASS" matters.

    python smoke_test.py            # keeps outputs in .smoke/ for inspection
"""
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
TMP = ROOT / ".smoke"


def synth_image(rng, n_potholes, shadow, size=720):
    h = w = size
    img = np.full((h, w), 112, np.float32) + cv2.GaussianBlur(rng.normal(0, 18, (h, w)).astype(np.float32), (0, 0), 1.2)
    img[: int(0.38 * h)] = 185 + rng.normal(0, 4, (int(0.38 * h), w))          # sky
    for y in range(int(0.45 * h), h, 80):                                      # dashed lane
        cv2.line(img, (w // 2, y), (w // 2 + 6, y + 40), 235, 8)
    if shadow:
        pts = np.array([[rng.integers(0, w // 2), int(0.5 * h)], [rng.integers(w // 2, w), int(0.55 * h)],
                        [rng.integers(w // 2, w), h - 1], [rng.integers(0, w // 3), h - 1]])
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [pts], 1)
        img[m > 0] *= 0.62
    boxes = []
    for _ in range(n_potholes):
        rx, ry = int(rng.integers(18, 50)), int(rng.integers(10, 30))
        cx, cy = int(rng.integers(int(0.3 * w), int(0.7 * w))), int(rng.integers(int(0.6 * h), h - 60))
        m = np.zeros((h, w), np.uint8)
        cv2.ellipse(m, (cx, cy), (rx, ry), float(rng.uniform(-20, 20)), 0, 360, 1, -1)
        m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=int(rng.integers(0, 3)))
        img[m > 0] = img[m > 0] * 0.45 + rng.normal(0, 6, int(m.sum()))
        ys, xs = np.nonzero(m)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    bgr = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return bgr, boxes


def write_voc(path, fname, w, h, boxes, other=()):
    ann = ET.Element("annotation")
    ET.SubElement(ann, "filename").text = fname
    size = ET.SubElement(ann, "size")
    ET.SubElement(size, "width").text, ET.SubElement(size, "height").text = str(w), str(h)
    for name, b in [("D40", b) for b in boxes] + list(other):
        o = ET.SubElement(ann, "object")
        ET.SubElement(o, "name").text = name
        bb = ET.SubElement(o, "bndbox")
        for k, v in zip(("xmin", "ymin", "xmax", "ymax"), b):
            ET.SubElement(bb, k).text = str(int(v))
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(ann).write(path)


def make_raw(n=90, seed=0, country="India"):
    rng = np.random.default_rng(seed)
    base = TMP / "downloads" / "rdd2022" / country / "train"
    for i in range(n):
        n_p = int(rng.integers(1, 3)) if i % 3 else 0                          # 1/3 negatives
        img, boxes = synth_image(rng, n_p, shadow=bool(rng.random() < 0.4))
        name = f"{country}_{i:06d}.jpg"
        (base / "images").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(base / "images" / name), img)
        write_voc(base / "annotations" / "xmls" / f"{country}_{i:06d}.xml", name, 720, 720, boxes,
                  other=[("D00", [10, 500, 30, 700])])
    if country != "India":
        return
    # two near-duplicates to exercise the dedup step
    for j, src in enumerate((1, 2)):
        im = cv2.imread(str(base / "images" / f"India_{src:06d}.jpg"))
        name = f"India_dup{j}.jpg"
        cv2.imwrite(str(base / "images" / name), cv2.GaussianBlur(im, (3, 3), 0))
        shutil.copy(base / "annotations" / "xmls" / f"India_{src:06d}.xml",
                    base / "annotations" / "xmls" / f"India_dup{j}.xml")
        x = base / "annotations" / "xmls" / f"India_dup{j}.xml"
        x.write_text(x.read_text().replace(f"India_{src:06d}.jpg", name))


def run(*args):
    cmd = [sys.executable, *map(str, args)]
    print("\n$ python", " ".join(map(str, args)))
    r = subprocess.run(cmd, cwd=ROOT, env=ENV)
    if r.returncode:
        raise SystemExit(f"FAILED: {' '.join(map(str, args))}")


ENV = dict(os.environ, CRATERSCOPE_MODELS=str(TMP / "models"), CRATERSCOPE_RESULTS=str(TMP / "results"),
           PYTHONIOENCODING="utf-8")


def unit_checks():
    sys.path.insert(0, str(ROOT))
    os.environ.update(CRATERSCOPE_MODELS=ENV["CRATERSCOPE_MODELS"])
    from ai_detector import YoloOnnx, letterbox
    from cascade import Cascade, VideoCascade
    from config import row_config

    # YOLO head decoding without a model: one confident box at the image centre
    dec = YoloOnnx.__new__(YoloOnnx)
    dec.conf, dec.iou = 0.25, 0.5
    out = np.zeros((1, 5, 8400), np.float32)
    out[0, :, 0] = [320, 320, 100, 60, 0.9]
    out[0, :, 1] = [322, 321, 100, 60, 0.8]                                    # duplicate -> NMS
    b, s, c = dec._decode(out)
    assert len(b) == 1 and abs(b[0, 0] - 270) < 1e-3, (b, s)
    lb, r, pad = letterbox(np.zeros((360, 640, 3), np.uint8), 640)
    assert lb.shape == (640, 640, 3) and pad == (0, 140), pad
    print("YOLO decode + letterbox: PASS")

    # Video cascade with a stand-in tracker (real one: trackers / supervision ByteTrack)
    class Det:
        def __init__(self, xyxy, conf):
            self.xyxy, self.confidence = xyxy, conf
            self.tracker_id = np.arange(1, len(xyxy) + 1)
    cascade = Cascade(row_config(6))
    vc = VideoCascade(cascade, k=2, tracker_fn=lambda d: d, sv_converter=Det, min_hits=1)
    rng = np.random.default_rng(1)
    for idx in range(6):
        frame, _ = synth_image(rng, 1, False)
        o = vc.step(frame, idx)
        assert o["ran_ai"] == (idx % 2 == 0)
    print(f"VideoCascade k=2: AI frames {vc.stats['ai_frames']}, gated out {vc.stats['gated_out']}: PASS")


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    make_raw()
    make_raw(n=30, seed=7, country="Japan")                                   # a train-only country
    ds, raw, crops = TMP / "ds", TMP / "raw", TMP / "crops"
    src = TMP / "downloads" / "rdd2022"
    run("prep_rdd2022.py", "--src", src, "--raw", raw, "--countries", "all")
    run("prep_split.py", "--raw", raw, "--out", ds, "--splits", TMP / "splits.json", "--neg-holdout", 3, "--group-hamming", 4,
        "--eval-sources", "rdd_india")
    for split in ("val", "test", "neg_test"):                                   # India only in eval splits
        bad = [p.name for p in (ds / "images" / split).glob("*") if not p.name.startswith("rdd_india__")]
        assert not bad, (split, bad[:3])
    run("prep_rdd_india.py", "--src", src, "--out", TMP / "raw_india" / "rdd_india")   # laptop: India only
    run("prep_split.py", "--raw", TMP / "raw_india", "--out", TMP / "ds_rebuilt", "--from-splits",
        TMP / "splits.json", "--build", "val", "test", "neg_test")
    for row in (2, 3, 4, 7):
        run("fit_gate.py", "--row", row, "--data", ds)
    run("prep_crops.py", "--row", 7, "--data", ds, "--out", crops)
    run("train_verifier.py", "--kind", "svm", "--crops", crops)
    for row in (0, 1, 2, 5):
        run("evaluate.py", "--row", row, "--split", "val", "--data", ds, "--warmup", 1)
    run("evaluate.py", "--row", 6, "--split", "val", "--data", ds, "--tune", "--warmup", 1)
    run("evaluate.py", "--row", 5, "--split", "test", "--data", ds, "--warmup", 1)
    run("evaluate.py", "--row", 6, "--split", "test", "--data", ds, "--warmup", 1)
    run("evaluate.py", "--compare", 5, 6, "--split", "test")
    unit_checks()
    print("\nSMOKE TEST PASS - outputs in", TMP)


if __name__ == "__main__":
    main()
