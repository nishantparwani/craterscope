"""Export models for CPU inference through ONNX Runtime (FP32 + INT8).

YOLO11n (H2):
    python export_models.py --yolo models/yolo11n.pt [--imgsz 640|416]
  runs the Ultralytics exports
    model.export(format="onnx", imgsz=640, simplify=True)                         -> FP32
    model.export(format="onnx", imgsz=640, int8=True, data=data.yaml, fraction=0.25) -> INT8
  (native ONNX INT8 needs ultralytics >= 8.4.60; older versions fall back to
  ONNX Runtime static quantisation with calibration images from val).

MobileNetV3 verifier (H1):
    python export_models.py --verifier models/verifier_mnv3.onnx
  static INT8 (QDQ) quantisation with ~300 validation crops for calibration.

INT8 is not automatically faster on every CPU: always compare FP32 vs INT8 with
evaluate.py on your laptop and report both (speed-mAP trade-off).
"""
import argparse
import shutil
from pathlib import Path

import numpy as np

from ai_detector import letterbox
from config import CROPS, DATASET, MODELS
from data_utils import list_images, load_split, write_data_yaml


def ort_int8(fp32: Path, out: Path, arrays, per_channel=True):
    """ONNX Runtime static INT8 (QDQ) quantisation with calibration arrays."""
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    name = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"]).get_inputs()[0].name

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(arrays)

        def get_next(self):
            a = next(self.it, None)
            return None if a is None else {name: a}

    reader = Reader()
    quantize_static(str(fp32), str(out), reader, quant_format=QuantFormat.QDQ, per_channel=per_channel,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    print(f"INT8 (onnxruntime static QDQ) -> {out}")


# ----------------------------------------------------------------- YOLO
def export_yolo(pt: Path, imgsz: int, data_root: Path, fraction: float):
    from ultralytics import YOLO

    suffix = "" if imgsz == 640 else f"_{imgsz}"
    dst_fp32 = MODELS / f"{pt.stem}{suffix}.onnx"
    dst_int8 = MODELS / f"{pt.stem}{suffix}_int8.onnx"
    data_yaml = write_data_yaml(data_root)

    # Export from a private copy of the weights in a scratch folder, so Ultralytics'
    # own output names (<stem>.onnx, <stem>_int8.onnx) never overwrite another size.
    import tempfile
    work = Path(tempfile.mkdtemp(prefix=f"yolo_export_{imgsz}_"))
    work_pt = work / f"{pt.stem}{suffix}.pt"
    shutil.copy2(pt, work_pt)

    model = YOLO(str(work_pt))
    fp32 = Path(model.export(format="onnx", imgsz=imgsz, simplify=True, dynamic=False))
    shutil.move(fp32, dst_fp32)
    print(f"FP32 -> {dst_fp32}")

    try:
        int8 = Path(YOLO(str(work_pt)).export(format="onnx", imgsz=imgsz, int8=True,
                                               data=str(data_yaml), fraction=fraction))
        if "int8" not in int8.name:
            raise RuntimeError("this ultralytics version ignored int8 for ONNX")
        shutil.move(int8, dst_int8)
        print(f"INT8 (ultralytics native) -> {dst_int8}")
    except Exception as e:
        print(f"native ONNX INT8 export unavailable ({e}); falling back to onnxruntime")
        from classical import load_bgr
        import cv2
        imgs = load_split(data_root, "val")[:200]
        arrays = []
        for p in imgs:
            lb, _, _ = letterbox(load_bgr(p), imgsz)
            arrays.append((cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None] / 255.0)
                          .astype(np.float32))
        ort_int8(dst_fp32, dst_int8, arrays)
    shutil.rmtree(work, ignore_errors=True)


# ----------------------------------------------------------------- verifier
def export_verifier(onnx_path: Path, crops_val: Path, n_calib: int = 300):
    from classical import load_bgr
    from detector import crops_to_tensor

    if onnx_path.suffix == ".pt":                # .pt from Colab -> FP32 ONNX first (needs torch)
        import torch
        from detector import build_mnv3
        from train_verifier import export_onnx
        model = build_mnv3(pretrained=False)
        model.load_state_dict(torch.load(onnx_path, map_location="cpu"))
        pt_path, onnx_path = onnx_path, onnx_path.with_suffix(".onnx")
        export_onnx(model.eval(), 128, onnx_path)
        print(f"FP32 {pt_path.name} -> {onnx_path}")
    files = list_images(crops_val)
    if not files:
        raise SystemExit(f"No calibration crops in {crops_val} - run prep_crops.py first")
    rng = np.random.default_rng(0)
    files = list(rng.choice(files, min(n_calib, len(files)), replace=False))
    arrays = [crops_to_tensor([load_bgr(f)]) for f in files]   # crops are already 128x128
    out = onnx_path.with_name(onnx_path.stem + "_int8.onnx")
    ort_int8(onnx_path, out, arrays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", default=None, help="e.g. models/yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--fraction", type=float, default=0.25, help="share of val used to calibrate")
    ap.add_argument("--data", default=str(DATASET))
    ap.add_argument("--verifier", default=None, help="models/verifier_mnv3.pt or models/verifier_mnv3.onnx")
    ap.add_argument("--crops", default=str(CROPS / "val"))
    args = ap.parse_args()
    if args.yolo:
        export_yolo(Path(args.yolo), args.imgsz, Path(args.data), args.fraction)
    if args.verifier:
        export_verifier(Path(args.verifier), Path(args.crops))
    if not (args.yolo or args.verifier):
        ap.print_help()


if __name__ == "__main__":
    main()
