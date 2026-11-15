"""Train the H2 detector (YOLO11n from COCO weights) on the frozen split.

    python train_yolo.py --model yolo11n.pt --epochs 100          # Colab T4
    python train_yolo.py --model yolov8n.pt --epochs 100          # comparison

    python train_yolo.py --resume /content/drive/MyDrive/cs_runs/yolo11n/weights/last.pt

Equivalent CLI:
    yolo detect train model=yolo11n.pt data=datasets/craterscope/data.yaml imgsz=640 epochs=100 batch=32
"""
import argparse
import shutil
from pathlib import Path

from config import DATASET, MODELS
from data_utils import write_data_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt", help="COCO-pretrained start weights")
    ap.add_argument("--data", default=str(DATASET))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0", help="'0' for the Colab GPU, 'cpu' otherwise")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--project", default="runs", help="on Colab use a Drive folder so runs survive disconnects")
    ap.add_argument("--resume", default=None, help="path to last.pt to resume an interrupted run")
    args = ap.parse_args()

    from ultralytics import YOLO

    data_yaml = write_data_yaml(Path(args.data))
    name = Path(args.model).stem
    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(data=str(data_yaml), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
                    device=args.device, patience=args.patience, workers=2, seed=0, deterministic=True,
                    project=args.project, name=name, exist_ok=True, plots=True)
    best = Path(model.trainer.best)
    MODELS.mkdir(parents=True, exist_ok=True)
    dst = MODELS / f"{name}.pt"
    shutil.copy2(best, dst)
    print(f"best weights -> {dst}")

    metrics = YOLO(str(dst)).val(data=str(data_yaml), split="val", imgsz=args.imgsz, device=args.device)
    print(f"val mAP@0.5 {metrics.box.map50:.4f}  mAP@0.5:0.95 {metrics.box.map:.4f}  "
          f"P {metrics.box.mp:.4f}  R {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
