"""Train the H1 crop verifier.

  --kind mnv3 : torchvision MobileNetV3-small, ImageNet weights, 128x128,
                class-weighted cross-entropy (row 7). Saves models/verifier_mnv3.pt
                and exports models/verifier_mnv3.onnx (dynamic batch).
  --kind svm  : LBP + HOG + linear SVM (row 6, non-deep ablation).
                Saves models/verifier_svm.joblib

    python train_verifier.py --kind mnv3 --epochs 20
    python train_verifier.py --kind svm
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

from config import CROPS, MODELS, SVM_JOBLIB, VERIFIER_ONNX, VERIFIER_PT
from detector import CLASSES, MEAN, STD, build_mnv3, lbp_hog_features


# ----------------------------------------------------------------- MobileNetV3
def train_mnv3(args):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = args.size
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.2, 0.02)], p=0.8),
        transforms.ToTensor(), transforms.Normalize(MEAN.tolist(), STD.tolist())])
    tf_val = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                                 transforms.Normalize(MEAN.tolist(), STD.tolist())])
    ds_tr = datasets.ImageFolder(Path(args.crops) / "train", tf_train)
    ds_va = datasets.ImageFolder(Path(args.crops) / "val", tf_val)
    assert ds_tr.classes == CLASSES, f"expected folders {CLASSES}, got {ds_tr.classes}"

    workers = 0 if os.name == "nt" else 2       # Windows: no fork, keep it simple
    dl_tr = DataLoader(ds_tr, args.batch, shuffle=True, num_workers=workers, pin_memory=device == "cuda")
    dl_va = DataLoader(ds_va, args.batch, shuffle=False, num_workers=workers)

    counts = np.bincount(ds_tr.targets, minlength=2).astype(np.float32)
    weights = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), device=device)
    print(f"train counts {dict(zip(CLASSES, counts.astype(int)))}  class weights {weights.tolist()}")

    model = build_mnv3(pretrained=True).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    best_ap = -1.0
    MODELS.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        t0, run_loss = time.time(), 0.0
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * len(y)
        sched.step()

        probs, ys = predict_loader(model, dl_va, device)
        ap = average_precision_score(ys, probs)
        f1 = f1_score(ys, probs >= 0.5)
        print(f"epoch {epoch + 1:02d}/{args.epochs} loss {run_loss / len(ds_tr):.4f} "
              f"val AP {ap:.4f} F1@0.5 {f1:.4f} ({time.time() - t0:.0f}s)")
        if ap > best_ap:
            best_ap = ap
            torch.save(model.state_dict(), VERIFIER_PT)
    print(f"best val AP {best_ap:.4f} -> {VERIFIER_PT}")

    model.load_state_dict(torch.load(VERIFIER_PT, map_location="cpu"))
    export_onnx(model.cpu().eval(), size, VERIFIER_ONNX)


def predict_loader(model, loader, device):
    import torch
    model.eval()
    probs, ys = [], []
    with torch.inference_mode():
        for x, y in loader:
            p = torch.softmax(model(x.to(device)), 1)[:, 1]
            probs.append(p.float().cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def export_onnx(model, size: int, out: Path):
    """Dynamic-batch ONNX so one call verifies all crops of an image."""
    import torch
    x = torch.randn(2, 3, size, size)
    kw = dict(input_names=["images"], output_names=["logits"], opset_version=17)
    try:   # TorchScript exporter: deprecated in torch>=2.9 but still the most predictable
        torch.onnx.export(model, (x,), str(out), dynamo=False,
                          dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}}, **kw)
    except Exception as e:
        print(f"TorchScript export failed ({e}); using dynamo exporter")
        batch = torch.export.Dim("batch", min=1, max=256)
        torch.onnx.export(model, (x,), str(out), dynamo=True, external_data=False,
                          dynamic_shapes=({0: batch},), **kw)
    check_onnx(model, out, size)


def check_onnx(model, path: Path, size: int):
    import onnxruntime as ort
    import torch
    x = torch.randn(5, 3, size, size)
    with torch.inference_mode():
        ref = model(x).numpy()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
    print(f"ONNX export {path.name}: max |diff| vs torch = {np.abs(ref - got).max():.2e}")


# ----------------------------------------------------------------- SVM
def load_crop_set(folder: Path, max_per_class: int, rng):
    from classical import load_bgr
    X, y = [], []
    for label, cls in enumerate(CLASSES):
        files = sorted((folder / cls).glob("*.png"))
        if len(files) > max_per_class:
            files = list(rng.choice(files, max_per_class, replace=False))
        X += [lbp_hog_features(load_bgr(f)) for f in files]
        y += [label] * len(files)
    return np.stack(X), np.array(y)


def train_svm(args):
    import joblib
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    rng = np.random.default_rng(0)
    X_tr, y_tr = load_crop_set(Path(args.crops) / "train", args.max_per_class, rng)
    X_va, y_va = load_crop_set(Path(args.crops) / "val", args.max_per_class, rng)
    print(f"SVM train {X_tr.shape}, val {X_va.shape}")
    svm = LinearSVC(C=args.svm_c, class_weight="balanced", max_iter=5000)
    model = make_pipeline(StandardScaler(), CalibratedClassifierCV(svm, cv=3))
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_va)[:, 1]
    print(f"val AP {average_precision_score(y_va, p):.4f}  F1@0.5 {f1_score(y_va, p >= 0.5):.4f}")
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, SVM_JOBLIB)
    print(f"saved {SVM_JOBLIB}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["mnv3", "svm"], default="mnv3")
    ap.add_argument("--crops", default=str(CROPS))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--max-per-class", type=int, default=20000)
    ap.add_argument("--svm-c", type=float, default=0.1)
    args = ap.parse_args()
    train_mnv3(args) if args.kind == "mnv3" else train_svm(args)


if __name__ == "__main__":
    main()
