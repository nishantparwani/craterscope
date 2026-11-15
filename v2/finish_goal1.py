"""Run every remaining Goal 1 step in order and write results/goal1_report.md.

    python finish_goal1.py                       # everything (several hours on a laptop CPU)
    python finish_goal1.py --clips clips         # + H3 video benchmark (see below)
    python finish_goal1.py --from-step 5         # resume after an interruption

Steps
  1  tune the classical stage on validation (proposal recall@0.5 for H1/H2, F1 for rows 0-5)
  2  refit the gates for the new classical settings (rows 2, 3, 4 and the shared 7/9 gate)
  3  tune YOLO11n's threshold on validation (row 8)
  4  split the gate's effect from INT8's at row 8's threshold; loosen the gate if it blocks
     more than 20% of YOLO's correct boxes
  5  rebuild verifier crops, retrain LBP/HOG SVM + MobileNetV3, export INT8
  6  tune H1 and H2 thresholds on validation (rows 6, 7, 9)
  7  H3: FPS + count error for k = 1, 2, 4 on dash-cam clips (only with --clips)
  8  run every ablation row ONCE on the test split (+ H2 with FP32 for comparison)
  9  paired bootstrap tests between the key rows
  10 write results/goal1_report.md

--clips: a folder of .mp4 files plus counts.csv with lines  file,true_count
         (e.g.  drive01.mp4,7) - count the potholes in each clip by hand once.
Each finished step is recorded in results/goal1_state.json, so a rerun continues.
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from config import CROPS, MODELS, RESULTS, ROWS, TUNED_JSON, YOLO_INT8, YOLO_ONNX, YOLO_PT

PY = sys.executable
STATE = RESULTS / "goal1_state.json"
TEST_ROWS = list(range(10))


# ----------------------------------------------------------------- helpers
def run(*args, allow_fail=False, capture=False):
    cmd = [PY, *map(str, args)]
    print(f"\n$ python {' '.join(map(str, args))}", flush=True)
    if capture:
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout[-3000:], r.stderr[-2000:] if r.returncode else "", flush=True)
    else:
        r = subprocess.run(cmd)
    if r.returncode and not allow_fail:
        raise SystemExit(f"step failed: python {' '.join(map(str, args))}")
    return r


def state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def mark(step, info=None):
    s = state()
    s[str(step)] = {"done": time.strftime("%Y-%m-%d %H:%M"), **(info or {})}
    RESULTS.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=2))


def tuned():
    return json.loads(TUNED_JSON.read_text()) if TUNED_JSON.exists() else {}


def counts(row, split, tag=""):
    f = RESULTS / f"row{row}_{split}{('_' + tag) if tag else ''}.npz"
    if not f.exists():
        return None
    a = np.load(f)["per_img"]
    return [int(x) for x in a.sum(0)]           # tp, fp, fn


def metrics(row, split, tag=""):
    f = RESULTS / f"row{row}_{split}{('_' + tag) if tag else ''}.json"
    return json.loads(f.read_text()) if f.exists() else None


def have_yolo():
    return YOLO_PT.exists() and YOLO_INT8.exists()


# ----------------------------------------------------------------- steps
def step1(a):
    run("tune_classical.py", "--group", "high_recall", "--max-boxes", a.max_boxes)
    run("tune_classical.py", "--group", "precision")


def step2(a):
    for r in (7, 2, 3, 4):
        run("fit_gate.py", "--row", r, "--min-recall", a.gate_recall)


def step3(a):
    if not YOLO_PT.exists():
        print("models/yolo11n.pt missing - skipping YOLO tuning")
        return
    run("evaluate.py", "--row", 8, "--split", "val", "--tune")


def diagnose(t8):
    runs = {"fp32_nogate": (8, None), "int8_nogate": (8, YOLO_INT8),
            "fp32_gate": (9, YOLO_ONNX), "int8_gate": (9, YOLO_INT8)}
    out = {}
    for tag, (row, weights) in runs.items():
        args = ["evaluate.py", "--row", row, "--split", "val", "--threshold", t8, "--tag", "diag_" + tag]
        if weights:
            args += ["--yolo", weights]
        run(*args)
        out[tag] = counts(row, "val", "diag_" + tag)
    return out


def step4(a):
    if not (have_yolo() and YOLO_ONNX.exists()):
        print("YOLO .pt / .onnx / _int8.onnx missing - skipping gate/INT8 diagnosis")
        return {"skipped": True}
    t8 = tuned().get("row8", {}).get("yolo_conf", 0.25)
    d = diagnose(t8)
    base = max(d["fp32_nogate"][0], 1)
    gate_keep, int8_keep = d["fp32_gate"][0] / base, d["int8_nogate"][0] / base
    print(f"\nat threshold {t8}: YOLO alone TP {d['fp32_nogate'][0]} | gate keeps {gate_keep:.0%} | "
          f"INT8 keeps {int8_keep:.0%}")
    history = [{"min_recall": a.gate_recall, "gate_keep": gate_keep}]
    for mr in (0.995, 0.999):
        if gate_keep >= 0.8:
            break
        print(f"gate blocks too many correct detections -> refitting with --min-recall {mr}")
        run("fit_gate.py", "--row", 7, "--min-recall", mr)
        run("evaluate.py", "--row", 9, "--split", "val", "--threshold", t8, "--yolo", YOLO_ONNX,
            "--tag", "diag_fp32_gate")
        d["fp32_gate"] = counts(9, "val", "diag_fp32_gate")
        gate_keep = d["fp32_gate"][0] / base
        history.append({"min_recall": mr, "gate_keep": gate_keep})
    return {"threshold": t8, "counts": d, "gate_keep": gate_keep, "int8_keep": int8_keep,
            "gate_history": history}


def step5(a):
    if CROPS.exists():
        shutil.rmtree(CROPS)                      # crops must match the new proposals
    run("prep_crops.py", "--row", 7)
    run("train_verifier.py", "--kind", "svm")
    if a.skip_mnv3:
        print("--skip-mnv3: keeping the existing MobileNetV3 verifier")
        return
    run("train_verifier.py", "--kind", "mnv3", "--epochs", a.mnv3_epochs, "--batch", 64)
    run("export_models.py", "--verifier", MODELS / "verifier_mnv3.pt")


def step6(a):
    from config import SVM_JOBLIB, VERIFIER_INT8, VERIFIER_ONNX, VERIFIER_PT
    available = {6: SVM_JOBLIB.exists(),
                 7: any(p.exists() for p in (VERIFIER_INT8, VERIFIER_ONNX, VERIFIER_PT)),
                 9: have_yolo()}
    for r in (6, 7, 9):
        if not available[r]:
            print(f"row {r}: model file missing - skipped")
            continue
        run("evaluate.py", "--row", r, "--split", "val", "--tune")


def step7(a):
    if not a.clips:
        print("no --clips folder: H3 video benchmark skipped")
        return {"skipped": True}
    from bench_video import bench_clip
    from cascade import Cascade
    from config import row_config
    folder = Path(a.clips)
    truth = {}
    if (folder / "counts.csv").exists():
        for line in csv.reader((folder / "counts.csv").read_text().splitlines()):
            if len(line) >= 2 and line[1].strip().isdigit():
                truth[line[0].strip()] = int(line[1])
    rows = []
    for row in (9, 7):
        try:
            cascade = Cascade(row_config(row))
        except Exception as e:
            print(f"row {row} not available for video: {e}")
            continue
        for v in sorted(folder.glob("*.mp4")):
            for res in bench_clip(cascade, v, true_count=truth.get(v.name)):
                res["row"] = row
                rows.append(res)
                print(res)
    (RESULTS / "h3_video.json").write_text(json.dumps(rows, indent=2))
    return {"clips": len(list(folder.glob('*.mp4')))}


def step8(a):
    for r in TEST_ROWS:
        if r in (8, 9) and not have_yolo():
            continue
        run("evaluate.py", "--row", r, "--split", "test", allow_fail=True)
    if YOLO_ONNX.exists():
        run("evaluate.py", "--row", 9, "--split", "test", "--yolo", YOLO_ONNX, "--tag", "fp32",
            allow_fail=True)


def step9(a):
    out = {}
    for x, y in ((5, 6), (5, 7), (6, 7), (8, 9), (7, 9)):
        if counts(x, "test") and counts(y, "test"):
            r = run("evaluate.py", "--compare", x, y, "--split", "test", capture=True, allow_fail=True)
            out[f"{x}->{y}"] = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "failed"
    return {"compare": out}


# ----------------------------------------------------------------- report
def fmt(v, nd=3):
    return "–" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def ci(m, k):
    v = m.get(k)
    if isinstance(v, str):
        v = json.loads(v)
    return f"[{v[0]:.3f}, {v[1]:.3f}]" if v else "–"


def step10(a):
    st = state()
    L = ["# CraterScope 2.0 - Goal 1 (Integrate AI): final results", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} by finish_goal1.py.", ""]

    # classical tuning
    L += ["## 1. Classical stage after tuning (validation)", ""]
    for g in ("high_recall", "precision"):
        f = RESULTS / f"tune_classical_{g}.json"
        if f.exists():
            t = json.loads(f.read_text())
            b, af = t.get("before") or {}, t["best"]
            L += [f"**{g}** ({t['images']} images, {t['n_gt']} pothole boxes): recall@0.5 "
                  f"{fmt(b.get('recall@0.5'))} -> **{fmt(af['recall@0.5'])}**, recall@0.3 "
                  f"{fmt(b.get('recall@0.3'))} -> {fmt(af['recall@0.3'])}, boxes per image "
                  f"{fmt(b.get('boxes_per_image'), 1)} -> {fmt(af['boxes_per_image'], 1)}.  ",
                  f"Chosen: `{json.dumps(t['chosen'])}`", ""]

    # gate vs INT8
    d = st.get("4", {})
    if d.get("counts"):
        c = d["counts"]
        L += ["## 2. Gate vs INT8 (validation, threshold " + fmt(d["threshold"], 2) + ")", "",
              "| Configuration | TP | FP | FN |", "| --- | --- | --- | --- |"]
        for k, lab in (("fp32_nogate", "YOLO11n FP32, no gate"), ("int8_nogate", "YOLO11n INT8, no gate"),
                       ("fp32_gate", "Gate + YOLO11n FP32"), ("int8_gate", "Gate + YOLO11n INT8")):
            if c.get(k):
                L.append(f"| {lab} | {c[k][0]} | {c[k][1]} | {c[k][2]} |")
        L += ["", f"The gate keeps {d['gate_keep']:.0%} of YOLO's correct boxes (after "
              f"{len(d['gate_history']) - 1} refit(s)); INT8 keeps {d['int8_keep']:.0%}.", ""]

    # test ablation
    L += ["## 3. Ablation on the test split (each row run once)", "",
          "| # | Configuration | TP | FP | FN | Precision [95% CI] | Recall [95% CI] | F1 | mAP@0.5 | "
          "FPPI (pothole-free) | Proposal recall@0.5 | Median ms | p95 ms |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r, tag in [(r, "") for r in TEST_ROWS] + [(9, "fp32")]:
        m, c = metrics(r, "test", tag), counts(r, "test", tag)
        if not m or not c:
            continue
        L.append(f"| {r}{'*' if tag else ''} | {m['config']} | {c[0]} | {c[1]} | {c[2]} | "
                 f"{fmt(m['P'])} {ci(m, 'P_CI')} | {fmt(m['R'])} {ci(m, 'R_CI')} | {fmt(m['F1'])} | "
                 f"{fmt(m.get('mAP50'))} | {fmt(m.get('FPPI_neg'))} | {fmt(m.get('proposal_recall@0.5'))} | "
                 f"{fmt(m.get('median_ms'), 1)} | {fmt(m.get('p95_ms'), 1)} |")
    L += ["", "\\* H2 with the FP32 ONNX model instead of INT8, for comparison.", ""]

    comp = st.get("9", {}).get("compare", {})
    if comp:
        L += ["## 4. Paired bootstrap tests (test split, F1)", ""] + [f"- row {k}: {v}" for k, v in comp.items()] + [""]

    h3 = RESULTS / "h3_video.json"
    if h3.exists():
        L += ["## 5. H3 video cascade (720p)", "", "| Row | Clip | k | FPS | Unique potholes | True count | "
              "Count error | AI frames | Gated out |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for v in json.loads(h3.read_text()):
            L.append(f"| {v['row']} | {Path(v['video']).name} | {v['k']} | {v['fps']} | {v['unique_potholes']} | "
                     f"{fmt(v['true_count'])} | {fmt(v['count_error'])} | {v['ai_frames']} | {v['gated_out']} |")
        L.append("")

    # checklist
    m7, m8, m9 = metrics(7, "test"), metrics(8, "test"), metrics(9, "test")
    def ok(cond):
        return "met" if cond else "not met"
    L += ["## 6. Roadmap targets", "",
          "| Target | Result | Status |", "| --- | --- | --- |"]
    if m7:
        L.append(f"| H1: P >= 0.50 with R >= 0.110 | P {fmt(m7['P'])}, R {fmt(m7['R'])} | "
                 f"{ok(m7['P'] >= 0.5 and m7['R'] >= 0.110)} |")
    if m8:
        L.append(f"| YOLO11n reference: P >= 0.70 | P {fmt(m8['P'])}, R {fmt(m8['R'])} | {ok(m8['P'] >= 0.7)} |")
    if m9:
        L.append(f"| Full hybrid <= 50 ms/image on CPU (H2) | median {fmt(m9['median_ms'], 1)} ms | "
                 f"{ok(m9['median_ms'] <= 50)} |")
    if h3.exists():
        best = max(json.loads(h3.read_text()), key=lambda v: v["fps"])
        L.append(f"| Video >= 15 FPS at 720p | {best['fps']} FPS (row {best['row']}, k={best['k']}) | "
                 f"{ok(best['fps'] >= 15)} |")
    L += ["", "Goal 1 itself (three AI stages, one ablation row each against the classical rows, "
          "measured once on the test split) is complete when sections 3 and 5 are filled.", ""]
    out = RESULTS / "goal1_report.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote {out}")


STEPS = {1: step1, 2: step2, 3: step3, 4: step4, 5: step5, 6: step6, 7: step7, 8: step8, 9: step9, 10: step10}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-step", type=int, default=None, help="start here (default: first unfinished)")
    ap.add_argument("--only", type=int, nargs="+", default=None, help="run just these steps")
    ap.add_argument("--clips", default=None, help="folder with .mp4 clips + counts.csv for H3")
    ap.add_argument("--max-boxes", type=float, default=8.0)
    ap.add_argument("--gate-recall", type=float, default=0.98)
    ap.add_argument("--mnv3-epochs", type=int, default=15)
    ap.add_argument("--skip-mnv3", action="store_true", help="retrain only the SVM (fast check)")
    a = ap.parse_args()

    done = state()
    steps = a.only or [s for s in STEPS if (a.from_step and s >= a.from_step) or
                       (not a.from_step and str(s) not in done)]
    if a.from_step is None and a.only is None and not steps:
        steps = [10]
    print(f"running steps {steps}")
    for s in steps:
        print(f"\n========== step {s}: {STEPS[s].__doc__ or ''}".rstrip())
        info = STEPS[s](a)
        mark(s, info if isinstance(info, dict) else None)
    print("\nGoal 1 run finished. Open results/goal1_report.md")


if __name__ == "__main__":
    main()
