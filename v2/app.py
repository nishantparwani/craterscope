"""CraterScope 2.0 Streamlit app.

    streamlit run app.py

Sidebar  : choose the pipeline (Classical / H1 / H1-SVM / H2 / YOLO reference)
Image tab: upload an image -> boxes, gate decision, per-stage timings
Tracks   : upload a video -> H3 AI-gated cascade, YOLO every k-th frame,
           ByteTrack IDs, one count per pothole
"""
import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from cascade import Cascade, VideoCascade, draw_boxes, heatmap, threaded_frames, video_fps
from config import preset

st.set_page_config(page_title="CraterScope 2.0", layout="wide")

PIPELINES = ["H1: classical + MobileNetV3", "H2: gate + YOLO11n (INT8)", "Classical only",
             "H1-SVM: classical + LBP/HOG SVM", "YOLO11n alone (FP32 reference)"]


@st.cache_resource(show_spinner="Loading models...")
def get_cascade(name: str, overrides_json: str) -> Cascade:
    cfg = preset(name)
    cfg.update(json.loads(overrides_json))
    return Cascade(cfg, require_gate=False)


def decode_upload(upload) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(upload.getvalue(), np.uint8), cv2.IMREAD_COLOR)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("CraterScope 2.0")
name = st.sidebar.radio("Pipeline", PIPELINES)
base = preset(name)
overrides = {}
with st.sidebar.expander("Thresholds (defaults = tuned on validation)"):
    if base["mode"] == "h1":
        overrides["verifier_threshold"] = st.slider("Verifier threshold", 0.0, 1.0,
                                                    float(base["verifier_threshold"]), 0.05)
    if base["mode"] == "yolo":
        overrides["yolo_conf"] = st.slider("YOLO confidence", 0.05, 0.95, float(base["yolo_conf"]), 0.05)
show_props = st.sidebar.checkbox("Show classical proposals (yellow)", value=True)
show_map = st.sidebar.checkbox("Show fused score map S", value=False)

try:
    cascade = get_cascade(name, json.dumps(overrides, sort_keys=True))
except (FileNotFoundError, ImportError) as e:
    st.error(f"This pipeline is not ready: {e}")
    st.info("Copy the trained files into the models/ folder (see README, step 5).")
    st.stop()
for w in cascade.warnings:
    st.sidebar.warning(w)
st.sidebar.caption(f"Ablation row {cascade.cfg['row']}: {cascade.cfg['label']}")

tab_img, tab_tracks = st.tabs(["Image", "Tracks"])

# ------------------------------------------------------------------ image
with tab_img:
    up = st.file_uploader("Road image", type=["jpg", "jpeg", "png", "bmp"])
    if up is not None:
        img = decode_upload(up)
        cascade.run(img)                         # warm-up so timings are steady-state
        r = cascade.run(img)
        vis = img
        if show_props and len(r["proposals"]):
            vis = draw_boxes(vis, r["proposals"], color=(0, 220, 255), thickness=1)
        vis = draw_boxes(vis, r["boxes"], r["scores"], color=(0, 0, 255), thickness=3)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Potholes", len(r["boxes"]))
        c2.metric("Gate", "open" if r["gate_open"] else "closed")
        c3.metric("Presence p", "-" if r["presence"] is None else f"{r['presence']:.2f}")
        c4.metric("Total time", f"{r['timings']['total']:.1f} ms")

        if show_map and r["S"] is not None:
            left, right = st.columns(2)
            left.image(vis, channels="BGR", caption="Detections (red) / proposals (yellow)")
            right.image(heatmap(r["S"]), channels="BGR", caption="Fused score S")
        else:
            st.image(vis, channels="BGR", caption="Detections (red) / proposals (yellow)")

        st.subheader("Per-stage time (ms)")
        st.table({"stage": list(r["timings"]), "ms": [round(v, 2) for v in r["timings"].values()]})
        ok, buf = cv2.imencode(".jpg", vis)
        st.download_button("Download annotated image", buf.tobytes(), f"craterscope_{up.name}.jpg",
                           "image/jpeg")

# ------------------------------------------------------------------ video
with tab_tracks:
    st.caption("H3: the classical gate runs on every k-th frame; the AI stage runs only when the "
               "gate opens; ByteTrack carries boxes in between and gives one ID per pothole.")
    vid = st.file_uploader("Dash-cam video", type=["mp4", "avi", "mov", "mkv"], key="video")
    k = st.select_slider("Run AI every k-th frame", options=[1, 2, 4], value=2)
    max_frames = st.number_input("Max frames (0 = all)", min_value=0, value=600, step=100)
    if vid is not None and st.button("Run video cascade"):
        src = Path(tempfile.gettempdir()) / f"cs_in_{vid.name}"
        src.write_bytes(vid.getvalue())
        fps = video_fps(src)
        try:
            vc = VideoCascade(cascade, k=k, fps=fps)
        except ImportError as e:
            st.error(f"{e}. Run: pip install trackers supervision")
            st.stop()
        out_path = Path(tempfile.gettempdir()) / f"cs_out_{Path(vid.name).stem}.mp4"
        writer = None
        frame_box, prog, stats_box = st.empty(), st.progress(0.0), st.empty()
        total = int(cv2.VideoCapture(str(src)).get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        limit = min(total, max_frames) if max_frames else total
        t0 = time.perf_counter()
        for idx, frame in enumerate(threaded_frames(src)):
            if idx >= limit:
                break
            o = vc.step(frame, idx)
            ids = list(o["tracks"])
            vis = draw_boxes(frame, [o["tracks"][i] for i in ids], ids=ids, color=(0, 0, 255))
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(vis)
            if idx % 10 == 0:
                frame_box.image(vis, channels="BGR")
                prog.progress(min(1.0, (idx + 1) / limit))
                stats_box.write(f"frame {idx + 1}/{limit} - unique potholes so far: {o['unique_potholes']}")
        if writer is not None:
            writer.release()
        elapsed = time.perf_counter() - t0
        prog.progress(1.0)
        s = vc.stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Processing FPS", f"{s['frames'] / max(elapsed, 1e-6):.1f}")
        c2.metric("Unique potholes", len(vc.counted))
        c3.metric("AI ran on", f"{s['ai_frames']} frames")
        c4.metric("Gated out", f"{s['gated_out']} frames")
        st.caption("FPS above includes drawing and Streamlit updates; evaluate headless for the paper.")
        if out_path.exists():
            st.download_button("Download annotated video (mp4)", out_path.read_bytes(),
                               out_path.name, "video/mp4")
