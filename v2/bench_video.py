"""H3 video benchmark: FPS and pothole count for k = 1, 2, 4 (headless, no drawing).

    python bench_video.py --video clip.mp4 --row 9 --true-count 7

Frames are resized to 720p first (roadmap target: >= 15 FPS at 720p).
Count error = |unique track IDs - true count| (label the 10 clips by hand).
"""
import argparse
import time

import cv2

from cascade import Cascade, VideoCascade, threaded_frames, video_fps
from config import row_config


def bench_clip(cascade, video, ks=(1, 2, 4), height=720, true_count=None):
    """FPS, unique track count and count error per k for one clip (headless)."""
    out = []
    fps_in = video_fps(video)
    for k in ks:
        vc = VideoCascade(cascade, k=k, fps=fps_in)
        t0, n = time.perf_counter(), 0
        for idx, frame in enumerate(threaded_frames(video)):
            h, w = frame.shape[:2]
            if h != height:
                frame = cv2.resize(frame, (round(w * height / h), height), interpolation=cv2.INTER_AREA)
            vc.step(frame, idx)
            n += 1
        fps = n / max(time.perf_counter() - t0, 1e-9)
        out.append({"video": str(video), "k": k, "fps": round(fps, 1), "frames": n,
                    "unique_potholes": len(vc.counted), "ai_frames": vc.stats["ai_frames"],
                    "gated_out": vc.stats["gated_out"], "true_count": true_count,
                    "count_error": None if true_count is None else abs(len(vc.counted) - true_count)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--row", type=int, default=9)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--true-count", type=int, default=None)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    cascade = Cascade(row_config(args.row))
    fps_in = video_fps(args.video)
    for k in args.ks:
        vc = VideoCascade(cascade, k=k, fps=fps_in)
        t0, n = time.perf_counter(), 0
        for idx, frame in enumerate(threaded_frames(args.video)):
            h, w = frame.shape[:2]
            if h != args.height:
                frame = cv2.resize(frame, (round(w * args.height / h), args.height), interpolation=cv2.INTER_AREA)
            vc.step(frame, idx)
            n += 1
        fps = n / (time.perf_counter() - t0)
        msg = f"k={k}: {fps:.1f} FPS over {n} frames, unique potholes={len(vc.counted)}, " \
              f"AI frames={vc.stats['ai_frames']}, gated out={vc.stats['gated_out']}"
        if args.true_count is not None:
            msg += f", count error={abs(len(vc.counted) - args.true_count)}"
        print(msg)


if __name__ == "__main__":
    main()
