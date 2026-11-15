# CraterScope 2.0: reported outcomes

The version 2 test split was evaluated. Thresholds were selected on validation, and comparisons use the same detector and split. The repository does not publish the earlier invalid mAP calculation.

| Target | Result | Status |
|---|---:|---|
| H1: precision ≥ 0.50 and recall ≥ 0.110 | Precision 0; recall 0 | Not met. Classical proposals miss nearly every pothole. |
| YOLO11n reference: precision ≥ 0.70 | Precision 0.887; recall 0.104 | Precision met; recall is below the 0.110 baseline. |
| Method A ≤ 40 ms/image | 41.7 ms | Essentially met. |
| Full hybrid ≤ 50 ms/image | 62.5 ms | Not met. |
| Video ≥ 15 FPS at 720p (H3) | Not measured | Run `python finish_goal1.py --only 7 9 10 --clips clips`. |

The supported positive finding is that a road strip removes 40% of the input with no measurable accuracy loss. The road-strip speed-up remains a projection. The gate does not provide a successful hybrid result.

## Version 3 research assessment

Yes as a piece of research; no as a headline result. The corrected evidence is more defensible, but novelty is limited and the result relies on one model. It may suit a workshop or mid-tier conference; a strong venue could reject it.

Priority follow-up work:

1. Measure a rectangular road-strip model's latency.
2. Repeat training across three or four lightweight architectures with multiple seeds.
3. Test a purpose-built cheap gate at lower pothole prevalence, where the cost model predicts gating may pay off.

The YOLO11n test mAP@0.5 result was 0.39; one run of one model is not a headline result. All three proposed follow-ups need a GPU route, which remains the blocker. These are research proposals, not published version 3 results.
