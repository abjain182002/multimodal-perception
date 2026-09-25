# Phase 1 daily targets — 25 September to 10 October 2026

- **Project:** ENG5059P multimodal perception MSc project
- **Planning timezone:** Asia/Kolkata
- **Technical freeze target:** 8–10 October 2026
- **Dissertation chapter drafting:** after 10 October 2026

This is a dated engineering checklist. A date is a target, not evidence that an experiment succeeded. Each day closes with either a verified result and a GitHub commit or an explicit blocker, owner, next action, and revised dependency. No planned component is silently removed to meet the calendar.

## Current verified starting point (25 September)

- The Mac pulled `6b913a1`, passed **100/100** repository tests in **0.551 s**, and reported a clean `main...origin/main` tree.
- The canonical SECOND split has 3,712 training and 3,769 validation frames without overlap. The full KITTI CLOCs GT audit covered 7,481 labelled frames and retained 28,374 exact Car GT boxes after range filtering; see `results/kitti_clocs_gt_audit.json`.
- PointPillars-only KITTI Car Moderate 3D AP40 at IoU 0.70 is **79.1299%** on the full validation split, as recorded in `results/pointpillars_kitti_ap40_strict_summary.csv`. This is a pretrained LiDAR baseline, not a learned-fusion result.
- CPU 3D IoU, aligned camera-box extraction, and the guarded target-example builder are implemented. The saved `reference_structural_smoke` archive has placeholder zero IoUs and cannot be used for training. **Genuine aligned dense 3D boxes from Colab are the immediate gate.**

## Daily engineering targets

| Date | Main target and execution lane | Evidence to close the day |
|---|---|---|
| **25 Sep (Fri)** | Mac: freeze the completed GT audit, split checks, CPU target-builder tests, and a clean baseline. | 100 Mac tests pass; full GT audit and current branch SHA recorded. **Complete.** |
| **26 Sep (Sat)** | Mac: independently check rotated camera-space 3D IoU on KITTI GT examples and boundary cases; inspect dimension order, bottom origin, yaw, and range filter. Prepare GPU export inputs and hashes. | Deterministic geometry preflight with test/output, no inferred 3D IoU from 2D projections, checked into GitHub. |
| **27 Sep (Sun)** | Mac: make the Colab export path executable for train-only frames; inventory the pinned PointPillars checkpoint, KITTI calibration, YOLO detections, frame IDs, and storage capacity. Prepare nuScenes taxonomy/calibration mapping and PointPainting/BEVFusion run requirements without claiming evaluation. | Reproducibility manifest template, GPU start checklist, split-safe train-only development IDs, and recorded missing dependencies. |
| **28 Sep (Mon)** | Mac + Colab: restore GPU access and export a small, *real* train-only batch of dense PointPillars `boxes_lidar`/`boxes_camera`, projections, scores, 2D detections, candidate IDs, and features in identical frame order. | NPZ hashes and the first verified real Car IoUs; inspect positives, negatives, ignored candidates, empty-GT and empty-2D frames. **If GPU remains unavailable, seek alternative authorized compute or supervisor guidance that day; record the blocked experiments.** |
| **29 Sep (Tue)** | Colab: extend export to a documented train-only corpus. Mac: run per-frame construction with strict frame IDs, score/distance/2D-feature reconstruction, GT filtering, and SHA-256 reload checks. | Explicit frame list and per-frame GT/candidate/row/label counts; no unknown split membership or unexplained all-negative labelled frames. |
| **30 Sep (Wed)** | Mac + Colab: cross-check sampled 3D IoUs against the pinned CLOCs/SECOND evaluator, audit score/IoU/range distributions and label imbalance, and freeze the data/target protocol. | Reference comparison with tolerances, checkpoint/config/split/export hashes, and reproducible train-only development selection. If mismatch remains, **do not train**. |
| **1 Oct (Thu)** | Colab: train learned CLOCs using real targets; tune only on development frames drawn from training, monitor positive counts, focal loss and candidate coverage. Mac: prepare fixed inference/evaluation scripts. | Checkpoints stored durably outside Git, their hashes/settings and train-only diagnostics committed; failures documented before further experiments. |
| **2 Oct (Fri)** | Colab: freeze model/config and run fused KITTI Car evaluation with the same strict 3D AP40 protocol as PointPillars, including Easy/Moderate/Hard and BEV/3D. | Comparable PointPillars versus CLOCs table, full frozen validation frame list, model hash, and honest result even if fusion is worse. Reserve final validation for frozen configuration. |
| **3 Oct (Sat)** | Mac + Colab: run PointPainting-style semantic enrichment on a clearly specified KITTI comparison, holding the LiDAR detector, split and metric protocol fixed where possible. | Implemented path, real-data predictions, evaluation and method-specific limitations; mark a smaller pilot explicitly if full evaluation cannot run. |
| **4 Oct (Sun)** | Mac + Colab: controlled camera-input and CLOCs ablations (YOLOv8x/YOLO26x where confidence is available; Falcon only under a defensible score rule), unmatched sentinel rows, and failure examples. | Matched frame lists, frozen thresholds, sample counts, ablation results. Never invent conventional Falcon confidences or mix depth MAE with 3D AP. |
| **5 Oct (Mon)** | Mac + Colab: check nuScenes sensor calibration, data access, official splits/taxonomy, evaluation API and BEVFusion checkpoint compatibility; establish a documented second-dataset baseline. | At least one provenance-backed nuScenes inference/evaluation path; any mini-set result labelled exploratory. No direct numerical KITTI-AP versus nuScenes-mAP ranking. |
| **6 Oct (Tue)** | Colab: attempt and evaluate BEVFusion on nuScenes; test a precisely defined within-dataset and/or cross-dataset comparison with class and sensor mappings stated. Mac: audit PointPainting/CLOCs error cases. | Predictions, metrics, frame/sample counts, configuration and checkpoint hashes, or exact technical blocker and next action for each method. |
| **7 Oct (Wed)** | Mac + Colab: controlled corruption/robustness analysis (camera, LiDAR or calibration perturbations) against clean baselines, with matched examples and severity. | Per-severity results, denominators, corruption settings and representative failure cases; distinguish KITTI and nuScenes protocols. |
| **8 Oct (Thu)** | Mac + Colab: complete essential ablations and standardized runtime measurements on named hardware with warm-up, synchronization and consistent scope. | Repeatable latency distributions plus uncertainty where feasible; no Mac-vs-T4 speed claim without hardware normalization. |
| **9 Oct (Fri)** | Mac: independently recompute key metrics from frozen artifacts, run all tests, inspect hashes and controlled case studies; Colab: close any reproducibility gaps. | Evidence ledger linking every claim to code, split, run, numeric output and checkpoint; clean GitHub milestone commits. |
| **10 Oct (Sat)** | Mac + Colab: freeze Phase 1 code, figures, experiment settings, compact results, dependencies, manifests and backup references. Audit each agreed component: KITTI, nuScenes, YOLO/Falcon, PointPillars, CLOCs, PointPainting-style, BEVFusion, transfer, robustness, ablations, runtime. | Final Phase 1 status matrix with **completed / partial / blocked** and exact evidence; clean synchronized Git branch and durable checkpoint copies. Dissertation writing begins **after** this date. |

## Daily closure and storage rule

1. Record the target, actual run, input frame list and split, checkpoint/config hashes, hardware, exact metrics with denominator, test result, and limitation in `DAILY_LOG.md` or an experiment-specific manifest.
2. Put **source code, tests, configs, small numerical results, evidence manifests and reviewed figures** on GitHub at every verified milestone, with a descriptive commit and push. On the Mac use explicit paths with `git add`, then `git diff --cached --check`, run the relevant tests, `git commit`, `git push origin main`, and verify `git status --short --branch`. Avoid force pushes and committing intermediate or fabricated results.
3. Keep KITTI/nuScenes datasets, raw bulk predictions and multi-gigabyte weights/checkpoints in durable backed-up project storage outside ordinary Git. Commit their exact paths or storage IDs **where appropriate**, SHA-256 hashes, source/permissions and retrieval instructions so each result can be audited. GitHub alone is not a suitable backup for those large or restricted assets.
4. If a day cannot meet its experimental exit criterion, log **blocked** with the concrete dependency; continue independent Mac work and escalate compute/experimental deadline risks promptly. A small successful pilot is not a completed full validation result.

## Scientific gates before any final performance claim

- CLOCs training requires genuine 3D boxes aligned with projected boxes, 2D detections, scores, candidate IDs and GT; real IoUs must be cross-checked and a defensible positive training signal confirmed. The existing smoke example is never training data.
- The 3,769-frame KITTI validation set is not a source of CLOCs hyperparameter choices. Use train-only development; evaluate the frozen final configuration and report actual evaluated frame counts.
- Each fusion strategy and dataset needs its own measured output and provenance. A connected adapter, passed unit tests or an exploratory nuScenes mini run alone does not complete an experimental claim.
- The full agreed Phase 1 scope remains the target. With GPU access unavailable as of the 25 September checkpoint, completion of every GPU experiment by 10 October cannot be guaranteed; record the risk and seek supervisor guidance rather than silently shrinking the work.

**Related records:** `PROJECT_HISTORY.md`, `DAILY_LOG.md`, `results/VALIDATED_RESULTS.md`, `docs/CLOCS_TARGET_PREPARATION.md`, `configs/splits/kitti_second_split.json`.
