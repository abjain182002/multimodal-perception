# Baseline-matched CLOCs result

The frozen YOLO26x+CLOCs checkpoint was evaluated on all 3,769 KITTI SECOND validation frames. PointPillars reproduced the historical baseline exactly across all nine strict Car IoU 0.70 AP40 metrics (maximum absolute AP error 0.0). This is validation-set evaluation, not KITTI test-server submission.

| 3D AP40 | PointPillars | CLOCs | Difference (points) |
| --- | ---: | ---: | ---: |
| Easy | 90.3449 | 90.1409 | -0.2040 |
| Moderate | 79.1299 | 79.2542 | +0.1243 |
| Hard | 76.1566 | 73.6326 | -2.5239 |

## Report-ready finding

On the frozen KITTI validation split, learned camera–LiDAR candidate rescoring produced mixed effects. Moderate Car 3D AP40 increased marginally from 79.1299 to 79.2542, while Easy and Hard performance decreased by 0.2040 and 2.5239 AP points respectively. The experiment therefore supports a difficulty-dependent trade-off rather than overall superiority of fusion. Baseline reproduction was verified before accepting the comparison.

## Training and selection

The checkpoint used 128 train-only frames, divided into 96 optimization and 32 internal development frames. All nonignored candidates (ratio0) outperformed the tested ratio8 negative-sampling policy on internal development candidate AP (0.85995 versus 0.74268). This is a single-seed comparison of these settings; candidate AP is not detection AP40. The checkpoint was selected on internal development data, not the 3,769-frame validation results.

## Reproduction investigation

An earlier raw-cloud experiment produced a lower PointPillars baseline and was retained as a separate diagnostic. Replaying historical predictions reproduced their original AP exactly, narrowing the discrepancy to inference/preprocessing. Camera-FOV cloud reduction and the KITTI box-range filter were aligned with the historical run before rerunning all validation frames. The corrected run reproduced aggregate baseline AP exactly; this does not assert bitwise prediction equality on every frame.

The earlier raw-cloud +4.9345 Moderate AP difference must not be substituted for this baseline-matched +0.1243 result. Figures and case studies in the earlier diagnostic directory remain historical artifacts and must be labelled accordingly.

## Coverage and limitations

Across all 14,385 labelled Cars, best-IoU coverage decreased from 10,650 to 10,447: 186 objects gained coverage and 389 lost it. For occlusion label 2, coverage decreased from 2,001/3,074 to 1,831/3,074. These diagnostics use best 3D IoU >= 0.70 without official difficulty/ignore or one-to-one matching. They suggest where to inspect failures but do not establish the cause of the Hard AP decrease.

The fusion model was trained using raw-cloud candidates and evaluated using reduced-FOV candidates, so a preprocessing distribution shift remains a limitation. Only one seed, a small training subset and one selected checkpoint were evaluated. No confidence interval, significance test, generalization claim or cross-dataset result is established. Further changes must be selected on internal development data; the inspected validation split is not an untouched final test set.

## Evidence and figures

- Numerical source: `results/clocs_full_val_reduced/ap40.json`.
- Coverage source: `results/clocs_full_val_reduced/coverage_diagnostic.json`.
- Archive receipt: `results/clocs_full_val_reduced/archive_receipt.json`.
- Figure sources: `matched_ap40.csv`, `matched_coverage.csv`, and `figure_manifest.json` in the same directory.
- Report figures: `matched_ap40.png/pdf` and `matched_coverage.png/pdf`; captions in `figure_captions.md`.

Regenerate from the repository root with `python3 scripts/plot_clocs_reduced_results.py` (NumPy and Matplotlib required). The script checks frame counts, AP parity and coverage arithmetic, then records source/output hashes.
