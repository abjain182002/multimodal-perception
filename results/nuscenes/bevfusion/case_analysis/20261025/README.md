# Four-case supervisor brief — 25 October 2026

This package addresses the supervisor's request to analyse cases, generate diagnostic data and discuss the findings critically. It consolidates earlier evidence and adds a fresh replay, a four-object score comparison and a compact discussion; it is not a new detector experiment.

**Start here:** [three-page brief](supervisor_case_brief.pdf) or [editable discussion](CRITICAL_CASE_DISCUSSION.md). Page 1 contains one combined figure. Pages 2–3 contain one all-class frame table and critical discussion.

## Finding

The selected truck, car and traffic-cone examples show that a diagnostic miss at score 0.30 can coexist with a nearby same-class saved prediction below that threshold. The selected bus is a shared success at all three tested thresholds. These selected cases explain particular operating-point outcomes; they do not establish why the networks produced those scores, population failure rates, a calibrated confidence model or a causal camera benefit.

The four selected frames are scene-0332 (fusion gain), scene-0562 (fusion regression), scene-0913 (shared success) and scene-0092 (shared miss). The first two already had detailed diagnostics on 20 October. This brief brings them together with the two counterexamples and retains the earlier exploratory selection history. Selection and exact filtered GT indices are in `selection.json`.

## Evidence and checks

- Original input ZIP SHA-256: `574a5ea93c4fc455317562caaa82a30f25ce3f8b20715ddbf2e05c0cff0b9bdd`.
- All six original frames replayed at confidence 0.30: match identities, distances, unmatched predictions and unmatched GT reproduced for both models.
- All 36 saved threshold summaries reproduced: six frames × two models × three thresholds.
- Four displayed camera images checked against preserved file sizes and SHA-256 hashes.
- Scores, distances and match states at 0.10/0.30/0.50 are in `object_evidence.csv` (eight model–object rows).
- All-class counts and paired GT outcomes are in `frame_evidence.csv` (four frame rows).
- The three-page PDF and standalone composite figure were visually checked. No new GPU inference, SDK filtering or mAP/NDS evaluation was performed.

## Reproduce

Use Python 3.12 and the versions in `requirements.txt`. Extract the preserved `nuscenes_six_case_annotation_inputs.zip` to a private input directory containing `candidate_contexts.json`, calibration/pose records and `dataset/`. The raw sensor archive is not redistributed in this folder; obtain the dataset under its original terms. Retain the ZIP and verify its hash before extraction.

Run from this directory:

```bash
python build_case_review.py --inputs /path/to/six_inputs --output .
python build_brief_pdf.py --markdown CRITICAL_CASE_DISCUSSION.md --figure four_case_comparison.png --output supervisor_case_brief.pdf
```

The first script reads preserved source records in `sources/`, verifies original matching/count results, writes the evidence tables and creates the figure. The second renders the Markdown brief using embedded DejaVu fonts supplied by Matplotlib. PDF binary hashes may change with creation timestamps; the numerical outputs should remain identical under the recorded environment.

## Interpretation boundaries

Scores are not demonstrated probabilities. Match labels require score >= threshold, the same class and XY centre distance strictly <2 m, with confidence-ordered one-to-one assignment. A nearby centre does not demonstrate complete 3D-box correctness. Four specific highlighted candidates have no assignment conflict at 0.30; this does not generalise to all candidates in those frames.

The source ground truth and predictions were previously filtered. This replay does not reapply those filters. GT indices identify positions in the saved filtered frame list, not original dataset annotation tokens. The six source frames were outcome-selected, and the displayed four were chosen editorially for contrasting examples. Do not tune deployment thresholds or estimate population rates from them.

Images use projected GT boxes and recorded calibration. No image enhancement, object-motion compensation or visibility/occlusion labelling was applied. Visible fencing or low contrast is an observation, not a causal explanation. The robot-corridor study remains a separate analysis with different outcomes.

## Primary-source connections

- CLOCs authors' [Step 4](https://github.com/pangsu0613/CLOCs#fusion-of-other-3d-and-2d-detectors) replaces original 3D scores with fused scores. This motivates examination of confidence/ranking, not a claim of a tested remedy here.
- TransFusion [paper abstract](https://arxiv.org/abs/2203.11496) proposes soft association for inferior image conditions and misalignment. It is an architectural alternative, not a demonstrated explanation or fix for these cases.

Sources checked 25 October 2026. Full literature synthesis and dissertation-wide integration remain separate tasks. The material is ready to show the supervisor; it has not been sent to him.
