# Include every case without repeating every explanation

All nine unique selected frames can contribute to the dissertation's critical discussion. They need not receive nine equally long subsections or nine full-size figures. Use a coverage table to account for every case, discuss shared mechanisms together, and spend the most space on cases that distinguish competing interpretations. There is no methodological requirement to omit cases from critical discussion; selection and weighting must instead be transparent.

## Coverage of the current case set

| Unique frame | Role and supported contribution | Main discussion connection |
|---|---|---|
| scene-0555 | Original fusion benefit, dominated by barriers; 7 → 18 matches, with one FP per pipeline | Class composition affects the interpretation of a selected gain |
| scene-0332 / A | Fusion benefit across classes; selected truck retained under darkening while frame counts decline | Confidence, geometry, assignment and selected-object versus frame outcomes |
| scene-0562 / B | Fusion regression; selected car briefly crosses 0.30 under half brightness while centre error increases | Threshold eligibility is distinct from box improvement |
| scene-0626 / E | Rain-labelled mixed frame; highlighted barrier is a LiDAR match and fusion miss despite smaller fusion centre error | Scores and localisation can disagree; different scenes do not isolate rain |
| scene-0913 / C | Rain-labelled shared bus success; quarter brightness reduces fusion frame TP 72 → 56 | One retained target cannot establish frame-wide robustness |
| scene-0092 / D | Shared cone miss at 0.30 despite nearby candidates; recovered at 0.10 in the focal follow-up | A miss does not imply candidate absence or justify a globally lower threshold |
| scene-1059 | Original night frame: both models match two nearer cars and miss two farther cars at 0.30 | Distance, annotated visibility and point support co-vary |
| scene-1060 | Night/rain frame with no eligible ten-class GT; each model has one unmatched pedestrian at 0.30 | Scene descriptions are not frame labels; recall is undefined here |
| scene-1067 | Added annotation-positive night/rain pedestrian: both models miss at 0.10/0.30/0.50, with nearby low-score candidates | Shared failure despite high annotated visibility; one current-scan point; no isolated weather cause |

The original six frames, A–D follow-up, rain extension and night additions overlap. **Count each unique frame once.** A repeated analysis under several image gains or thresholds does not create an independent scene. The three night frames share an acquisition-log prefix and are not independent representative environments.

Sources: [original six-case captions](../../results/nuscenes/bevfusion/case_studies/20261013/annotated_cases/README.md), [A–D and ten-class follow-up](../../results/nuscenes/bevfusion/case_analysis/20261028/README.md), [original night cases](../../results/nuscenes/bevfusion/case_analysis/20261028/night_cases/README.md), and [additional pedestrian case](../../results/nuscenes/bevfusion/case_analysis/20261028/night_rain_pedestrian/README.md). Historical and fresh clean numerical records remain distinct; the coverage table does not pool their metrics.

## Suggested critical-discussion structure

1. **Where fusion helps and regresses:** synthesise 0555, A and B. Explain class composition and the separate pretrained-pipeline comparison boundary.
2. **What a detection label hides:** use B and D to distinguish confidence, localisation and assignment. Compare complete-frame counts rather than one target alone.
3. **Controlled versus recorded environmental conditions:** connect A–D darkening to C/E rain examples and 1059/1067 night cases. Give competing explanations and measured difficulty context without assigning weather causality.
4. **Annotation and interpretation limits:** contrast annotation-negative 1060 with annotation-positive 1067; retain both. Discuss confidence scores, visibility bands, temporal sweeps, scene correlation and illustrative selection.
5. **Practical implications:** connect these limitations to aggregate robustness and the separate offline corridor-alert analysis. Suggested remedies need literature attribution; untested remedies remain proposals.

For every major claim, state the evidence, alternative explanations, limitation and implication. Do not turn the discussion into a second results catalogue. A compact case-coverage table, readable selected composites and appendix/repository detail allow every case to be included within the 40–50-page report target. The exact allocation remains an editorial decision after laying out the chapters.

## Key takeaway

Include all cases through a transparent, thematic synthesis. Give each enough space to establish its distinct contribution; avoid interpreting repetition as more evidence or adding experiments simply to increase the case count.
