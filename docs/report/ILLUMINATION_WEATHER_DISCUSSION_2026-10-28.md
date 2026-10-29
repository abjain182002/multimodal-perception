> **29 October 2026 update:** The historical case discussion below is retained. Subsequent [natural rain/night group analysis](NATURAL_WEATHER_GROUPS_2026-10-28.md) and [synthetic fog study](SYNTHETIC_FOG_RESULTS_2026-10-29.md) are now complete. Earlier statements that broader group evaluation remains future work are superseded by those reports within their stated scope; representative real-weather causal evaluation remains unestablished.

# Illumination, rain and diagnostic limitations

This section integrates the four-case darkening follow-up, completed ten-class diagnostics, real rain-labelled examples and two initial night frames plus an additional annotation-positive night/rain pedestrian frame. It is a report draft based on preserved project evidence, not a new inference experiment. Numerical sources are linked below; literature-based mechanisms and mitigation claims require separate primary-paper attribution when integrated into the dissertation.

## Question and evaluation design

The investigation asks how selected detection outcomes respond to reduced image values and what additional real-condition examples reveal about the limits of those outcomes. Two complementary designs are used. The controlled follow-up changes image-value gain on identical A–D frames while retaining each pipeline's checkpoint, LiDAR inputs and metadata. The real rain/night cases describe behaviour in recorded scenes; their different geometry, object composition and recording context prevent a causal comparison of weather effects.

The four frames were originally chosen to illustrate fusion gain, fusion regression, shared success and shared failure. They are outside the predefined 600-frame challenge subset. Their highlighted clean outcomes reproduce, but fresh clean predictions are not numerically identical to the earlier preserved predictions. The rain comparison uses its own original records; those values must not be substituted for the fresh four-case clean outputs. These selected cases explain particular behaviours and cannot estimate their prevalence.

Local diagnostic matching requires class agreement, score at or above the selected threshold, and a score-ordered one-to-one assignment at XY centre distance strictly below 2 m. Thresholds 0.10, 0.30 and 0.50 were inspected; 0.30 is the primary operating point. Ten-class frame tables use recovered original annotations and range, zero-GT-point and bicycle-rack filtering. This local filtering/matching implementation is distinct from the official nuScenes AP/NDS evaluator and from independent evaluator validation. Counts cover the complete frame, while each displayed crop shows one camera view. [S1–S3]

## Gains depend on the selected class composition

The original scene-0555 example supplies a useful contrast to A's multi-class fusion gain: matches rise from 7 to 18, but nine of its eleven fusion-only matches are barriers and two are pedestrians, with one FP for each pipeline. A's eleven fusion-only matches instead comprise five cars, four pedestrians and two trucks. Thus, similar total gains can arise from different class compositions. These deliberately selected frames should not be used to estimate which classes benefit most across the dataset. Together with B's regression, they motivate reporting class mix and selection history alongside attractive aggregate counts. [S8]

## Controlled illumination sensitivity: selected objects and complete frames

| Case / target | Highlighted clean outcome | Fusion target at gain 0.50 / 0.25 | Fusion frame TP / FP / FN: clean → 0.50 → 0.25 |
|---|---|---|---|
| A / truck | Fusion match; LiDAR miss | Match / match | 33/3/13 → 30/4/16 → 26/6/20 |
| B / car | LiDAR match; fusion miss | Match / miss | 14/3/24 → 15/3/23 → 12/2/26 |
| C / bus | Both match | Match / match | 72/1/7 → 69/1/10 → 56/1/23 |
| D / cone | Both miss | Miss / miss | 11/1/34 → 9/1/36 → 8/1/37 |

All entries use score 0.30. The table is generated from the same sources as the consolidated [case evidence](../../results/supervisor_followup/20261028/case_evidence.csv). TP, FP and FN denote diagnostic true positives, unmatched predictions and missed eligible ground-truth objects. [S1–S3]

**Case B separates confidence eligibility from geometric quality.** Half brightness raises the car's fusion score from 0.286743 to 0.316402, crossing 0.30, while XY centre error rises from 0.095702 to 0.179738 m. Quarter brightness reduces the score to 0.276067 and the car again misses at this threshold. The nearby candidate exists in every tested condition; the binary outcome is sensitive to the operating point. At half brightness, precisely this GT car is gained without losing another previously matched object; the FP count remains three, although FP identities need not remain identical. Fusion still has fewer frame matches than LiDAR's 21 TP / 2 FP / 17 FN. This supports a bounded recovery, not uniform improvement. A score increase alone is insufficient evidence of a better box or a globally preferable threshold. [S1–S3]

**Cases A and C show why one successful object cannot establish robustness.** A's truck retains a match at 0.30 even as the complete frame's fusion TP count falls by seven between clean and quarter brightness. Its quarter-brightness score is 0.490191, so it fails at threshold 0.50. C's bus stays matched, but total fusion TP falls from 72 to 56 and FN rises from 7 to 23. The bus's centre error also increases from approximately 0.499 to 0.654 m. Both detection retention and degradation are therefore present in the same example. [S1–S3]

**Case D is a low-score candidate rather than evidence of complete candidate absence.** The cone has nearby same-class predictions but remains below 0.30; at 0.10 it is matched in all displayed conditions. The frame-wide FP/FN trade-off must accompany any threshold discussion. The case does not justify choosing 0.10 for deployment. [S1–S3]

Original visibility bands and current-scan point counts constrain possible explanations. A/B/C/D have 73/2/65/1 current LiDAR points and visibility bands 80–100% / 80–100% / 60–80% / 80–100%. Consequently, describing B as severely occluded would contradict its recovered visibility record. Sparse current-scan support provides difficulty context, but the model also uses nine preceding sweeps. Visibility bands are neither measured image brightness nor per-camera visible-pixel fractions. Distance, geometry, image appearance and point support remain competing explanations; no causal failure mechanism is isolated. [S3]

The separately predefined 600-frame darkening experiment provides aggregate context: mAP ×100 falls from 68.1300 to 67.2352 and 64.5150 at gains 1, 0.50 and 0.25. Case B's local increase can coexist with that aggregate decline. Because A–D are outside that sample, their counts are not a decomposition of the 600-frame metric. Multiplying encoded image values tests input sensitivity and does not reproduce night lighting, sensor noise or atmospheric attenuation. [S6]

## Common weather focus: what the recorded rain and night examples add

Rain is the common-weather example actually represented by the selected scene descriptions. The bus in rainy scene-0913 is matched by both models. In rainy scene-0626, the selected barrier matches for LiDAR at score 0.379211 but falls below threshold for fusion at 0.204313. Fusion's centre error is nevertheless smaller: 0.322135 versus 0.397904 m. Thus, rain-labelled scenes include shared success and fusion regression. They differ in class, geometry, distance and context, so the contrast cannot establish rain as the cause. Rain intensity was not measured. No dry, sunny or overcast comparison should be invented from missing description words. [S4]

The two real-night frames were selected using metadata before inspecting predictions: the middle timestamped validation frame in each of the first two alphabetically ordered night-labelled scenes. Scene-1059 contains four eligible cars; both models yield 2 TP / 0 FP / 2 FN at 0.30. The nearer cars at 9.54 and 15.87 m have high annotated visibility and 147/110 current-scan points. The missed cars at 28.45 and 33.43 m have visibility 0–40% and 4/7 points. These co-varying measurements support object-difficulty discussion, not attribution to darkness alone. At 0.10 LiDAR matches four cars with one FP and fusion three with none; at 0.50 LiDAR matches one and fusion two, both without FPs. Threshold choice changes the apparent comparison. [S5]

Scene-1060 combines night and rain, but its chosen frame contains no ground truth in the ten detection classes. Each model produces one unmatched pedestrian candidate at 0.30. Recall is undefined for this frame. A scene description mentioning a pedestrian does not supply a pedestrian annotation for every frame, and an unmatched prediction under this protocol does not prove physical absence of a person or a hallucination mechanism. This case exposes an annotation-interpretation limit; it is not a positive night/rain pedestrian test. The two selected frames are from related acquisition-log prefixes and are not independent representative recording environments. [S5]

## Additional annotation-positive night/rain pedestrian

The final bounded addition supplies the positive labelled target absent from the selected scene-1060 frame. A rule fixed before viewing predictions selected scene-1067 from three explicitly night/rain-labelled validation scenes with preserved keyframes. The two alphabetically earlier scenes contain no eligible pedestrian frame under the declared support/range criteria; scene-1067 contains 22. The chosen eligible frame is nearest its scene's middle timestamped validation frame. This is deterministic, availability-restricted sampling, not a representative prevalence study. The ranking was checked against the preserved candidate audit rather than independently rescanning the complete source metadata locally. [S7]

The pedestrian is 30.0595 m away, with annotated visibility 80–100%, one annotated current LiDAR point and no current radar points. Raw-scan recomputation confirms the one-point count; preceding model sweeps are not included in it. GT-only projection selects CAM_BACK, where the box spans approximately 23.9 × 48.8 pixels. The original crop is very dark, so the supplied annotation and geometric projection must not be described as independently establishing clear visual identity. [S7]

At score 0.30, LiDAR and fusion both match four of seven eligible frame objects, with zero unmatched predictions and three misses. The highlighted pedestrian is a shared miss: nearest same-class candidate scores are 0.029148 and 0.008018, with XY errors 0.212734 and 0.214623 m. Neither pipeline has any same-class candidate within 2 m eligible at even 0.10, and the pedestrian remains missed at all three predeclared thresholds. Thus the evidence identifies a score-exclusion outcome despite nearby candidates. It does not identify why the learned scores are low. No post-hoc lower threshold was searched. [S7]

This additional case demonstrates that high annotated visibility and a geometrically close candidate do not guarantee an eligible detection in the tested operating range. Low image brightness, distance, point support and model differences remain competing explanations; rain and darkness co-occur. The added frame shares the acquisition-log prefix of the original night cases, and adds no independent environment-level replication. Preserve the earlier annotation-negative scene-1060 case alongside it: the pair distinguishes annotation availability from detection success. [S7]

## Implications, limitations and completion boundary

The combined evidence supports three conclusions: selected fusion outcomes can improve or regress; confidence and localisation can change differently; and object-level success can hide substantial frame-wide losses. Rain and night images add recorded environmental context to the controlled darkening test. They do not establish representative common-weather accuracy, isolate illumination/weather effects, or demonstrate physical robotics safety.

For the dissertation, retain the A–D composite, the existing rain/night comparisons and the compact evidence table. Explain each result through its question, measurement, alternative explanations and implication rather than expanding the number of illustrations. Broader weather-stratified evaluation would require verified frame/scene labels, a declared selection and aggregation protocol, adequate support and uncertainty analysis. The added positive annotated pedestrian example is now complete, with its shared failure retained. Broader weather-stratified evaluation remains future work.

The case-analysis synthesis is complete within the above scope. Report-wide literature integration, final formatting and poster work remain separate. The latest derived bundles still require verified preservation on the user's Drive; Git publication does not establish that copy.

## Evidence sources

- **S1:** [A–D protocol, figures and interpretation](../../results/nuscenes/bevfusion/case_analysis/20261028/README.md).
- **S2:** [Exact target scores and centre errors](../../results/nuscenes/bevfusion/case_analysis/20261028/results/object_evidence.csv).
- **S3:** [Ten-class counts, original visibility and assignment checks](../../results/nuscenes/bevfusion/case_analysis/20261028/metadata_completion/README.md), including its linked CSVs and completion audit.
- **S4:** [Rain comparison and box diagnostics](../../results/nuscenes/bevfusion/case_analysis/20261028/extension/README.md).
- **S5:** [Real night cases, original-image comparisons and limitations](../../results/nuscenes/bevfusion/case_analysis/20261028/night_cases/README.md).
- **S6:** [Predefined 600-frame darkening study](../../results/nuscenes/bevfusion/image_darkening/20261015/README.md).

- **S7:** [Added annotation-positive night/rain pedestrian: selection, figures and diagnostics](../../results/nuscenes/bevfusion/case_analysis/20261028/night_rain_pedestrian/README.md).

For coverage of every unique selected frame, use the [nine-case discussion map](CASE_COVERAGE_AND_STRUCTURE_2026-10-28.md); repeated A–D analyses do not create additional independent scenes.

- **S8:** [All original six-case captions and selection limitations](../../results/nuscenes/bevfusion/case_studies/20261013/annotated_cases/README.md).
