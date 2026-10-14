# Six nuScenes case studies — 13 October 2026

These are selected illustrations from two pretrained pipelines, not a controlled camera ablation. Matching uses confidence >=0.30, identical class and global XY centre distance <2 m, after the saved official evaluation filters. TP = matched prediction; FP = unmatched prediction; FN = unmatched ground-truth object. A miss here can reflect confidence, class, localisation or assignment; it does not prove no prediction was produced.

Camera wireframes represent ground truth, not model predictions. Numbered objects are linked to the paired bird's-eye-view panels. Coloured dashed outlines are selected ground truth; solid outlines are matched predictions; crosses indicate missed ground truth at this operating point. Grey boxes show other ground truth. All-class panel counts include objects not highlighted.

## Findings and captions

1. **scene-0555 — fusion benefit dominated by barriers.** LiDAR-only matches 7 objects; fusion matches 18. The 11 fusion-only matches comprise 9 barriers and 2 pedestrians. Both have one false positive. The crops illustrate two barriers and a pedestrian. The scene includes fencing and a pole near selected barriers. These observations do not establish why scores or localisation differ.
2. **scene-0332 — fusion benefit across classes.** Matched objects rise from 22 to 33: 5 cars, 4 pedestrians and 2 trucks are matched only by fusion. False positives rise from 2 to 3. The three crops show a truck, car and pedestrian; this case balances the barrier-heavy example. Visual classification labels follow the dataset annotations.
3. **scene-0562 — fusion regression.** LiDAR-only matches 21 objects and fusion 14. Four barriers, two pedestrians and one car are matched only by LiDAR. False positives are 2 and 3 respectively. Fencing and shade are visible around highlighted objects. Occlusion and lighting are plausible investigative directions, not established explanations.
4. **scene-0626 — mixed outcomes in wet conditions.** LiDAR-only matches 25 objects and fusion 19. Eight barriers are matched only by LiDAR, while fusion gains two other matches: the net difference is six. Neither model has false positives at the chosen threshold. Wet road surfaces and image softness are visible. Do not conclude that rain caused the regression without controlled tests.
5. **scene-0913 — shared success despite difficult-looking imagery.** Both models match 71 of the same objects (47 barriers, 18 cars, 4 buses and 2 trucks); fusion matches one additional object. Each has one false positive. Crops include a barrier, car and bus. Some projected objects are partly obstructed; projection does not prove unobstructed visibility. This selected frame is not a weather-robustness benchmark.
6. **scene-0092 — shared failure dominated by roadside objects.** Both miss 33 of the same objects: 28 barriers, 4 traffic cones and one truck. Overall matched counts are 10 and 11, and false positives are 3 and 1. The shared misses show that fusion does not resolve every failure. Confidence-threshold sensitivity, geometry and class confusion require further diagnosis.

## Selection and limitations

The original candidate search ranked all 6,019 frames by each outcome count and retained up to five distinct scenes per category. Six frames were selected after overview review for illustrative clarity and a mix of outcomes. This favours crowded scenes and frequent classes. Do not use the six frames as the robustness evaluation subset or estimate population performance from them.

Within each frame, highlighted objects are selected by projected box area, prioritising class diversity. Only boxes fully within a camera image and in front of its near plane are eligible. This is a geometric selection, not an occlusion/visibility measurement. The stored GT index refers to the filtered per-frame list, not an original annotation token.

Projection uses recorded calibration and camera-time ego pose with sample-time boxes; no object-motion compensation. Camera images are cropped and enlarged without enhancement. Camera boxes can be partly occluded by nearer objects. BEV is expressed in the LiDAR-time ego frame. The figures do not rerun inference or recompute mAP/NDS.

## Files and reproduction

- Six PNG figures, suitable for discussion and further editorial refinement.
- annotation_records.json: exact sample tokens, GT indices, class labels, camera channels and saved matching records for each numbered object.
- build_six_figures.py: reproducible plotting code. Put the contents of the supplied nuscenes_six_case_annotation_inputs.zip in a sibling directory named six_inputs, then run the script. It writes annotated_cases.
- input_archive_sha256.txt and manifest.json: provenance and output hashes.
- index.html: open locally to browse the figures.

Next experiments: predeclare a representative robustness subset; compare clean and missing-view inputs at identical tokens; assess score-threshold sensitivity; measure latency/memory under a common setup. Reduced-camera masking measures missing-view robustness of the six-camera checkpoint, not a purpose-trained two-camera detector.
