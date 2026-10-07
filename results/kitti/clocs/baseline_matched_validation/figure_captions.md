AP40 figure: Baseline-matched KITTI SECOND validation on 3,769 frames, Car strict IoU 0.70. Top: AP40; bottom: paired differences. The historical PointPillars baseline reproduces exactly across all nine metrics. One trained fusion checkpoint; no confidence intervals or significance test.

Coverage figure: All 14,385 labelled Cars, grouped by camera depth and occlusion label. Coverage uses each GT object's best retained-detection 3D IoU >= 0.70; it does not apply official difficulty/ignore matching or one-to-one assignment. Occlusion 0/1/2/3 denotes fully visible/partly occluded/largely occluded/unknown. These diagnostics do not establish the cause of Hard AP degradation.

