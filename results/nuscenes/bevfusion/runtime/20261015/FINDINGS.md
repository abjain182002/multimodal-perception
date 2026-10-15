# Findings: runtime, memory and accuracy context

## Measured result

The median of the three session medians is 287.8116 ms for the LiDAR-only pipeline and 544.9786 ms for the intact six-camera fusion pipeline: 1.8935×. Maximum PyTorch peak allocated memory is 609.8184 versus 2,231.7979 MiB: 3.6598×. Peak reserved memory is 864 versus 3,278 MiB. The allocated/reserved values are identical across the three sessions within each model.

Session medians range from 278.6903 to 288.8058 ms for LiDAR and 541.2600 to 545.9939 ms for fusion. Across paired frames and repeats, the median fusion/LiDAR latency ratio is 1.9170; the empirical 5th–95th percentile range is 1.7437–2.1639. This is descriptive spread, not a confidence interval. Repeated frames are not independent scene observations.

Pooled loading-and-collation means are 63.9772 ms for LiDAR and 355.4288 ms for fusion. These separately recorded CPU dataset times include local input reading and transforms; cache effects and the first pass affect them. They are not pure disk bandwidth and should not be confused with primary test_step latency. No full physical-system throughput, energy consumption or robot-control deadline was measured.

## Accuracy versus computation

The separately preserved 600-frame accuracy study reports LiDAR-only mAP/NDS of 64.3076/69.4430 and intact fusion 68.1318/71.2913 (metrics multiplied by 100): gains of 3.8242 and 1.8484 points. These results use the same checkpoint identities and source family, but accuracy was not rerun on the 150 timing frames, and the timing protocol explicitly disables TF32. The accuracy-versus-latency figure therefore combines two disclosed evaluation scopes, not a single simultaneous accuracy-and-speed test.

The evidence supports a practical trade-off: this tested fusion pipeline improves recorded-input accuracy while requiring more serial inference time and allocator memory under the declared settings. It does not establish a universal optimum, a monetary cost-per-AP figure, or that this implementation satisfies a mobile robot's deadline. Deployment rates and deadlines must be specified for a particular application and assessed end to end.

## Controls and limitations

- Frozen 150 tokens, one per official validation scene, selected as the second of four preselected timestamp positions, independent of outputs. Not a random frame-weighted sample.
- Three fresh subprocess sessions per model; order L/F, F/L, L/F. Warm-up is 20 samples. CPU intra/inter-op threads are one. Model evaluation mode, inference_mode and no autocast; TF32 and cuDNN benchmarking disabled.
- All six raw measurement files and stored summary statistics were verified. Latency mean, median, p95, minimum and maximum and loading statistics were recomputed. Ordered tokens and paired LiDAR point counts agree.
- Clock snapshots at session end vary from 945 to 1,185 MHz SM clock, and temperatures from 71 to 77°C. These snapshots neither prove sustained throttling nor characterize clocks throughout a pass. Shared Colab scheduling, clocks and temperature are uncontrolled confounders.
- Initial compilation used CUDA 12.1.66 and Ninja; binary hashes can differ from other builds. Core package/source/checkpoint versions and synthetic-operation checks are recorded.
- Model weights and TF32 settings were checked by executed code. Host CPU loading uses one thread, and different threading, kernels, batching or precision can produce different results.
- PyTorch memory peaks omit non-PyTorch allocations and do not measure host RAM. Reserved and allocated are alternative allocator metrics, not additive costs.
- No significance test, continuous power measurement, repeat on another GPU, or embedded-system run was performed. Three sessions are descriptive repeated measurements.
- Runtime of masked camera conditions was not benchmarked. Their existing code retains all six view slots, so camera-count reductions cannot be treated as measured computational savings.

## Architecture and next work

The [architecture diagrams](../../../../../docs/architecture/README.md) distinguish the project orchestration from reused models and show the actual pinned BEVFusion processing order. ConvFuser precedes SECOND/SECONDFPN; camera view transformation also uses LiDAR depth cues and calibration. The separate LiDAR-only checkpoint omits the camera/fusion path.

The [cost-analysis framework](../../../../cost_performance/README.md) records initial official sources, missing price information and scenario formulas. A current matched sensor bill of materials remains pending. Additional image-degradation challenge tests should be separately frozen and evaluated; they are not part of this runtime result.
