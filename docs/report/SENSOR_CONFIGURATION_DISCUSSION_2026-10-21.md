# Sensor configuration: draft discussion and compact table

Draft date: 21 October 2026. Use with the [source-backed assessment](../SENSOR_CONFIGURATION_ASSESSMENT_2026-10-21.md). This text documents evidence and feasibility; it does not report a new experiment or supervisor-approved exclusion.

## Compact main-report table

| Input configuration | Project evidence | Interpretation |
|---|---|---|
| One camera | KITTI 2D pilots; monocular 3D not evaluated | A camera-only 3D comparison requires an additional model and common evaluator |
| One LiDAR | KITTI PointPillars/PV-RCNN and nuScenes LiDAR | Existing single-modality benchmarks |
| One camera + one LiDAR | KITTI CLOCs | Measured fusion with inconsistent gains across seeds/difficulties |
| Two cameras available + one LiDAR | nuScenes front/back masking | Availability test within a six-view model; physical cost/compute savings not measured |
| Six cameras + one LiDAR | Full nuScenes BEVFusion validation; separate T4 timing | Established accuracy/computation trade-off for the tested pipelines |
| Multiple physical LiDARs, with/without cameras | Feasibility/proposed protocol only | Separate sensor data, compatible weights and a common evaluation are required |

Suggested caption: Evidence for sensor configurations in the software-only project. KITTI evaluates Car strict 3D AP40 at IoU 0.70 on 3,769 validation frames; full nuScenes uses mAP/NDS on 6,019 samples. These metrics and scopes are separate. Masking removes camera information while retaining the original view slots and model weights. No hardware deployment or monetary optimum is established.

## Discussion draft

The experiments show that the value of additional sensing depends on the complete detection pipeline. On KITTI, adding image candidates through the aligned CLOCs implementation changes Moderate 3D AP40 from the PointPillars baseline of 79.1299 to 79.8616, 76.2666 and 78.7616 across three seeds; all three reduce Hard AP. On full nuScenes validation, the separate pretrained fusion pipeline improves mAP/NDS by 3.8798/1.8998 points over LiDAR-only. These outcomes support a conditional benefit rather than a general rule that more modalities improve detection. The models, training histories and datasets differ, preventing a universal ranking by sensor count.

Camera availability also exposes a deployment limitation. On the frozen 600-frame subset, retaining front/back views within the six-camera model produces mAP 63.6269, below the LiDAR-only pipeline's 64.3076. The intervention changes information availability but retains six view slots, so it cannot demonstrate the accuracy or savings of a purpose-trained two-camera system. The separately measured T4 latency and allocator-memory increases for intact fusion establish a computation trade-off; they do not quantify sensor prices, energy use or embedded performance.

A monocular KITTI baseline is a practical next candidate because it could use the same validation identities and AP40 evaluator as the existing one-LiDAR and one-camera-plus-LiDAR results. The inspected ImVoxelNet configuration provides such a path, conditional on input checks, checkpoint loading and a measured resource pilot. Physical multi-LiDAR evaluation requires different data: Waymo provides separate calibrated scanners and a geometric merge is supported by OpenPCDet, but an accessible compatible checkpoint and a practical data/runtime budget have not been established here. A proposed follow-on study therefore fixes sensor placements and ground-truth eligibility before comparing top-only, two-, three- and five-LiDAR inputs. Neither reduced-camera masking nor temporal point accumulation is reported as a physical multi-LiDAR result.

References and exact source versions: [assessment source list](../SENSOR_CONFIGURATION_ASSESSMENT_2026-10-21.md#sources-and-evidence). Before final inclusion, convert these references to the dissertation's required citation style and align wording with any later experiment or scope decision.
