# Cost versus performance: dated component scenarios

Date: 21 October 2026. This completes a bounded, reproducible scenario analysis using sourced list prices and existing compute/accuracy records. A complete installed-system quotation, physical deployment comparison and monetary optimum remain unestablished.

## Interpretation and scenario

The illustrative application is a research mobile robot using purchased RGB cameras and a compact 3D LiDAR. This is an analytical scenario, not a supervisor-approved deployment design. It is grounded in the project's robotic-perception question and Mid-360's manufacturer positioning for low-speed robotics. The evaluated KITTI and nuScenes equipment differs from these products, so their detection scores cannot be assigned to these priced rigs.

Two questions are answered with different evidence: how much additional computation the tested fusion pipeline required, and how sensor-count choices change a transparent component budget. They do not jointly establish accuracy per dollar for real hardware. Product selection, field of view, temporal data, sensor response and model compatibility would need deployment validation.

## Observed prices and source quality

All amounts below are US dollars observed on public pages on 21 October 2026. No checkout, purchase or formal quotation request was made. The [source register](source_register.json) records URLs, provenance, exclusions and inaccessible prices. The observations are not guaranteed current stock or delivered UK prices.

| Item | Observed amount | Source and qualification |
|---|---:|---|
| OAK-1 standard, SKU A00109 | 269.00 | Luxonis manufacturer store [C1]; USB camera device with onboard processing |
| OAK-1 PoE standard, SKU A00124 | 379.00 | Luxonis manufacturer store [C2]; different interface/enclosure product |
| USB C/A screw-locking cable | 14.99 | Luxonis manufacturer store [C3]; listed separately |
| PoE injector, U-POE | 24.99 | Luxonis manufacturer store [C4]; listed separately |
| Livox Mid-360 | 899.00 | Robotics Center retailer reference merchandise listing [C5]; backorder, price subject to confirmation |
| Ouster OS1 | Not established | Manufacturer requests a configuration-specific quotation [C6] |

The USB and PoE camera prices define two product options, not a statistical price interval for an identical camera. The OAK processors are included in the purchase price but have not been shown to run PointPillars, CLOCs, ImVoxelNet or BEVFusion; they do not replace the measured T4 by assumption.

The Mid-360 seller separately displays shipping and tax qualifications. Only its merchandise reference price is used. Its technical specification table includes a weight inconsistent with the manufacturer's page, so no technical performance claim is taken from that retailer record. Ouster's older public price announcements and Livox's US$749 trial-sample launch price from 10 January 2023 are not treated as current prices. A different retailer's US$3,800 sold-out Mid-360 listing was inspected but excluded: package/terms were not reconciled, so it is not a comparable upper bound. DJI's UK product URL redirected to another region and exposed no usable price. These exclusions prevent a false claim of a verified market-price range.

## Component scenarios

Assumptions: quantity-one unit prices are multiplied without discounts; each LiDAR is the same US$899 Mid-360 reference item; the camera option is fixed within a scenario. These assumptions do not represent Waymo's heterogeneous scanner rig. No detection accuracy has been measured with these products.

| Configuration | Sensor devices: USB option | Sensor devices: PoE option | Devices + listed camera interfaces: USB | Devices + listed camera interfaces: PoE |
|---|---:|---:|---:|---:|
| One camera | 269.00 | 379.00 | 283.99 | 403.99 |
| Two cameras | 538.00 | 758.00 | 567.98 | 807.98 |
| Six cameras | 1,614.00 | 2,274.00 | 1,703.94 | 2,423.94 |
| One LiDAR | 899.00 | 899.00 | 899.00 | 899.00 |
| One camera + one LiDAR | 1,168.00 | 1,278.00 | 1,182.99 | 1,302.99 |
| Two cameras + one LiDAR | 1,437.00 | 1,657.00 | 1,466.98 | 1,706.98 |
| Six cameras + one LiDAR | 2,513.00 | 3,173.00 | 2,602.94 | 3,322.94 |
| Two LiDARs | 1,798.00 | 1,798.00 | 1,798.00 | 1,798.00 |
| Three LiDARs | 2,697.00 | 2,697.00 | 2,697.00 | 2,697.00 |

The [generated table](tables/component_table.md) and [22-row CSV](tables/component_scenarios.csv) also include camera-plus-three-LiDAR scenarios. A single camera increment adds US$269/379 in device price under the two options, while an additional identical LiDAR adds US$899. This is an accounting consequence of the chosen prices. It does not establish proportional coverage, accuracy or a defensible choice of sensor placement.

The interface columns assume one cable per USB camera or one injector per PoE camera. For a multi-camera rig, powered hubs, network switches, Ethernet cabling, triggering and power conversion require a design and further prices. A mains-powered injector is not a complete battery-powered robot solution. Calibration, mounting, host compute, storage, integration labour, shipping, taxes, maintenance and energy are unpriced additive items. They have not been set to zero.

## Sensitivity and missing costs

The model uses:

`P = n_C*(camera unit price + listed camera interface) + n_L*LiDAR reference price`

`planning partial subtotal = (n_C*(camera + interface) + n_L*LiDAR reference price*q)*(1+a)`

Here `q` is an assumed LiDAR-price multiplier of 0.75, 1.00 or 1.25, and `a` is an assumed contingency of 0%, 15% or 30%. These are transparent stress tests selected for analysis, not quotations, industry-standard integration percentages or confidence intervals. Camera-only rows omit the irrelevant LiDAR variation. The script produces [162 sensitivity rows](tables/assumption_sensitivity.csv).

For example, one USB camera plus one LiDAR has a priced partial subtotal of US$1,182.99. A 15% or 30% contingency on that fixed-price subtotal gives US$1,360.44 or US$1,537.89, respectively. It still excludes the unpriced items. A percentage contingency cannot certify enough budget for calibration or software labour, especially when a second sensor changes interfaces and integration requirements. A complete capital budget would add those actual costs; operating cost would additionally need a declared lifetime, duty cycle, electricity use, maintenance and any recurring compute charges.

## Measured accuracy and computing trade-off

The following values come from the preserved [600-frame accuracy context and 150-frame runtime matrix](../../nuscenes/bevfusion/runtime/20261015/tables/accuracy_compute_matrix.csv). They are not performance measurements of the components priced above.

| Pipeline | Accuracy: mAP / NDS ×100, 600 frames | T4 median latency, ms | Peak allocated GPU memory, MiB |
|---|---:|---:|---:|
| LiDAR-only | 64.3076 / 69.4430 | 287.8116 | 609.8184 |
| Six-camera fusion | 68.1318 / 71.2913 | 544.9786 | 2,231.7979 |

The timing uses one T4, batch one, FP32, no autocast, TF32 disabled, one CPU thread and three fresh sessions/model over 150 selected frames after 20 warm-ups. The median is the median of session medians; memory is the maximum PyTorch peak allocated value. Timing covers `test_step` and excludes CPU file loading/collation. Accuracy was measured separately on 600 frames, rather than rerun on the 150 timing frames, and its runtime settings were not made identical retrospectively. These scope differences must remain in any combined table or plot.

Within that disclosed context, fusion adds 3.8242 mAP points and 1.8484 NDS points while requiring 1.8935 times the latency and 3.6598 times the peak allocated memory. Among these two points, neither dominates the other on NDS versus latency: one has higher accuracy, the other lower latency. That two-point observation does not establish a global Pareto frontier or a monetary optimum. Missing camera-only and multi-LiDAR accuracy/latency cannot be estimated from sensor prices or interpolated by count.

The full 6,019-frame nuScenes scores remain a separate accuracy result: 64.4301/69.2353 for LiDAR and 68.3099/71.1352 for fusion. Do not substitute those values into a row labelled 600 frames, or pool KITTI AP40, nuScenes NDS and differently bounded PV-RCNN timings into a common ranking. Published hardware FPS/TOPS, T4 allocator memory and sensor datasheet power are also distinct measurements; no energy or embedded inference result has been collected here.

## Implication for the supervisor's question

Existing experiments answer whether the evaluated fused pipeline improves detection and how it behaves under specific missing/degraded inputs. They also expose a measured computation penalty. The component analysis shows the financial questions that a proposed one-, two- or six-camera rig and an additional physical LiDAR would raise. It does not identify which physical configuration is best.

A defensible design choice would first specify the task's coverage, range, accuracy, latency, power and environmental requirements, then test suitable hardware/model combinations and price the resulting integration. The next experimental candidate remains the bounded [camera-only KITTI feasibility preflight](../../../docs/SENSOR_CONFIGURATION_ASSESSMENT_2026-10-21.md); the present analysis neither requires another GPU run nor proves that candidate fits available resources. Physical multi-LiDAR remains a documented follow-on study.

## Reproduction and completion boundary

Run from repository root:

```bash
python scripts/build_cost_scenarios.py
```

The standard-library script reads dated [price assumptions](inputs/prices.json), [source records](source_register.json) and a value-preserving copy of the measured CSV (line endings normalised), and writes two CSVs, a Markdown table and a JSON analysis summary. It recalculates price arithmetic, accuracy deltas, latency/memory ratios and the limited dominance check. It makes no web requests and performs no inference. Original experimental files are unchanged. The input copy's values were compared with the GitHub original before publication.

The sourced scenario analysis and report discussion are complete at this stated scope. Still unverified: current delivered UK quotations, a full installed bill of materials, measured financial/energy efficiency, hardware deployment and supervisor acceptance of the illustrative application. These omissions are explicit boundaries, not invented zero costs.

Use one compact component table plus the existing runtime/accuracy figure in the 40–50-page report. Supporting sensitivity rows and source details can remain in the repository; no additional figure was added.
