# Cost versus performance — evidence and remaining gaps

## Dated scenario analysis — 21 October 2026

The [sourced component analysis](20261021/README.md) now provides USD list-price observations, 22 component scenarios, 162 assumption-sensitivity rows and a standard-library reproduction script. It connects these accounting scenarios to the existing measured T4 accuracy/latency/memory trade-off without assigning dataset accuracy to replacement hardware. The [report discussion draft](../../docs/report/COST_PERFORMANCE_DISCUSSION_2026-10-21.md) uses one compact table and the existing runtime figure. A complete installed bill of materials, delivered UK quotations and monetary/energy efficiency remain unestablished. The dated framework below records the earlier starting point.

## Original framework and initial source register

This is a source register and analysis framework, started on 15 October 2026. The compute comparison is measured; a matched sensor bill of materials is not yet established. Do not present this document as a completed monetary cost/performance benchmark.

Update, 21 October 2026: the [sensor-configuration assessment](../../docs/SENSOR_CONFIGURATION_ASSESSMENT_2026-10-21.md) documents existing one-camera-plus-one-LiDAR KITTI evidence, candidate camera-only 3D paths and a deferred physical multi-LiDAR protocol. These scope recommendations do not complete financial sourcing or establish supervisor acceptance.

## Measured compute trade-off

The [nuScenes runtime study](../nuscenes/bevfusion/runtime/20261015/README.md) compares two official pretrained pipelines on the same 150 frames, with three isolated sessions per model. Median-of-session-medians latency is 287.81 ms for LiDAR-only and 544.98 ms for fusion (1.894×). Maximum peak allocated PyTorch GPU memory is 609.82 versus 2,231.80 MiB (3.660×). These are a cloud T4 protocol, not energy consumption, financial cost, or embedded-device performance.

The separately measured 600-frame accuracy comparison gives intact fusion +3.8242 mAP points and +1.8484 NDS points over the LiDAR-only checkpoint. The accuracy and timing sample sizes differ and the runtime TF32 policy is explicitly disabled; the accuracy-context records are preserved separately. No accuracy was recomputed in the runtime study.

## Hardware scenario matrix

Let C_L and C_C denote quote-based unit LiDAR and complete camera costs. Add integration, synchronization, calibration, compute, storage, power and maintenance separately. All expressions below are accounting scenarios, not measured deployment totals.

| Physical scenario | Sensor-only expression | Evidence status |
|---|---|---|
| One camera | C_C | No comparable camera-only nuScenes 3D benchmark yet |
| Two cameras | 2 C_C | No camera-only stereo/two-view 3D experiment; geometry and model must be specified |
| Six cameras | 6 C_C | Camera-only surround 3D remains unmeasured; documented pretrained candidates exist |
| One LiDAR | C_L | LiDAR pipeline measured using recorded dataset inputs |
| One LiDAR + one camera | C_L + C_C | KITTI CLOCs measured with mixed seed/difficulty outcomes; not a nuScenes or hardware deployment result |
| One LiDAR + two cameras | C_L + 2 C_C | Existing experiment masks four views of a six-view model; no purpose-trained two-camera or physical two-camera deployment result |
| One LiDAR + six cameras | C_L + 6 C_C | Recorded-input fusion accuracy and T4 compute measured |
| Three physical LiDARs | 3 C_L | No multi-LiDAR data/model experiment; temporal sweeps are not additional physical LiDARs |

The repeated C_L expression assumes identical unit prices. For a rig with different scanner types, use the sum of their individual sourced costs; Waymo's top and short-range units are distinct roles. No current quote-based totals are implied by this table.

The six-camera masking result cannot be assigned the physical two-camera sensor cost while claiming its execution is a two-camera deployment. All six view slots are still processed. Products with different scanning patterns, optics, coverage, synchronization and environmental protection cannot inherit the dataset's measured accuracy merely because their prices are known.

## Initial primary-source register

Accessed 15 October 2026.

1. [nuScenes data format and sensor setup](https://www.nuscenes.org/data-format): documents six cameras and one spinning Velodyne HDL32E LiDAR, alongside other sensors. This grounds physical sensor count; this experiment does not use the radar inputs. It does not supply a current hardware purchase quote.
2. [Ouster OS1 manufacturer page](https://ouster.com/products/hardware/os1): the pricing FAQ directs customers to obtain a quote, depending on type, resolution and configuration. A verified current numerical quote is therefore unavailable in this register. OS1 is a potential comparison product, not the dataset's HDL32E.
3. [Raspberry Pi Global Shutter Camera announcement, 9 March 2023](https://www.raspberrypi.com/news/new-raspberry-pi-global-shutter-camera/): historical launch price US$50. This is not a verified 2026 price, a complete camera-system cost, an automotive camera equivalent or a price for the dataset's cameras. Lenses and interfaces require separate sourcing.
4. [Livox Mid-360 manufacturer page](https://www.livoxtech.com/mid-360): documents a LiDAR intended for low-speed robotics. Its sensing geometry and data distribution differ from the evaluated dataset. The [DJI store](https://store.dji.com/uk/product/livox-mid-360) did not expose an accessible regional price during this check; no monetary estimate is recorded.

Historical press-release prices and unrelated hobby-camera prices must not be mixed into a current matched bill of materials.

## Next steps before a monetary conclusion

- Agree the deployment scenario with the supervisor: mobile robot, research vehicle or another specified application.
- Select comparable sensor products and record dated region/currency, quantity, taxes, included lenses, cables, interfaces and synchronization requirements.
- Obtain current prices or quotes; state missing values rather than inventing them. Supplier quotations remain to be obtained.
- Add compute and integration costs with explicit assumptions and sensitivity ranges.
- Report runtime/accuracy and hardware-accounting scenarios as distinct evidence. No energy efficiency or optimum physical sensor configuration has been demonstrated.


