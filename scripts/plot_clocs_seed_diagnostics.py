"""Plot verified CLOCs seed and epoch diagnostics using the standard library."""
from pathlib import Path
import csv
import hashlib
import json

root = Path("results/clocs_seed_diagnostics_20261001")
load = lambda name: json.loads((root / name).read_text())
runs = [
    ("Seed 20260928, epoch 12", load("seed_20260928_ap40.json")),
    ("Seed 20260929, epoch 12", load("seed_20260929_ap40.json")),
    ("Seed 20260930, epoch 12", load("seed_20260930_ap40.json")),
    ("Seed 20260928, epoch 24", load("epoch24_ap40.json")),
]
for _, report in runs:
    assert report["status"] == "PASS"
    assert report["baseline_max_absolute_ap40_error"] == 0.0

with (root / "ap40_comparison.csv").open("w", newline="") as handle:
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(["run", "metric", "difficulty", "pointpillars", "fusion", "difference_ap_points"])
    for label, report in runs:
        ap = report["ap40_strict_iou_070"]
        for metric in ("2D", "BEV", "3D"):
            for difficulty in ("easy", "moderate", "hard"):
                baseline = ap["pointpillars"][metric][difficulty]
                fused = ap["fusion"][metric][difficulty]
                writer.writerow([label, metric, difficulty, baseline, fused, fused-baseline])

def begin(title, subtitle):
    return [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="560" viewBox="0 0 1000 560">',
        '<rect width="1000" height="560" fill="white"/>',
        '<g font-family="Arial,sans-serif" fill="#172033">',
        f'<text x="40" y="38" font-size="23">{title}</text>',
        f'<text x="40" y="66" font-size="14">{subtitle}</text>',
    ]

colors = ["#2563eb", "#d97706", "#059669", "#9333ea"]
svg = begin("CLOCs 3D AP40: difference from PointPillars",
            "3,769 KITTI validation frames; Car IoU 0.70; exact baseline parity in every run")
y = lambda value: 420 - (value + 8) / 10 * 290
for value in range(-8, 3, 2):
    yy = y(value)
    svg += [
        f'<line x1="100" y1="{yy}" x2="940" y2="{yy}" stroke="#dce1e8"/>',
        f'<text x="85" y="{yy+5}" text-anchor="end" font-size="13">{value:+d}</text>',
    ]
svg.append(f'<line x1="100" y1="{y(0)}" x2="940" y2="{y(0)}" stroke="#172033" stroke-width="2"/>')
for group, difficulty in enumerate(("easy", "moderate", "hard")):
    center = 240 + group * 280
    svg.append(f'<text x="{center}" y="451" text-anchor="middle" font-size="16">{difficulty.title()}</text>')
    for index, (_, report) in enumerate(runs):
        ap = report["ap40_strict_iou_070"]
        delta = ap["fusion"]["3D"][difficulty] - ap["pointpillars"]["3D"][difficulty]
        x = center - 100 + index * 52
        top, bottom = sorted((y(delta), y(0)))
        svg.append(f'<rect x="{x}" y="{top}" width="42" height="{bottom-top}" fill="{colors[index]}"/>')
        label_y = y(delta) - 7 if delta >= 0 else y(delta) + 17
        svg.append(f'<text x="{x+21}" y="{label_y}" text-anchor="middle" font-size="12">{delta:+.2f}</text>')
for index, (label, _) in enumerate(runs):
    x, yy = 100 + (index % 2) * 430, 492 + (index // 2) * 25
    svg += [f'<rect x="{x}" y="{yy-12}" width="14" height="14" fill="{colors[index]}"/>',
            f'<text x="{x+23}" y="{yy}" font-size="13">{label}</text>']
svg.append("</g></svg>")
(root / "three_seed_and_epoch24_3d_deltas.svg").write_text("\n".join(svg))

history = load("epoch24_history.json")
before = load("epoch24_before_extension_dev.json")
points = [(12, before["fusion_candidate_ap"])] + [
    (item["epoch"], item["dev"]["fusion_candidate_ap"]) for item in history
]
assert [epoch for epoch, _ in points] == list(range(12, 25))
assert all(item["learning_rates"] == [0.002] for item in history)
svg = begin("Epoch extension: development candidate AP",
            "Seed 20260928; fixed 32-frame development set; restored Adam; constant LR 0.002")
xx = lambda epoch: 110 + (epoch-12)/12 * 820
yy = lambda ap: 420 - (ap-.875)/.010 * 290
for ap in (.875, .8775, .880, .8825, .885):
    svg += [
        f'<line x1="110" y1="{yy(ap)}" x2="930" y2="{yy(ap)}" stroke="#dce1e8"/>',
        f'<text x="95" y="{yy(ap)+5}" text-anchor="end" font-size="13">{ap:.4f}</text>',
    ]
svg.append('<polyline fill="none" stroke="#2563eb" stroke-width="3" points="' +
           " ".join(f"{xx(epoch)},{yy(ap)}" for epoch, ap in points) + '"/>')
for epoch, ap in points:
    svg.append(f'<circle cx="{xx(epoch)}" cy="{yy(ap)}" r="4" fill="#2563eb"/>')
    if epoch % 2 == 0:
        svg.append(f'<text x="{xx(epoch)}" y="449" text-anchor="middle" font-size="13">{epoch}</text>')
svg += [
    '<text x="520" y="480" text-anchor="middle" font-size="15">Epoch</text>',
    '<text x="110" y="516" font-size="13">Truncated AP axis: 0.875–0.885. Candidate AP is not KITTI detection AP40.</text>',
    "</g></svg>",
]
(root / "epoch12_to24_candidate_ap.svg").write_text("\n".join(svg))

(root / "FINDINGS.md").write_text("""# CLOCs seed and epoch diagnostics — 1 October 2026

All four runs evaluated 3,769 validation frames with Car strict IoU 0.70 AP40.
PointPillars reproduced all nine frozen baseline metrics exactly.

At epoch 12, Moderate 3D AP40 was 79.8616, 76.2666 and 78.7616 across the
three seeds, against PointPillars at 79.1299. Only one seed improved Moderate;
all three reduced Hard AP. These are descriptive results, not significance tests.

For seed 20260928, extending training from epoch 12 to 24 increased development
candidate AP from 0.877935 to 0.883568. Adam state was restored, the learning rate
remained 0.002, and no scheduler was used. Epoch 24 was development-selected.
Moderate 3D AP40 nevertheless fell from 79.8616 to 79.4874; Hard fell from
74.8719 to 74.7797. Candidate-AP improvement did not translate into improved
3D detection AP40 in this extension. A training plateau remains unestablished.

The study used 96 optimization and 32 development frames. The original seed
selection used development AP before its full-validation run. Validation has
subsequently been inspected iteratively and is not an untouched final test.
The extension checkpoint lacked saved RNG states, so continuation is not
claimed to reproduce an uninterrupted run bitwise.

These results support a limitation of this small-data implementation and
protocol; they do not establish that CLOCs or camera–LiDAR fusion generally fails.
The epoch-24 run is a one-seed diagnostic, separate from the three-seed study.

Figure captions:
- three_seed_and_epoch24_3d_deltas.svg: Fusion minus paired PointPillars 3D AP40
  in AP points. Epoch-24 continuation is shown separately from epoch-12 seeds.
- epoch12_to24_candidate_ap.svg: Development candidate AP during continuation.
  The truncated vertical axis is explicit; this measure is not detection AP40.
""")

names = [
    "ap40_comparison.csv", "three_seed_and_epoch24_3d_deltas.svg",
    "epoch12_to24_candidate_ap.svg", "FINDINGS.md",
]
manifest = {
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "artifacts": {
        name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names
    },
}
(root / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("FIGURES, COMPARISON AND FINDINGS SAVED:", root)
