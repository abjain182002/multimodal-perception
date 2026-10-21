"""Regenerate dated component scenarios; requires only the Python standard library.

Run from repository root: python scripts/build_cost_scenarios.py
No web requests, inference, or changes to original experiment evidence.
"""
import argparse
import csv
import hashlib
import io
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


CONFIGURATIONS = [
    ("1C", 1, 0), ("2C", 2, 0), ("6C", 6, 0), ("1L", 0, 1),
    ("1C+1L", 1, 1), ("2C+1L", 2, 1), ("6C+1L", 6, 1),
    ("2L", 0, 2), ("3L", 0, 3), ("1C+3L", 1, 3), ("6C+3L", 6, 3),
]


def money(value):
    return str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base = args.root / "results/cost_performance/20261021"
    output = args.output or base / "tables"
    output.mkdir(parents=True, exist_ok=True)
    spec = json.loads((base / "inputs/prices.json").read_text(encoding="utf-8"))
    sources = json.loads((base / "source_register.json").read_text(encoding="utf-8"))
    source_ids = {item["id"] for item in sources["sources"]}
    for price in spec["prices"].values():
        assert price["source_id"] in source_ids
        assert Decimal(price["amount"]) > 0
    assert spec["currency"] == "USD"
    prices = spec["prices"]
    lidar_price = Decimal(prices["lidar"]["amount"])
    variants = [
        ("USB_OAK1", Decimal(prices["camera_usb"]["amount"]),
         Decimal(prices["usb_cable"]["amount"])),
        ("PoE_OAK1", Decimal(prices["camera_poe"]["amount"]),
         Decimal(prices["poe_injector"]["amount"])),
    ]
    base_rows, sensitivity = [], []
    for name, nc, nl in CONFIGURATIONS:
        for variant, camera_price, interface_price in variants:
            sensor = nc * camera_price + nl * lidar_price
            interface = nc * interface_price
            subtotal = sensor + interface
            common = dict(configuration=name, cameras=nc, lidars=nl, camera_variant=variant,
                          currency="USD", source_date=spec["access_date"])
            base_rows.append(dict(common, sensors_usd=money(sensor),
                                  listed_camera_interfaces_usd=money(interface),
                                  priced_partial_subtotal_usd=money(subtotal),
                                  complete_system_cost="UNPRICED", measured_product_accuracy="UNMEASURED"))
            factors = spec["assumptions"]["sensor_price_lidar_multipliers"] if nl else ["1.00"]
            for factor_text in factors:
                factor = Decimal(factor_text)
                for fraction_text in spec["assumptions"]["subtotal_contingency_fractions"]:
                    fraction = Decimal(fraction_text)
                    changed = nc * (camera_price + interface_price) + nl * lidar_price * factor
                    sensitivity.append(dict(common, assumed_lidar_price_multiplier=factor_text,
                                             assumed_contingency_fraction=fraction_text,
                                             planning_partial_subtotal_usd=money(changed * (1 + fraction)),
                                             meaning="ASSUMPTION_SENSITIVITY_NOT_QUOTE_OR_TOTAL_COST"))
    assert len(base_rows) == 22 and len(sensitivity) == 162
    write_csv(output / "component_scenarios.csv", base_rows)
    write_csv(output / "assumption_sensitivity.csv", sensitivity)

    matrix_bytes = (base / "inputs/measured_accuracy_compute.csv").read_bytes()
    measured = list(csv.DictReader(io.StringIO(matrix_bytes.decode("utf-8"))))
    assert {r["model"] for r in measured} == {"lidar", "lidar_camera"}
    by_model = {r["model"]: r for r in measured}
    l, f = by_model["lidar"], by_model["lidar_camera"]
    assert all(r["accuracy_samples"] == "600" and r["runtime_samples"] == "150"
               and r["runtime_repeats"] == "3" for r in measured)
    value = lambda row, key: Decimal(row[key])
    latency_ratio = value(f, "median_of_session_medians_ms") / value(l, "median_of_session_medians_ms")
    memory_ratio = value(f, "max_peak_allocated_MiB") / value(l, "max_peak_allocated_MiB")
    deltas = {key + "_points": str((value(f, key) - value(l, key)) * 100) for key in ["mAP", "NDS"]}
    dominated = {}
    for row in measured:
        dominated[row["model"]] = any(
            value(other, "median_of_session_medians_ms") <= value(row, "median_of_session_medians_ms")
            and value(other, "NDS") >= value(row, "NDS")
            and (value(other, "median_of_session_medians_ms") < value(row, "median_of_session_medians_ms")
                 or value(other, "NDS") > value(row, "NDS"))
            for other in measured if other["model"] != row["model"])
    summary = {
        "status": "DATED_SCENARIO_ARITHMETIC_AND_MEASURED_CONTEXT_PASS",
        "source_date": spec["access_date"], "base_rows": len(base_rows),
        "sensitivity_rows": len(sensitivity), "accuracy_points": deltas,
        "latency_ratio": str(latency_ratio), "peak_allocated_memory_ratio": str(memory_ratio),
        "dominated_by_other_in_two_pipeline_nds_latency_context": dominated,
        "pareto_scope": "Two pipeline points with separately disclosed accuracy/timing samples; no global or financial frontier",
        "financial_accuracy_join_permitted": False,
        "complete_installed_cost_available": False,
        "assumption_intervals_are_supplier_quotes": False,
        "sources": {
            "measured_input_sha256": hashlib.sha256(matrix_bytes).hexdigest(),
            "price_input_sha256": hashlib.sha256((base / "inputs/prices.json").read_bytes()).hexdigest(),
            "source_register_sha256": hashlib.sha256((base / "source_register.json").read_bytes()).hexdigest(),
        },
    }
    (output / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# Reproduced component subtotals (USD)", "",
             "These are partial priced components; compute, integration and delivered costs remain unpriced.", "",
             "| Configuration | Sensors: USB option | Sensors: PoE option | With listed camera interfaces: USB | With listed camera interfaces: PoE |",
             "|---|---:|---:|---:|---:|"]
    for name, _, _ in CONFIGURATIONS:
        a, b = [r for r in base_rows if r["configuration"] == name]
        lines.append(f'| {name} | {a["sensors_usd"]} | {b["sensors_usd"]} | {a["priced_partial_subtotal_usd"]} | {b["priced_partial_subtotal_usd"]} |')
    (output / "component_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
