#!/usr/bin/env python3
"""Check frozen aggregate BAFT evidence and optionally regenerate report CSVs.

Uses Python's standard library only. This checks aggregate consistency, not
independent reproduction of training, private predictions, or user bootstraps.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from collections import defaultdict


MODELS = {
    "aggregate_gbdt": "Aggregate GBDT",
    "gru": "GRU multitask",
    "transformer": "Transformer",
    "time_transformer": "Time-aware Transformer",
    "baft_latent": "BAFT latent only",
    "baft_accounting": "BAFT accounting only",
    "baft_full": "BAFT full",
}
SEEDS = tuple(range(20260819, 20260824))
PRIMARY = ("aggregate_gbdt", "gru", "transformer", "baft_full")
NATIVE = "validation_gate_native"
PROJECTED = "validation_gate_hard_projection"
REPORT_METRICS = {
    "Accuracy, % (mean ± seed SD)": (NATIVE, "accuracy", 100),
    "Macro-F1, % (mean ± seed SD)": (NATIVE, "macro_f1", 100),
    "Top-3, % (mean ± seed SD)": (NATIVE, "top3", 100),
    "Feasible, % (mean ± seed SD)": (NATIVE, "predicted_feasible_rate", 100),
    "FVA, % (mean ± seed SD)": (NATIVE, "financially_valid_accuracy", 100),
    "ECE, % (mean ± seed SD)": (NATIVE, "ece", 100),
    "Balance MAE": (NATIVE, "balance_mae", 1),
    "Expected accounting residual MAE": (NATIVE, "expected_accounting_residual_mae", 1),
    "Operational accounting residual MAE": (NATIVE, "operational_accounting_residual_mae", 1),
    "Native SPS*, %": (NATIVE, "operational_sps_prior", 100),
    "Latent acceleration MSE": ("latent_native", "latent_acceleration_mse", 1),
    "Projected balance MAE": (PROJECTED, "balance_mae", 1),
    "Projected operational residual MAE": (PROJECTED, "operational_accounting_residual_mae", 1),
}
EFFECT_FIELDS = {
    "BAFT minus Transformer": "absolute_effect",
    "Seed-effect 95% CI low": "seed_effect_ci95_low",
    "Seed-effect 95% CI high": "seed_effect_ci95_high",
    "User-cluster 95% CI low": "user_effect_ci95_low",
    "User-cluster 95% CI high": "user_effect_ci95_high",
    "Cohen dz": "cohen_dz",
    "Paired seed t p": "seed_t_p",
    "Exact Wilcoxon p": "seed_wilcoxon_exact_p",
    "Exact sign-flip p": "seed_signflip_exact_p",
    "User-cluster p": "cluster_centered_bootstrap_p",
    "Holm seed-t p": "seed_t_p_holm",
    "Holm user-cluster p": "cluster_centered_bootstrap_p_holm",
    "Noninferiority p (one-sided)": "noninferiority_seed_t_p_one_sided",
}
EFFECT_SPECS = (
    (NATIVE, "accuracy"),
    (NATIVE, "financially_valid_accuracy"),
    (NATIVE, "operational_accounting_residual_mae"),
    (NATIVE, "balance_mae"),
    ("latent_native", "latent_acceleration_mse"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-10),
            f"{label}: {actual!r} != {expected!r}")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def indexed(rows, columns):
    output = {}
    for row in rows:
        key = tuple(row[column] for column in columns)
        require(key not in output, f"Duplicate row: {key}")
        output[key] = row
    return output


def check_hashes(root):
    manifest = json.loads((root / "reference_manifest.json").read_text())
    listed = set()
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        require(not relative.is_absolute() and ".." not in relative.parts,
                f"Manifest path must be relative: {relative}")
        require(relative.as_posix() not in listed, f"Duplicate manifest entry: {relative}")
        listed.add(relative.as_posix())
        path = root / relative
        require(not path.is_symlink(), f"Unexpected symlink: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == entry["sha256"], f"Checksum mismatch: {relative}")
    actual_files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    require(actual_files == listed | {"reference_manifest.json"},
            "Reference directory contains unlisted or missing files")
    return len(listed)


def check_seed_summaries(seed_rows, frozen):
    seed_values = {}
    grouped = defaultdict(list)
    for row in seed_rows:
        key = (row["model"], row["condition"], row["metric"])
        seed_key = (*key, int(row["seed"]))
        require(seed_key not in seed_values, f"Duplicate seed metric: {seed_key}")
        value = float(row["value"])
        require(math.isfinite(value), f"Nonfinite seed metric: {seed_key}")
        seed_values[seed_key] = value
        grouped[key].append((int(row["seed"]), value))
    require({row["model"] for row in seed_rows} == set(MODELS), "Expected seven models")
    require(set(grouped) == set(frozen), "Seed and frozen metric keys differ")
    require(len(seed_rows) == 2385 and len(frozen) == 477, "Frozen release metric count differs")
    for key, values in grouped.items():
        require(tuple(sorted(seed for seed, _ in values)) == SEEDS,
                f"Incomplete five-seed coverage: {key}")
        row = frozen[key]
        require(row["n_seeds"] == "5", f"Seed count differs: {key}")
        require(row["seed_list"] == ";".join(map(str, SEEDS)), f"Seed list differs: {key}")
        require(row["n_users"] == "5253" and row["n_sequences"] == "8850",
                f"Gold test cohort counts differ: {key}")
        numbers = [value for _, value in values]
        close(row["estimate"], statistics.mean(numbers), f"Seed mean {key}")
        close(row["seed_sd"], statistics.stdev(numbers), f"Sample seed SD {key}")
    return seed_values


def check_effects(effects, values):
    for row in effects:
        condition, metric = row["condition"], row["metric"]
        differences = [values[(row["reference"], condition, metric, seed)]
                       - values[(row["comparator"], condition, metric, seed)] for seed in SEEDS]
        mean = statistics.mean(differences)
        sd = statistics.stdev(differences)
        label = (row["reference"], row["comparator"], condition, metric)
        require(row["paired_seeds"] == "5", f"Paired seed count: {label}")
        close(row["absolute_effect"], mean, f"Paired mean {label}")
        close(row["paired_difference_sd"], sd, f"Paired SD {label}")
        if sd > 1e-14:
            close(row["cohen_dz"], mean / sd, f"Cohen dz {label}")
            # Exact closed form of the two-sided Student t p-value with df=4.
            t = abs(mean / (sd / math.sqrt(5)))
            u = t / math.sqrt(t * t + 4)
            p = 1 - 1.5 * u + 0.5 * u ** 3
            close(row["seed_t_p"], p, f"Paired seed t p {label}")
            half_width = 2.7764451051977987 * sd / math.sqrt(5)
            close(row["seed_effect_ci95_low"], mean - half_width, f"Seed CI low {label}")
            close(row["seed_effect_ci95_high"], mean + half_width, f"Seed CI high {label}")
    family = [row for row in effects if row["analysis_plan_family"] == "BAFT_vs_Transformer_six_endpoints"]
    require(len(family) == 6, "Expected the original six-endpoint analysis-stage family")
    for column in ("seed_t_p", "seed_signflip_exact_p", "cluster_centered_bootstrap_p"):
        available = sorted((row for row in family if row[column]), key=lambda row: float(row[column]))
        previous = 0.0
        for rank, row in enumerate(available):
            expected = min(1.0, max(previous, (len(available) - rank) * float(row[column])))
            close(row[column + "_holm"], expected, f"Holm {column}/{row['metric']}")
            previous = expected


def check_auxiliary(root, frozen, values):
    eta = read_csv(root / "frozen_evidence/validation_selected_eta.csv")
    expected_pairs = {(model, str(seed)) for model in MODELS for seed in SEEDS}
    require(set(indexed(eta, ("model", "seed"))) == expected_pairs,
            "Validation eta matrix must cover seven models and five seeds")
    for row in read_csv(root / "frozen_evidence/presentation_points_wide.csv"):
        require(row["n_seeds"] == "5", "Wide summary seed count differs")
        for metric, value in row.items():
            if metric not in {"model", "condition", "n_seeds"} and value:
                close(value, frozen[(row["model"], row["condition"], metric)]["estimate"],
                      f"Wide summary {row['model']}/{row['condition']}/{metric}")
    for row in read_csv(root / "frozen_evidence/constraint_transfer_effects.csv"):
        model, metric = row["model"], row["metric"]
        before = float(frozen[(model, row["from_condition"], metric)]["estimate"])
        after = float(frozen[(model, row["to_condition"], metric)]["estimate"])
        close(row["absolute_effect_after_minus_before"], after - before,
              f"Constraint transfer {model}/{metric}")
    latent = read_csv(root / "latent_stability/latent_stability_seed.csv")
    require(len(latent) == 30, "Expected six neural models by five latent seeds")
    latent_metrics = ("latent_acceleration_mse", "velocity_change_l2_mean", "transition_norm_mean",
                      "bound_excess_mse", "bound_excess_mean", "bound_violation_rate")
    for row in latent:
        for metric in latent_metrics:
            close(row[metric], values[(row["model"], "latent_native", metric, int(row["seed"]))],
                  f"Latent seed {row['model']}/{row['seed']}/{metric}")
    for row in read_csv(root / "latent_stability/latent_stability_summary.csv"):
        for metric in latent_metrics:
            source = frozen[(row["model"], "latent_native", metric)]
            close(row[metric + "_mean"], source["estimate"], f"Latent mean {row['model']}/{metric}")
            close(row[metric + "_sd"], source["seed_sd"], f"Latent SD {row['model']}/{metric}")


def check_benchmarks(seed_rows, summary_rows, training):
    grouped = defaultdict(list)
    for row in seed_rows:
        grouped[(row["model"], row["mode"])].append(row)
    require(len(seed_rows) == 70 and len(summary_rows) == 14, "Inference benchmark coverage differs")
    require(set(grouped) == {(model, mode) for model in MODELS
                           for mode in ("gold_test_throughput", "batch1_latency")},
            "Expected two inference modes for seven models")
    for row in summary_rows:
        rows = grouped[(row["model"], row["mode"])]
        require(tuple(sorted(int(item["seed"]) for item in rows)) == SEEDS,
                "Inference benchmark seeds differ")
        require(row["seeds"] == "5", "Inference benchmark summary seed count differs")
        for column, value in row.items():
            suffix = next((ending for ending in ("_across_seed_median", "_across_seed_q1", "_across_seed_q3")
                           if column.endswith(ending)), None)
            if suffix and value:
                metric = column[:-len(suffix)]
                ordered = sorted(float(item[metric]) for item in rows)
                # Five values: the linear 25%, 50%, 75% quantiles are observed order statistics.
                position = {"_across_seed_q1": 1, "_across_seed_median": 2, "_across_seed_q3": 3}[suffix]
                close(value, ordered[position], f"Benchmark aggregation {row['model']}/{column}")
    require({row["model"] for row in training} == set(PRIMARY) and len(training) == 4,
            "Expected four primary training benchmarks")
    require(all(row["seed"] == "20260908" for row in training), "Training benchmark seed differs")


def report_tables(frozen, effects, inference, training):
    model_rows = []
    for model, display in MODELS.items():
        row = {"Model": display}
        for column, (condition, metric, scale) in REPORT_METRICS.items():
            if model == "aggregate_gbdt" and condition == "latent_native":
                row[column] = "n/a"
                continue
            source = frozen[(model, condition, metric)]
            row[column] = f"{float(source['estimate']) * scale:.2f} ± {float(source['seed_sd']) * scale:.2f}"
        model_rows.append(row)
    effect_index = indexed(effects, ("reference", "comparator", "condition", "metric"))
    effect_rows = []
    for condition, metric in EFFECT_SPECS:
        source = effect_index[("baft_full", "transformer", condition, metric)]
        row = {"Endpoint": metric, "Condition": condition}
        row.update({display: source[column] for display, column in EFFECT_FIELDS.items()})
        effect_rows.append(row)
    inference_index = indexed(inference, ("model", "mode"))
    training_index = indexed(training, ("model",))
    compute_rows = []
    for model in PRIMARY:
        train = training_index[(model,)]
        batch = inference_index[(model, "gold_test_throughput")]
        latency = inference_index[(model, "batch1_latency")]
        compute_rows.append({
            "Model": MODELS[model],
            "End-to-end training wall time, s": train["external_end_to_end_wall_seconds"],
            "Parameters": train["parameter_count"],
            "Peak RSS, MB": train["peak_rss_mb"],
            "Full-pipeline throughput, examples/s": batch["throughput_examples_per_second_median_across_seed_median"],
            "Batch-1 full-pipeline latency, ms/example": latency["latency_ms_per_example_median_across_seed_median"],
            "Gate time on full cohort, ms": float(batch["gate_seconds_median_across_seed_median"]) * 1000,
            "Projection time on full cohort, ms": float(batch["projection_seconds_median_across_seed_median"]) * 1000,
        })
    return {
        "FROZEN_ALL_MODELS_RESULTS.csv": model_rows,
        "FROZEN_PRIMARY_RESULTS.csv": [row for row in model_rows if row["Model"] in {MODELS[m] for m in PRIMARY}],
        "FROZEN_BAFT_VS_TRANSFORMER_EFFECTS.csv": effect_rows,
        "FROZEN_COMPUTE_BENCHMARK.csv": compute_rows,
    }


def check_reports(root, reports):
    for name, expected in reports.items():
        actual = read_csv(root / "reports" / name)
        require(len(actual) == len(expected), f"Report row count differs: {name}")
        for row, source in zip(actual, expected):
            require(list(row) == list(source), f"Report columns differ: {name}")
            for column, value in source.items():
                if str(row[column]) != str(value):
                    close(row[column], value, f"Report {name}/{row.get('Model', row.get('Endpoint'))}/{column}")
    primary = {row["Model"]: row for row in reports["FROZEN_PRIMARY_RESULTS.csv"]}
    require(primary["BAFT full"]["Accuracy, % (mean ± seed SD)"] == "54.60 ± 0.26", "BAFT accuracy headline differs")
    require(primary["Transformer"]["Accuracy, % (mean ± seed SD)"] == "54.66 ± 0.36", "Transformer accuracy headline differs")
    require(primary["BAFT full"]["Operational accounting residual MAE"] == "9.75 ± 0.94", "BAFT residual headline differs")
    require(primary["BAFT full"]["Latent acceleration MSE"] == "2.19 ± 0.46", "BAFT latent headline differs")
    effects = {row["Endpoint"]: row for row in reports["FROZEN_BAFT_VS_TRANSFORMER_EFFECTS.csv"]}
    close(effects["accuracy"]["Paired seed t p"], 0.7247476414759692, "Accuracy paired seed t p")
    close(effects["accuracy"]["Holm user-cluster p"], 0.6616676664667066, "Accuracy Holm user-cluster p")
    close(effects["financially_valid_accuracy"]["User-cluster p"], 0.2433513297340531, "FVA unadjusted user-cluster p")
    close(effects["financially_valid_accuracy"]["Holm user-cluster p"], 0.4867026594681063, "FVA Holm user-cluster p")


def check_reference(root):
    file_count = check_hashes(root)
    frozen_rows = read_csv(root / "frozen_evidence/frozen_results_table.csv")
    frozen = indexed(frozen_rows, ("model", "condition", "metric"))
    seed_rows = read_csv(root / "frozen_evidence/seed_metrics_long.csv")
    values = check_seed_summaries(seed_rows, frozen)
    effects = read_csv(root / "frozen_evidence/paired_effects_and_tests.csv")
    check_effects(effects, values)
    check_auxiliary(root, frozen, values)
    inference = read_csv(root / "benchmarks/inference_benchmark_summary.csv")
    training = read_csv(root / "benchmarks/training_benchmark.csv")
    check_benchmarks(read_csv(root / "benchmarks/inference_benchmark_seed.csv"), inference, training)
    reports = report_tables(frozen, effects, inference, training)
    check_reports(root, reports)
    return reports, file_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results" / "reference")
    parser.add_argument("--export-dir", type=Path, help="After successful checks, regenerate the four report CSVs here")
    args = parser.parse_args()
    try:
        reports, count = check_reference(args.reference_dir)
        if args.export_dir:
            reference_path = args.reference_dir.resolve()
            export_path = args.export_dir.resolve()
            require(export_path != reference_path and reference_path not in export_path.parents,
                    "Export outside the reference directory to preserve the frozen evidence")
            args.export_dir.mkdir(parents=True, exist_ok=True)
            for name, rows in reports.items():
                with (args.export_dir / name).open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(f"PASS: {count} checksummed files; 7 models × 5 seeds; 2,385 seed metrics; 477 summaries; 4 report tables.")
    print("Checked paired effects, seed t tests, Holm adjustments, latent summaries, and benchmark aggregates.")
    print("User-cluster intervals/tests are preserved source results; private event data are required to recompute them.")
    if args.export_dir:
        print(f"Exported 4 report CSVs to {args.export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
