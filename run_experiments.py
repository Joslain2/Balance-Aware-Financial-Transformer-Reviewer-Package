#!/usr/bin/env python3
"""Run the recovered BAFT pipeline with explicit inputs and content-checked restarts.

The launcher uses only the Python standard library. Scientific dependencies are
needed by executed stages, but are not imported for --help or --dry-run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "src" / "baft"
STAGES = ("clean", "validate", "cache", "train", "predictions", "latent", "evidence",
          "inference-benchmark", "training-benchmark")
PAPER_MODELS = ["aggregate_gbdt", "gru", "transformer", "time_transformer",
                "baft_latent", "baft_accounting", "baft_full"]
PAPER_SEEDS = list(range(20260819, 20260824))
MODELS = set(PAPER_MODELS) | {"profile_transformer", "baft_full_profile"}


@dataclass
class Action:
    stage: str
    name: str
    command: list[str]
    inputs: list[Path]
    outputs: list[Path]


def options(settings: dict) -> list[str]:
    return [item for key, value in settings.items()
            for item in ("--" + key.replace("_", "-"), str(value))]


def command(script: str, *arguments: object) -> list[str]:
    return [sys.executable, str(CODE / script), *map(str, arguments)]


def read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"name", "purpose", "paper_protocol", "stages", "models", "seeds", "threads",
                "cleaning", "cache", "training", "predictions", "latent", "evidence",
                "inference_benchmark", "training_benchmark"}
    if set(config) != required:
        raise ValueError(f"Config fields differ: missing={required - set(config)}, extra={set(config) - required}")
    models, seeds = config["models"], config["seeds"]
    if not models or len(models) != len(set(models)) or not set(models) <= MODELS:
        raise ValueError("Config must contain distinct supported model names")
    if not seeds or len(seeds) != len(set(seeds)) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Config seeds must be distinct nonnegative integers")
    if config["threads"] < 1 or not set(config["stages"]) <= set(STAGES):
        raise ValueError("Invalid threads or stages in config")
    if not any(model != "aggregate_gbdt" for model in models):
        raise ValueError("The latent/evidence pipeline requires at least one neural model")
    if config["paper_protocol"]:
        canonical = json.loads((ROOT / "configs" / "paper.json").read_text(encoding="utf-8"))
        if config != canonical or models != PAPER_MODELS or seeds != PAPER_SEEDS:
            raise ValueError("paper_protocol=true requires the unmodified paper configuration")
    return config


def build_actions(config: dict, source: Path | None, output: Path) -> list[Action]:
    clean = output / "cleaned"
    cache = output / "gold_cache" / "baft_gold_cache.npz"
    metadata = cache.with_name("baft_gold_cache_metadata.json")
    runs = output / "model_runs"
    regenerated = output / "regenerated" / "model_runs"
    latent = output / "results" / "latent"
    evidence = output / "results" / "evidence"
    clean_products = [clean / name for name in (
        "mobile_money_cleaned_transactions.parquet", "baft_sequence_manifest.parquet",
        "excluded_rows.parquet", "cleaning_metadata.json", "quality_rule_counts.csv",
        "service_quality_profile.csv", "sequence_cohort_summary.csv",
        "gold_sequence_class_distribution.csv", "accounting_exceptions.csv")]
    stems = [f"{model}_seed{seed}" for model in config["models"] for seed in config["seeds"]]
    checkpoints = [runs / (stem + (".joblib" if stem.startswith("aggregate_gbdt_") else ".pt")) for stem in stems]
    records = [runs / f"{stem}.json" for stem in stems]
    prediction_products = [regenerated / f"{stem}_gold_test_predictions.npz" for stem in stems]
    cached = [cache, metadata]
    actions = []
    if source is not None:
        actions.append(Action("clean", "clean", command("clean_mobile_money_dataset.py", "--input", source,
            "--output-dir", clean, *options(config["cleaning"])), [source], clean_products))
    actions.append(Action("validate", "validate", command("validate_cleaned_dataset.py", "--dataset-dir", clean),
                          clean_products[:4], []))
    actions.append(Action("cache", "cache", command("build_gold_cache.py", "--clean-dir", clean,
        "--output", cache, "--metadata-output", metadata, *options(config["cache"])), clean_products[:4], cached))
    for model in config["models"]:
        for seed in config["seeds"]:
            stem = f"{model}_seed{seed}"
            arguments = ["--cache", cache, "--metadata", metadata, "--seed", seed, "--output-dir", runs]
            if model == "aggregate_gbdt":
                invocation = command("train_gold_gbdt.py", *arguments)
                checkpoint = runs / f"{stem}.joblib"
            else:
                invocation = command("train_gold_neural.py", *arguments, "--model", model,
                                     "--threads", config["threads"], *options(config["training"]))
                checkpoint = runs / f"{stem}.pt"
            actions.append(Action("train", "train_" + stem, invocation, cached,
                [checkpoint, runs / f"{stem}.json", runs / f"{stem}_gold_test_predictions.npz"]))
    actions.append(Action("predictions", "predictions", command("regenerate_predictions.py",
        "--runs-dir", regenerated, "--cache", cache, "--metadata", metadata,
        "--models", *config["models"], "--seeds", *config["seeds"], "--threads", config["threads"],
        *options(config["predictions"])), cached + checkpoints + records, prediction_products))
    latent_products = [latent / name for name in ("latent_stability_seed.csv", "latent_stability_events.parquet",
        "latent_stability_summary.csv", "latent_stability_paired_tests.csv", "latent_stability_manifest.json")]
    actions.append(Action("latent", "latent", command("evaluate_latent_stability.py", "--cache", cache,
        "--metadata", metadata, "--runs-dir", runs, "--output-dir", latent,
        "--threads", config["threads"], *options(config["latent"])),
        cached + [path for path in checkpoints if path.suffix == ".pt"], latent_products))
    actions.append(Action("evidence", "evidence", command("finalize_evidence.py", "--runs-dir", regenerated,
        "--cache", cache, "--metadata", metadata, "--latent-results", latent, "--output-dir", evidence,
        *options(config["evidence"])), cached + prediction_products + latent_products,
        [evidence / name for name in ("frozen_results_table.csv", "seed_metrics_long.csv",
         "paired_effects_and_tests.csv", "constraint_transfer_effects.csv", "validation_selected_eta.csv",
         "presentation_points_wide.csv", "metric_definitions_and_manifest.json", "source_hashes.json")]))
    benchmark = output / "results" / "inference_benchmark"
    benchmark_driver = output / "benchmark_code" / "driver" / "benchmark_models.py"
    actions.append(Action("inference-benchmark", "inference_benchmark", [sys.executable, str(benchmark_driver), *map(str, (
        "--cache", cache, "--metadata", metadata, "--checkpoints-dir", runs, "--output-dir", benchmark,
        "--models", *config["models"], "--seeds", *config["seeds"], "--threads", config["threads"],
        *options(config["inference_benchmark"])))], cached + checkpoints + records,
        [benchmark / f"inference_benchmark_{name}" for name in ("seed.csv", "summary.csv", "manifest.json")]
        + [benchmark / "worker_results" / f"{stem}.json" for stem in stems]))
    benchmark_training = output / "results" / "training_benchmark"
    settings = config["training_benchmark"]
    actions.append(Action("training-benchmark", "training_benchmark", command("run_standard_training_benchmark.py",
        "--cache", cache, "--metadata", metadata, "--code-dir", CODE, "--output-dir", benchmark_training,
        "--models", *settings["models"], "--threads", config["threads"],
        *options({key: value for key, value in settings.items() if key != "models"})), cached,
        [benchmark_training / "training_benchmark.csv", benchmark_training / "training_benchmark_manifest.json"]
        + [benchmark_training / model / f"{model}_seed{settings['seed']}{suffix}"
           for model in settings["models"] for suffix in
           (".json", "_gold_test_predictions.npz", ".joblib" if model == "aggregate_gbdt" else ".pt")]
        + [benchmark_training / model / filename for model in settings["models"]
           for filename in ("benchmark_stdout.txt", "benchmark_stderr.txt")]))
    return actions


class Fingerprints:
    """Avoid repeatedly hashing unchanged multi-gigabyte inputs in one invocation."""

    def __init__(self) -> None:
        self.memo: dict[tuple, str] = {}

    def file(self, path: Path) -> str:
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if key not in self.memo:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            self.memo[key] = digest.hexdigest()
        return self.memo[key]

    def files(self, paths: list[Path]) -> dict[str, str]:
        return {str(path): self.file(path) for path in paths}


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def reject_extra_runs(directory: Path, config: dict) -> None:
    stems = {f"{model}_seed{seed}" for model in config["models"] for seed in config["seeds"]}
    allowed = {stem + suffix for stem in stems for suffix in
               (".pt", ".joblib", ".json", "_gold_test_predictions.npz")}
    extras = [path.name for path in directory.glob("*")
              if path.suffix in {".pt", ".joblib", ".json", ".npz"} and path.name not in allowed]
    if extras:
        raise ValueError(f"Unexpected model artifacts in {directory}: {extras}. Use a separate output directory.")


def prepare_predictions(action: Action, output: Path) -> None:
    # The recovered regeneration command expects a sibling code directory. Its
    # own skip check does not establish checkpoint identity. A separate staging
    # directory preserves the training NPZs and forces real checkpoint inference.
    destination = output / "regenerated" / "model_runs"
    snapshot = destination.parent / "code"
    snapshot.mkdir(parents=True, exist_ok=True)
    for path in CODE.glob("*.py"):
        shutil.copy2(path, snapshot / path.name)
    for path in action.inputs:
        if path.parent == output / "model_runs":
            shutil.copy2(path, destination / path.name)
    for path in action.outputs:
        path.unlink(missing_ok=True)


def prepare_inference_benchmark(output: Path) -> None:
    # The recovered benchmark prepends a legacy ../final_run/code import path.
    # Supply that layout in a controlled snapshot so its worker subprocesses
    # import the same canonical source, independent of neighboring directories.
    snapshot = output / "benchmark_code" / "final_run" / "code"
    driver = output / "benchmark_code" / "driver"
    snapshot.mkdir(parents=True, exist_ok=True)
    driver.mkdir(parents=True, exist_ok=True)
    for path in CODE.glob("*.py"):
        shutil.copy2(path, snapshot / path.name)
    shutil.copy2(CODE / "benchmark_models.py", driver / "benchmark_models.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "paper.json")
    parser.add_argument("--input", type=Path, help="Source ZIP; required when clean is selected")
    parser.add_argument("--output-dir", type=Path, required=True, help="Dedicated generated-output directory")
    parser.add_argument("--stages", nargs="+", choices=STAGES, help="Selected stages, executed in pipeline order")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without checking data, importing ML libraries, or writing files")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Skip only actions with matching inputs, code, config and output hashes")
    mode.add_argument("--overwrite", action="store_true", help="Rerun selected actions and replace their generated outputs")
    args = parser.parse_args()
    config = read_config(args.config.expanduser().resolve())
    output = args.output_dir.expanduser().resolve()
    source = args.input.expanduser().resolve() if args.input else None
    stages = set(args.stages or config["stages"])
    if "clean" in stages and source is None:
        parser.error("--input is required for the clean stage")
    # The package's reference evidence is read-only input, never a run target.
    protected = [ROOT / name for name in ("src", "scripts", "configs", "docs", "tests", "reference_results")]
    protected.append(ROOT / "results" / "reference")
    if output == ROOT or output in ROOT.parents or any(output.is_relative_to(path) for path in protected):
        parser.error("--output-dir must be a dedicated generated directory outside source and reference results")
    actions = [action for action in build_actions(config, source, output) if action.stage in stages]
    print(f"Configuration: {config['name']}; paper_protocol={config['paper_protocol']}", flush=True)
    print(config["purpose"], flush=True)
    if args.dry_run:
        for action in actions:
            if action.stage == "predictions":
                print("[predictions] Stage copies of checkpoint/JSON files and source code; regenerate separate NPZs.")
            if action.stage == "inference-benchmark":
                print("[inference-benchmark] Stage a driver and source snapshot with the required import layout.")
            print(f"[{action.name}] {shlex.join(action.command)}")
        print(f"Dry run only: {len(actions)} commands; no data or execution was validated.")
        return 0
    fingerprints = Fingerprints()
    source_hashes = fingerprints.files([Path(__file__).resolve(), *sorted(CODE.glob("*.py"))])
    versions = {}
    for distribution in ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "joblib", "torch", "threadpoolctl"):
        try:
            versions[distribution] = package_metadata.version(distribution)
        except package_metadata.PackageNotFoundError:
            versions[distribution] = "not installed"
    state = output / "_runner"
    state.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(CODE) + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = str(config["threads"])
    run_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "paper_protocol": config["paper_protocol"],
        "expected_model_seed_pairs": [[model, seed] for model in config["models"] for seed in config["seeds"]],
        "selected_stages": [stage for stage in STAGES if stage in stages],
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(), "packages": versions,
        "source_hashes": source_hashes, "actions": [], "status": "running",
    }
    write_json(state / "invocation.json", run_manifest)
    for action in actions:
        if fingerprints.files([Path(__file__).resolve(), *sorted(CODE.glob("*.py"))]) != source_hashes:
            raise RuntimeError("Source files changed during this invocation. Restart with a new output directory "
                               "or rebuild from the earliest affected stage using --overwrite.")
        if action.stage in {"train", "predictions", "latent", "evidence", "inference-benchmark"}:
            reject_extra_runs(output / "model_runs", config)
            reject_extra_runs(output / "regenerated" / "model_runs", config)
        absent = [str(path) for path in action.inputs if not path.is_file()]
        if absent:
            raise FileNotFoundError(f"{action.name}: missing prerequisite files: {absent}. Run the preceding stages first.")
        specification = {"config": config, "source_hashes": source_hashes, "command": action.command,
                         "inputs": fingerprints.files(action.inputs), "python": sys.version, "packages": versions}
        signature = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
        marker = state / f"{action.name}.json"
        previous = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else None
        complete = (previous is not None and previous.get("signature") == signature
                    and all(path.is_file() for path in action.outputs)
                    and previous.get("outputs") == fingerprints.files(action.outputs))
        if args.resume and complete:
            print(f"SKIP verified {action.name}", flush=True)
            run_manifest["actions"].append({"name": action.name, "status": "verified_existing"})
            continue
        if not args.overwrite and (previous is not None or any(path.exists() for path in action.outputs)):
            raise RuntimeError(f"{action.name}: existing outputs are not eligible for this invocation. "
                               "Use --resume for matching completed work, a new output directory, or --overwrite "
                               "with every stage from the earliest changed input through evidence.")
        for path in action.outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
        if action.stage == "predictions":
            prepare_predictions(action, output)
        if action.stage == "inference-benchmark":
            prepare_inference_benchmark(output)
        marker.unlink(missing_ok=True)
        log = logs / f"{action.name}.log"
        print(f"RUN {action.name}; log={log}", flush=True)
        with log.open("w", encoding="utf-8") as handle:
            handle.write(shlex.join(action.command) + "\n")
            handle.flush()
            process = subprocess.run(action.command, cwd=ROOT, env=environment,
                                     stdout=handle, stderr=subprocess.STDOUT, check=False)
        if process.returncode:
            raise RuntimeError(f"{action.name} failed with exit code {process.returncode}; inspect {log}")
        absent = [str(path) for path in action.outputs if not path.is_file()]
        if absent:
            raise RuntimeError(f"{action.name} exited successfully but expected outputs are missing: {absent}")
        write_json(marker, {"signature": signature, "specification": specification,
                            "outputs": fingerprints.files(action.outputs),
                            "completed_utc": datetime.now(timezone.utc).isoformat()})
        run_manifest["actions"].append({"name": action.name, "status": "completed"})
        write_json(state / "invocation.json", run_manifest)
    run_manifest["status"] = "completed_selected_stages"
    write_json(state / "invocation.json", run_manifest)
    print("Completed selected stages. Protocol identity is recorded in _runner/invocation.json.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
