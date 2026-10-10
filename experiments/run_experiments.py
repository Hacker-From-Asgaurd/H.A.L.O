"""Main experiment execution script for HALO benchmarks and ablations.

Runs all baseline policies and full HALO method across identical seeded streams (>= 10 seeds).
Computes all 7 evaluation metrics, computes mean +/- standard deviation, and outputs CSV reports.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.baselines import SimulationRunner
from src.config import HaloConfig, load_config
from src.metrics import EvaluationMetrics
from src.stream import generate_image_stream


METHODS_BENCHMARK = [
    ("fifo", "FIFO (Arrival Order)"),
    ("random", "Random Selection"),
    ("classifier_only", "Classifier-Only Priority"),
    ("priority_aging_no_expiry", "Priority + Aging (No Expiry)"),
    ("halo_greedy", "HALO Full (Greedy S/Byte)"),
    ("halo_knapsack", "HALO Full (0/1 Knapsack)"),
]

ABLATION_METHODS = [
    ("halo_greedy", "Full Method (HALO)"),
    ("ablation_no_alerts", "- No Alerts"),
    ("ablation_no_aging", "- No Aging Bonus"),
    ("ablation_no_expiry", "- No Delivery Expiry"),
    ("ablation_no_dedup", "- No Duplicate Filter"),
]


def run_benchmark_experiments(
    config: HaloConfig,
    seeds: List[int],
    results_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute all benchmark policies across fixed seeds and save CSV results.

    Args:
        config: HaloConfig instance.
        seeds: List of random seeds.
        results_dir: Directory where CSV files are saved.

    Returns:
        Tuple of (raw_results_df, summary_stats_df).
    """
    runner = SimulationRunner(config)
    raw_rows: List[Dict] = []
    storage_traces: List[Dict] = []

    print(f"\n=======================================================")
    print(f"Running HALO Benchmark Evaluation across {len(seeds)} Seeds")
    print(f"Seeds: {seeds}")
    print(f"=======================================================\n")

    for seed in seeds:
        print(f"-> Generating stream for Seed {seed}...")
        stream = generate_image_stream(config, seed=seed)

        for method_key, method_label in METHODS_BENCHMARK:
            metrics, downlinked, storage_history = runner.run(
                method=method_key,
                seed=seed,
                stream=stream,
            )
            row = metrics.to_dict()
            row["method_label"] = method_label
            raw_rows.append(row)

            # Save storage history for seed 42 to plot
            if seed == 42:
                for ts, s_bytes in storage_history:
                    storage_traces.append({
                        "method": method_key,
                        "method_label": method_label,
                        "timestamp_hours": ts / 3600.0,
                        "storage_used_mb": s_bytes / (1024 * 1024),
                    })

    raw_df = pd.DataFrame(raw_rows)
    raw_csv_path = results_dir / "halo_benchmark_results.csv"
    raw_df.to_csv(raw_csv_path, index=False)
    print(f"[Saved] Raw trial records: {raw_csv_path}")

    # Save storage time-series
    storage_df = pd.DataFrame(storage_traces)
    storage_csv_path = results_dir / "halo_storage_timeseries.csv"
    storage_df.to_csv(storage_csv_path, index=False)
    print(f"[Saved] Storage time-series: {storage_csv_path}")

    # Compute mean and standard deviation across seeds
    metric_cols = [
        "utility_per_pass",
        "urgent_delay_hours",
        "max_wait_hours",
        "stale_fraction",
        "mean_storage_mb",
        "bytes_wasted_mb",
    ]

    summary_rows = []
    for method_key, method_label in METHODS_BENCHMARK:
        sub = raw_df[raw_df["method"] == method_key]
        entry = {"method": method_key, "method_label": method_label, "num_runs": len(sub)}
        for col in metric_cols:
            mean_val = sub[col].mean()
            std_val = sub[col].std()
            entry[f"{col}_mean"] = round(mean_val, 4)
            entry[f"{col}_std"] = round(std_val, 4)
            entry[f"{col}_formatted"] = f"{mean_val:.2f} ± {std_val:.2f}"
        summary_rows.append(entry)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = results_dir / "halo_benchmark_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"[Saved] Benchmark Summary (Mean +/- Std): {summary_csv_path}")

    return raw_df, summary_df


def run_ablation_experiments(
    config: HaloConfig,
    seeds: List[int],
    results_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute ablation experiments and save ablation CSV reports.

    Args:
        config: HaloConfig instance.
        seeds: List of random seeds.
        results_dir: Output directory.

    Returns:
        Tuple of (raw_ablation_df, summary_ablation_df).
    """
    runner = SimulationRunner(config)
    raw_rows: List[Dict] = []

    print(f"\n=======================================================")
    print(f"Running HALO Ablation Studies across {len(seeds)} Seeds")
    print(f"=======================================================\n")

    for seed in seeds:
        stream = generate_image_stream(config, seed=seed)
        for method_key, method_label in ABLATION_METHODS:
            metrics, _, _ = runner.run(
                method=method_key,
                seed=seed,
                stream=stream,
            )
            row = metrics.to_dict()
            row["ablation_label"] = method_label
            raw_rows.append(row)

    raw_df = pd.DataFrame(raw_rows)
    raw_csv_path = results_dir / "halo_ablation_results.csv"
    raw_df.to_csv(raw_csv_path, index=False)
    print(f"[Saved] Raw ablation records: {raw_csv_path}")

    # Summary
    metric_cols = [
        "utility_per_pass",
        "urgent_delay_hours",
        "max_wait_hours",
        "stale_fraction",
        "mean_storage_mb",
        "bytes_wasted_mb",
    ]

    summary_rows = []
    for method_key, method_label in ABLATION_METHODS:
        sub = raw_df[raw_df["method"] == method_key]
        entry = {"method": method_key, "ablation_label": method_label, "num_runs": len(sub)}
        for col in metric_cols:
            mean_val = sub[col].mean()
            std_val = sub[col].std()
            entry[f"{col}_mean"] = round(mean_val, 4)
            entry[f"{col}_std"] = round(std_val, 4)
            entry[f"{col}_formatted"] = f"{mean_val:.2f} ± {std_val:.2f}"
        summary_rows.append(entry)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = results_dir / "halo_ablation_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"[Saved] Ablation Summary: {summary_csv_path}")

    return raw_df, summary_df


def main():
    """CLI entrypoint for running complete experiment suite."""
    parser = argparse.ArgumentParser(description="HALO Experiment Runner")
    parser.add_argument("--num-seeds", type=int, default=10, help="Number of seeds to run")
    parser.add_argument("--quick", action="store_true", help="Quick verification run with 2 seeds")
    args = parser.parse_args()

    config = load_config()
    results_dir = Path(config.experiments.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        seeds = config.experiments.seed_list[:2]
    else:
        seeds = config.experiments.seed_list[: args.num_seeds]

    raw_bench, summary_bench = run_benchmark_experiments(config, seeds, results_dir)
    raw_abl, summary_abl = run_ablation_experiments(config, seeds, results_dir)

    print("\n" + "=" * 95)
    print("HALO BENCHMARK EVALUATION SUMMARY (Mean +/- Std over 10 Seeds)")
    print("=" * 95)
    header = f"{'Method':<32} | {'Utility/Pass (High)':<19} | {'Urgent Delay (Low)':<18} | {'Stale Share (Low)':<16}"
    print(header)
    print("-" * 95)
    for _, row in summary_bench.iterrows():
        print(f"{row['method_label']:<32} | {row['utility_per_pass_formatted']:<19} | {row['urgent_delay_hours_formatted']:<18} | {row['stale_fraction_formatted']:<16}")
    print("=" * 95)


if __name__ == "__main__":
    main()
