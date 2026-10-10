"""Unit tests for Task 9: Running all methods and validating CSV output."""

import os
from pathlib import Path
import pytest
from src.baselines import SimulationRunner
from src.config import load_config
from src.stream import generate_image_stream
from experiments.run_experiments import run_benchmark_experiments


def test_one_command_runs_all_methods_and_writes_csv(tmp_path):
    """Verify one command runs all methods on the same seeded stream and writes a results CSV (Task 9)."""
    config = load_config()
    seeds = [42, 101]

    raw_df, summary_df = run_benchmark_experiments(
        config=config,
        seeds=seeds,
        results_dir=tmp_path,
    )

    # 1. Output files exist
    results_csv = tmp_path / "halo_benchmark_results.csv"
    summary_csv = tmp_path / "halo_benchmark_summary.csv"
    assert results_csv.exists()
    assert summary_csv.exists()

    # 2. Contains all 6 benchmark policies
    expected_methods = {
        "fifo",
        "random",
        "classifier_only",
        "priority_aging_no_expiry",
        "halo_greedy",
        "halo_knapsack",
    }
    actual_methods = set(raw_df["method"].unique())
    assert expected_methods.issubset(actual_methods)

    # 3. All Section 8 metrics present in output
    for col in [
        "utility_per_pass",
        "urgent_delay_hours",
        "max_wait_hours",
        "stale_fraction",
        "mean_storage_mb",
        "bytes_wasted_mb",
    ]:
        assert col in raw_df.columns
        assert f"{col}_mean" in summary_df.columns
        assert f"{col}_std" in summary_df.columns

    # 4. Check HALO outperforms FIFO baseline on utility delivered
    fifo_util = summary_df.loc[summary_df["method"] == "fifo", "utility_per_pass_mean"].values[0]
    halo_util = summary_df.loc[summary_df["method"] == "halo_greedy", "utility_per_pass_mean"].values[0]
    assert halo_util > fifo_util
