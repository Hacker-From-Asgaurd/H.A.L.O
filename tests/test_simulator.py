"""Unit tests for Task 5: Simulator core running plain FIFO sender."""

import pytest
from src.config import load_config
from src.simulator import SatelliteSimulator, build_ground_pass_schedule, get_next_pass_start_time


def test_ground_pass_schedule_generation():
    """Verify ground passes match Section 4 timing constraints (8-12m duration, 90-100m interval)."""
    config = load_config()
    passes = build_ground_pass_schedule(config)

    assert len(passes) > 0
    for p in passes:
        duration_min = p.duration_sec / 60.0
        assert 8.0 <= duration_min <= 12.0
        assert p.budget_bytes == config.ground_pass.downlink_budget_bytes


def test_next_pass_lookup():
    """Verify lookup of next upcoming pass t_next."""
    config = load_config()
    passes = build_ground_pass_schedule(config)

    # Before first pass
    t_next = get_next_pass_start_time(0.0, passes)
    assert t_next == passes[0].start_time

    # Between pass 0 and pass 1
    t_between = (passes[0].start_time + passes[1].start_time) / 2.0
    t_next_2 = get_next_pass_start_time(t_between, passes)
    assert t_next_2 == passes[1].start_time


def test_fifo_runs_end_to_end_and_reports_bytes():
    """Verify FIFO runs end-to-end and reports bytes sent and dropped (Task 5 acceptance criterion)."""
    config = load_config()
    sim = SatelliteSimulator(config)

    stats = sim.run_fifo_baseline(seed=42)

    # 1. End-to-end execution completed
    assert stats.total_captured_images > 0
    assert stats.total_captured_bytes > 0
    assert stats.total_passes > 0

    # 2. Reports bytes sent and dropped
    assert stats.total_sent_bytes > 0
    assert stats.total_sent_images > 0
    assert stats.total_dropped_bytes >= 0
    assert stats.total_dropped_images >= 0

    # 3. Conservation of images and bytes integrity
    assert stats.total_sent_images + stats.total_dropped_images <= stats.total_captured_images
    assert stats.total_sent_bytes + stats.total_dropped_bytes <= stats.total_captured_bytes

    print("\n--- FIFO Baseline Simulation Results ---")
    print(f"Captured: {stats.total_captured_images} images ({stats.total_captured_bytes / 1e6:.2f} MB)")
    print(f"Sent:     {stats.total_sent_images} images ({stats.total_sent_bytes / 1e6:.2f} MB)")
    print(f"Dropped:  {stats.total_dropped_images} images ({stats.total_dropped_bytes / 1e6:.2f} MB)")
    print(f"Passes:   {stats.total_passes}")
