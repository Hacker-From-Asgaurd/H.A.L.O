"""Unit tests for config loading and parameter schema validation."""

import pytest
from pathlib import Path
from src.config import load_config, HaloConfig


def test_config_loads_properly():
    """Verify config.yaml loads without errors and returns HaloConfig."""
    config = load_config()
    assert isinstance(config, HaloConfig)
    assert config.system.seed == 42


def test_section_4_parameters_present():
    """Verify all parameters specified in Section 4 are present and match specifications."""
    config = load_config()

    # 1. Satellite & Ground Pass specifications
    assert config.satellite.storage_capacity_mb == 500.0
    assert config.satellite.storage_capacity_bytes == 500 * 1024 * 1024
    assert config.satellite.capture_interval_sec == 45.0
    assert 8.0 <= config.ground_pass.pass_duration_min <= 12.0
    assert 90.0 <= config.ground_pass.pass_interval_min <= 100.0
    assert config.ground_pass.downlink_budget_bytes > 0

    # 2. Decay rates table (per hour)
    assert config.decay_rates.fire_or_smoke == 1.0
    assert config.decay_rates.ship_or_vehicle == 0.5
    assert config.decay_rates.flood == 0.05
    assert config.decay_rates.crop_or_land_use == 0.005

    # Test lookup helper
    assert config.decay_rates.get_decay_rate("fire") == 1.0
    assert config.decay_rates.get_decay_rate("ship") == 0.5
    assert config.decay_rates.get_decay_rate("flood") == 0.05
    assert config.decay_rates.get_decay_rate("crops") == 0.005

    # 3. Utility model parameters
    assert config.utility_model.expiry_threshold == 0.15
    assert config.utility_model.aging_beta == 0.04
    assert config.utility_model.aging_w_max_hours == 6.0
    assert config.utility_model.covered_area_base_score == 0.05

    # 4. Filters & Schedulers
    assert config.quality_filter.min_quality_score > 0
    assert config.duplicate_filter.min_change_threshold > 0
    assert config.scheduler.default_algorithm in ["greedy", "knapsack"]
    assert config.storage.eviction_policy == "halo"

    # 5. Alerts and Events
    assert len(config.alerts.scenario_alerts) >= 2
    assert len(config.events.scenario_events) >= 3
