"""Unit tests for Task 6: Utility decay, delivery-time expiry, and bounded-wait aging."""

import numpy as np
import pytest
from src.config import load_config
from src.stream import ImageRecord
from src.utility import UtilityModel, UtilityQueue, UtilityQueueItem


@pytest.fixture
def dummy_record():
    """Create a dummy ImageRecord for utility testing."""
    return ImageRecord(
        image_id="TEST-001",
        timestamp=0.0,
        tile_id="TILE-01",
        size_bytes=1_000_000,
        pixels=np.zeros((64, 64, 3), dtype=np.uint8),
        ground_truth_label="fire_or_smoke",
        ground_truth_importance=1.0,
    )


def test_decay_formula_exact(dummy_record):
    """Verify U_i(t) = B_i * A_i * e^(-lambda * (t - t_i))."""
    model = UtilityModel()
    base_b = 0.8
    boost_a = 2.0
    lambda_val = 1.0  # 1.0 per hour
    age_hours = 2.0

    expected_u = 0.8 * 2.0 * np.exp(-1.0 * 2.0)  # 1.6 * exp(-2) ≈ 0.216536
    actual_u = model.compute_decayed_utility(base_b, boost_a, lambda_val, age_hours)
    assert abs(actual_u - expected_u) < 1e-5


def test_expired_image_is_deleted(dummy_record):
    """Verify an expired image whose value at t_next < theta is pruned from queue."""
    config = load_config()
    queue = UtilityQueue(config)

    # Fast-decaying event (fire lambda=1.0), low base score B=0.3
    # After 3 hours, U(3) = 0.3 * exp(-3.0) ≈ 0.0149 << theta (0.15)
    item = UtilityQueueItem(
        image=dummy_record,
        base_score=0.3,
        decay_rate=1.0,
        alert_boost=1.0,
    )
    queue.add(item)
    assert len(queue) == 1

    # Simulate expiry check before next pass at t_next = 3.5 hours = 12,600s
    t_curr = 3.0 * 3600.0
    t_next = 3.5 * 3600.0
    pruned = queue.prune_expired(current_time_sec=t_curr, next_pass_time_sec=t_next)

    # Image must be deleted from queue
    assert len(pruned) == 1
    assert pruned[0].image.image_id == "TEST-001"
    assert len(queue) == 0


def test_old_unexpired_image_gains_score(dummy_record):
    """Verify an unexpired image gains priority score due to aging bonus."""
    config = load_config()
    model = UtilityModel(config)

    # Slow-decaying event (crop lambda=0.005), high base score B=0.9
    # At t=0, utility = 0.9, aging bonus = 0
    # At t=2 hours, utility = 0.9 * exp(-0.005*2) = 0.891
    # Aging bonus = beta * min(2, 6) = 0.04 * 2 = 0.08
    # Score S(2) = 0.891 + 0.08 = 0.971 > S(0)
    u_initial = model.compute_decayed_utility(0.9, 1.0, 0.005, age_hours=0.0)
    score_initial = model.compute_aging_score(u_initial, wait_time_hours=0.0)

    u_later = model.compute_decayed_utility(0.9, 1.0, 0.005, age_hours=2.0)
    score_later = model.compute_aging_score(u_later, wait_time_hours=2.0)

    assert u_later >= config.utility_model.expiry_threshold
    assert score_later > score_initial


def test_aging_never_raises_expired_image():
    """Verify aging bonus is strictly zero when U_i(t) < theta (aging never revives stale data)."""
    config = load_config()
    model = UtilityModel(config)
    theta = config.utility_model.expiry_threshold

    # Utility below expiry threshold
    stale_u = theta - 0.05
    # Wait time is large (e.g. 5 hours)
    score_with_aging = model.compute_aging_score(stale_u, wait_time_hours=5.0)

    # Critical criterion: S_i(t) must equal stale_u, no bonus added!
    assert score_with_aging == stale_u
    assert score_with_aging < theta
