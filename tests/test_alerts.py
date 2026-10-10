"""Unit tests for Task 7: Ground alert delivery, boost timing, and expiry."""

import numpy as np
import pytest
from src.alerts import AlertManager, GroundAlert
from src.config import load_config
from src.stream import ImageRecord
from src.utility import UtilityQueue, UtilityQueueItem


@pytest.fixture
def sample_queue():
    """Create a test utility queue containing matching and non-matching images."""
    config = load_config()
    queue = UtilityQueue(config)

    # Image 1: In TILE-04, event_type fire_or_smoke
    img1 = ImageRecord(
        image_id="IMG-001",
        timestamp=3600.0,
        tile_id="TILE-04",
        size_bytes=1_000_000,
        pixels=np.zeros((64, 64, 3), dtype=np.uint8),
        ground_truth_label="fire_or_smoke",
        ground_truth_importance=1.0,
        event_type="fire_or_smoke",
    )
    item1 = UtilityQueueItem(image=img1, base_score=0.8, decay_rate=1.0)
    queue.add(item1)

    # Image 2: In TILE-09, forest (different region)
    img2 = ImageRecord(
        image_id="IMG-002",
        timestamp=3650.0,
        tile_id="TILE-09",
        size_bytes=1_000_000,
        pixels=np.zeros((64, 64, 3), dtype=np.uint8),
        ground_truth_label="forest",
        ground_truth_importance=0.45,
    )
    item2 = UtilityQueueItem(image=img2, base_score=0.5, decay_rate=0.01)
    queue.add(item2)

    return queue


def test_alert_lifecycle_and_timing(sample_queue):
    """Verify an alert raises matching images only after the next pass and stops after expiry."""
    config = load_config()
    manager = AlertManager(config)
    # Clear scenario alerts for isolated timing test
    manager.alerts.clear()

    # Create alert for TILE-04 scheduled for delivery in Pass #2
    pass_interval_sec = 95.0 * 60.0  # 5,700s
    pass_2_time = 2 * pass_interval_sec  # 11,400s
    expiry_sec = pass_2_time + (2.0 * 3600.0)  # expires 2 hours after delivery (18,600s)

    alert = GroundAlert(
        alert_id="TEST-ALT",
        region_tile="TILE-04",
        event_type="fire_or_smoke",
        boost_factor=3.0,
        created_time_sec=1000.0,
        expiry_time_sec=expiry_sec,
        delivery_pass_index=2,
    )
    manager.add_alert(alert)

    # 1. Before pass #2 (e.g. t = 8,000s, pass index 1): Alert created but NOT delivered yet!
    manager.on_pass_occurred(pass_index=1, pass_time_sec=pass_interval_sec)
    manager.apply_to_queue(sample_queue, current_time_sec=8000.0)
    sample_queue.update_scores(current_time_sec=8000.0)

    item1 = sample_queue.items[0]
    # Alert boost must remain 1.0 (unboosted) before delivery pass
    assert item1.alert_boost == 1.0

    # 2. During pass #2 (t = 11,400s): Contact established, alert delivered!
    manager.on_pass_occurred(pass_index=2, pass_time_sec=pass_2_time)
    assert alert.is_delivered is True

    # Check right after delivery: Alert must raise matching images
    t_after_delivery = pass_2_time + 60.0
    manager.apply_to_queue(sample_queue, current_time_sec=t_after_delivery)
    sample_queue.update_scores(current_time_sec=t_after_delivery)

    assert item1.alert_boost == 3.0
    # Non-matching image in TILE-09 should NOT be boosted
    assert sample_queue.items[1].alert_boost == 1.0

    # 3. After alert expiry (t = 19,000s > expiry_sec 18,600s): Alert expires and stops boosting
    t_post_expiry = expiry_sec + 400.0
    manager.apply_to_queue(sample_queue, current_time_sec=t_post_expiry)
    sample_queue.update_scores(current_time_sec=t_post_expiry)

    # Must revert back to 1.0
    assert item1.alert_boost == 1.0


def test_covered_area_rule():
    """Verify older images in alerted region are penalized under covered-area rule."""
    config = load_config()
    queue = UtilityQueue(config)
    manager = AlertManager(config)
    manager.alerts.clear()

    alert = GroundAlert(
        alert_id="ALT-COV",
        region_tile="TILE-04",
        event_type="fire_or_smoke",
        boost_factor=2.5,
        created_time_sec=0.0,
        expiry_time_sec=20000.0,
        delivery_pass_index=1,
        is_delivered=True,
        delivered_at_sec=1000.0,
    )
    manager.add_alert(alert)

    # Add older image for TILE-04
    img_old = ImageRecord(
        image_id="OLD-01",
        timestamp=2000.0,
        tile_id="TILE-04",
        size_bytes=1000,
        pixels=np.zeros((64, 64, 3), dtype=np.uint8),
        ground_truth_label="fire_or_smoke",
        ground_truth_importance=1.0,
    )
    item_old = UtilityQueueItem(image=img_old, base_score=0.8, decay_rate=1.0)
    queue.add(item_old)

    # Add newer image for TILE-04
    img_new = ImageRecord(
        image_id="NEW-02",
        timestamp=4000.0,
        tile_id="TILE-04",
        size_bytes=1000,
        pixels=np.zeros((64, 64, 3), dtype=np.uint8),
        ground_truth_label="fire_or_smoke",
        ground_truth_importance=1.0,
    )
    item_new = UtilityQueueItem(image=img_new, base_score=0.8, decay_rate=1.0)
    queue.add(item_new)

    manager.apply_to_queue(queue, current_time_sec=5000.0)

    # Older image marked covered and base score penalized to covered_area_base_score (0.05)
    assert item_old.is_covered is True
    assert item_old.base_score == config.utility_model.covered_area_base_score
    # Newer image retains original base score
    assert item_new.is_covered is False
    assert item_new.base_score == 0.8
