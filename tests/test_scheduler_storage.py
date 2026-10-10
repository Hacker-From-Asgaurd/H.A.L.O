"""Unit tests for Task 8: Greedy & Knapsack schedulers and Storage eviction policy."""

import numpy as np
import pytest
from src.config import load_config
from src.scheduler import greedy_schedule, knapsack_schedule
from src.storage import StorageManager
from src.stream import ImageRecord
from src.utility import UtilityQueue, UtilityQueueItem


def make_item(img_id: str, size_bytes: int, score: float) -> UtilityQueueItem:
    """Helper to construct dummy UtilityQueueItem."""
    img = ImageRecord(
        image_id=img_id,
        timestamp=0.0,
        tile_id="TILE-01",
        size_bytes=size_bytes,
        pixels=np.zeros((64, 64, 3), dtype=np.uint8),
        ground_truth_label="fire_or_smoke",
        ground_truth_importance=1.0,
    )
    item = UtilityQueueItem(image=img, base_score=score, decay_rate=0.0)
    item.current_score = score
    item.current_utility = score
    return item


def test_schedulers_respect_downlink_budget_in_all_conditions():
    """Verify both greedy and knapsack schedulers strictly obey byte budget D."""
    budget_d = 5_000_000  # 5 MB

    candidates = [
        make_item("IMG-1", size_bytes=2_000_000, score=0.8),
        make_item("IMG-2", size_bytes=3_000_000, score=0.9),
        make_item("IMG-3", size_bytes=1_500_000, score=0.7),
        make_item("IMG-4", size_bytes=2_500_000, score=0.6),
        make_item("IMG-5", size_bytes=4_000_000, score=0.95),
    ]

    # Test Greedy
    sel_greedy, bytes_greedy, score_greedy = greedy_schedule(candidates, budget_d)
    assert bytes_greedy <= budget_d
    assert sum(it.size_bytes for it in sel_greedy) == bytes_greedy
    assert bytes_greedy > 0

    # Test Knapsack
    sel_knapsack, bytes_knapsack, score_knapsack = knapsack_schedule(candidates, budget_d, discretization_kb=10)
    assert bytes_knapsack <= budget_d
    assert sum(it.size_bytes for it in sel_knapsack) == bytes_knapsack
    assert bytes_knapsack > 0


def test_storage_manager_evicts_sent_first_then_lowest_score():
    """Verify StorageManager evicts already-sent images first, then lowest score per byte."""
    config = load_config()
    # Configure tiny storage capacity of 10 MB for deterministic eviction testing
    config.satellite.storage_capacity_mb = 10.0
    storage_mgr = StorageManager(config)
    queue = UtilityQueue(config)

    # Add 3 items of 4 MB each (total 12 MB > 10 MB limit)
    # Item 1: High score per byte (0.8 / 4MB = 0.20/MB)
    it1 = make_item("IMG-1", size_bytes=4 * 1024 * 1024, score=0.8)
    # Item 2: Medium score per byte (0.4 / 4MB = 0.10/MB) - Marked already downlinked!
    it2 = make_item("IMG-2", size_bytes=4 * 1024 * 1024, score=0.4)
    storage_mgr.record_downlinked("IMG-2")
    # Item 3: Low score per byte (0.2 / 4MB = 0.05/MB) - Unsent
    it3 = make_item("IMG-3", size_bytes=4 * 1024 * 1024, score=0.2)

    queue.add(it1)
    queue.add(it2)
    queue.add(it3)

    assert queue.total_bytes == 12 * 1024 * 1024

    # Enforce limit: must evict IMG-2 first because it's already downlinked
    evicted = storage_mgr.enforce_storage_limit(queue)
    assert len(evicted) >= 1
    assert evicted[0].image.image_id == "IMG-2"
    assert queue.total_bytes <= storage_mgr.capacity_bytes

    # Now add another large item: Item 4 (4 MB, high score 0.9)
    it4 = make_item("IMG-4", size_bytes=4 * 1024 * 1024, score=0.9)
    evicted_2 = storage_mgr.enforce_storage_limit(queue, incoming_item=it4)

    # Next evicted must be Item 3 because it has the lowest score per byte among unsent
    assert len(evicted_2) >= 1
    assert evicted_2[0].image.image_id == "IMG-3"
