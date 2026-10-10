"""Unit tests for ImageStreamGenerator and deterministic stream reproducibility."""

import numpy as np
import pytest
from src.config import load_config
from src.stream import generate_image_stream, ImageStreamGenerator


def test_seeded_run_produces_identical_stream():
    """Verify that a seeded run produces the exact same stream twice (Task 2 acceptance criterion)."""
    config = load_config()
    seed = 12345
    duration_test = 2.0  # 2 hours for quick, comprehensive unit test

    stream1 = generate_image_stream(config, seed=seed, duration_hours=duration_test)
    stream2 = generate_image_stream(config, seed=seed, duration_hours=duration_test)

    assert len(stream1) > 0
    assert len(stream1) == len(stream2)

    for r1, r2 in zip(stream1, stream2):
        assert r1.image_id == r2.image_id
        assert r1.timestamp == r2.timestamp
        assert r1.tile_id == r2.tile_id
        assert r1.size_bytes == r2.size_bytes
        assert r1.ground_truth_label == r2.ground_truth_label
        assert r1.is_event_image == r2.is_event_image
        assert r1.event_type == r2.event_type
        assert r1.ground_truth_importance == r2.ground_truth_importance
        # Array equality check
        np.testing.assert_array_equal(r1.pixels, r2.pixels)


def test_event_injection_present():
    """Verify injected events are tagged as event images in the stream."""
    config = load_config()
    # Check over 24 hours to capture scenario events (e.g. fire at hour 4.5, flood at hour 10.0)
    stream = generate_image_stream(config, seed=42, duration_hours=12.0)

    event_images = [img for img in stream if img.is_event_image]
    assert len(event_images) > 0

    fire_images = [img for img in event_images if img.event_type == "fire_or_smoke"]
    assert len(fire_images) > 0
    # Verify fire event tile corresponds to TILE-04 as defined in config
    assert fire_images[0].tile_id == "TILE-04"
