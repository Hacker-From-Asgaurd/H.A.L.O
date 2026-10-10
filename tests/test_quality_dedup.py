"""Unit tests for Stage 2 (Quality filter) and Stage 3 (Duplicate check) on hand-picked images."""

import cv2
import numpy as np
import pytest
from src.config import load_config
from src.dedup import DuplicateFilter
from src.quality import QualityFilter


@pytest.fixture
def test_images():
    """Generate hand-picked good and flawed test images."""
    h, w = 64, 64
    rng = np.random.RandomState(42)

    # 1. Good clear image with high texture/contrast (e.g. urban/infrastructure grid)
    good = np.zeros((h, w, 3), dtype=np.uint8)
    good[::4, :, :] = 190
    good[:, ::4, :] = 140
    good += rng.randint(0, 30, (h, w, 3), dtype=np.uint8)

    # 2. Cloudy image (bright cumulus clouds: brightness ~215, low saturation, normal variance)
    cloudy = np.full((h, w, 3), 215, dtype=np.uint8)
    cloudy += rng.randint(0, 20, (h, w, 3), dtype=np.uint8)

    # 3. Blurry image (has wide regional landscape tones so variance > 20, but smoothed edges so Laplacian < 20)
    blurry_base = np.zeros((h, w, 3), dtype=np.uint8)
    blurry_base[:32, :, :] = 80
    blurry_base[32:, :, :] = 160
    blurry = cv2.GaussianBlur(blurry_base, (15, 15), 5.0)

    # 4. Underexposed image (near-black)
    underexposed = np.full((h, w, 3), 8, dtype=np.uint8)

    # 5. Overexposed image (near-white saturated)
    overexposed = np.full((h, w, 3), 250, dtype=np.uint8)

    # 6. Blank / defective sensor image (constant zero variance)
    blank = np.full((h, w, 3), 2, dtype=np.uint8)

    return {
        "good": good,
        "cloudy": cloudy,
        "blurry": blurry,
        "underexposed": underexposed,
        "overexposed": overexposed,
        "blank": blank,
    }


def test_quality_filter_accepts_good_image(test_images):
    """Verify QualityFilter accepts clear high-contrast satellite images."""
    q_filter = QualityFilter()
    res = q_filter.check_quality(test_images["good"])
    assert res.passes is True
    assert res.quality_score >= 0.35
    assert res.reason == "passed"


def test_quality_filter_rejects_cloudy_image(test_images):
    """Verify QualityFilter rejects cloud-covered scenes."""
    q_filter = QualityFilter()
    res = q_filter.check_quality(test_images["cloudy"])
    assert res.passes is False
    assert res.reason == "excessive_clouds"


def test_quality_filter_rejects_blurry_image(test_images):
    """Verify QualityFilter rejects out-of-focus or motion-blurred scenes."""
    q_filter = QualityFilter()
    res = q_filter.check_quality(test_images["blurry"])
    assert res.passes is False
    assert res.reason == "blurry_image"


def test_quality_filter_rejects_bad_exposure_and_blank(test_images):
    """Verify QualityFilter catches underexposed, overexposed, and blank sensor frames."""
    q_filter = QualityFilter()

    res_under = q_filter.check_quality(test_images["underexposed"])
    assert res_under.passes is False
    assert res_under.reason in ["underexposed", "blank_image"]

    res_over = q_filter.check_quality(test_images["overexposed"])
    assert res_over.passes is False
    assert res_over.reason in ["overexposed", "blank_image", "excessive_clouds"]

    res_blank = q_filter.check_quality(test_images["blank"])
    assert res_blank.passes is False
    assert res_blank.reason == "blank_image"


def test_duplicate_filter_drops_near_duplicates_and_keeps_changes(test_images):
    """Verify DuplicateFilter detects identical frames and admits changed scenes."""
    dedup = DuplicateFilter()
    tile = "TILE-01"

    good_img = test_images["good"]
    # 1. First capture of tile should always pass (novel area)
    res1 = dedup.check_duplicate(tile, good_img)
    assert res1.is_duplicate is False
    assert res1.change_score == 1.0
    dedup.record_kept(tile, good_img)

    # 2. Identical or near-identical image (tiny 1-pixel noise)
    near_dup = good_img.copy()
    near_dup[0, 0, 0] = (int(near_dup[0, 0, 0]) + 1) % 255
    res2 = dedup.check_duplicate(tile, near_dup)
    assert res2.is_duplicate is True
    assert res2.reason == "near_duplicate"

    # 3. New image with substantial change (e.g. fire plume or flood inundation)
    changed_img = good_img.copy()
    changed_img[20:50, 20:50, :] = 255  # Large bright altered region
    res3 = dedup.check_duplicate(tile, changed_img)
    assert res3.is_duplicate is False
    assert res3.change_score > 0.12
    assert res3.reason == "significant_change_detected"
