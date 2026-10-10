"""Onboard duplicate detection and scene change computation.

Implements Stage 3 of the onboard pipeline:
Compares a newly captured image against the last kept image of the same geographic tile.
Drops near-duplicates where change c_i < min_change_threshold.
Supplies change score c_i in [0, 1] for base score formula B_i = q_i * m_i * c_i.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from src.config import DuplicateFilterConfig, load_config


@dataclass
class DuplicateCheckResult:
    """Outcome of duplicate comparison."""
    is_duplicate: bool
    change_score: float  # c_i in [0, 1]
    similarity: float  # Structural similarity or histogram correlation
    reason: str


class DuplicateFilter:
    """Tracks the last retained image per geographic tile and drops static near-duplicates."""

    def __init__(self, config: Optional[DuplicateFilterConfig] = None):
        """Initialize duplicate filter.

        Args:
            config: DuplicateFilterConfig. If None, loads from root config.
        """
        if config is None:
            config = load_config().duplicate_filter
        self.config = config
        # Stores the last accepted image pixels for each tile ID: {tile_id: pixels}
        self.last_kept_per_tile: Dict[str, np.ndarray] = {}

    def reset(self) -> None:
        """Clear cache of previously kept images across all tiles."""
        self.last_kept_per_tile.clear()

    def check_duplicate(self, tile_id: str, current_pixels: np.ndarray) -> DuplicateCheckResult:
        """Compare current image with last retained image for the specified tile.

        Args:
            tile_id: Geographic tile identifier.
            current_pixels: Current RGB image array (H, W, 3).

        Returns:
            DuplicateCheckResult: Whether dropped as duplicate, change score c_i, and similarity.
        """
        if tile_id not in self.last_kept_per_tile:
            # First observation of this tile: novel scene by definition
            return DuplicateCheckResult(
                is_duplicate=False,
                change_score=1.0,
                similarity=0.0,
                reason="first_tile_observation",
            )

        prev_pixels = self.last_kept_per_tile[tile_id]

        # Convert to grayscale for structural similarity calculation
        gray_curr = cv2.cvtColor(current_pixels, cv2.COLOR_RGB2GRAY)
        gray_prev = cv2.cvtColor(prev_pixels, cv2.COLOR_RGB2GRAY)

        # Vectorized structural difference for high-speed simulation throughput
        diff = cv2.absdiff(gray_curr, gray_prev)
        mae = float(np.mean(diff)) / 255.0
        # Scaled change metric: 0 for identical, ~0.15 for minor noise, 0.4+ for events/novelty
        raw_change = float(np.clip(mae * 4.0, 0.0, 1.0))
        sim_val = 1.0 - raw_change

        # If similarity exceeds threshold or change is lower than threshold, mark duplicate
        is_dup = (
            raw_change < self.config.min_change_threshold
            or sim_val > self.config.max_duplicate_similarity
        )

        if is_dup:
            change_c = max(0.02, raw_change)
            reason = "near_duplicate"
        else:
            excess = (raw_change - self.config.min_change_threshold) / max(0.01, (1.0 - self.config.min_change_threshold))
            change_c = float(np.clip(0.40 + 0.60 * excess, 0.40, 1.0))
            reason = "significant_change_detected"

        return DuplicateCheckResult(
            is_duplicate=is_dup,
            change_score=change_c,
            similarity=sim_val,
            reason=reason,
        )

    def record_kept(self, tile_id: str, pixels: np.ndarray) -> None:
        """Register that an image for this tile was accepted and stored onboard.

        Args:
            tile_id: Geographic tile identifier.
            pixels: RGB image array (H, W, 3).
        """
        self.last_kept_per_tile[tile_id] = pixels.copy()
