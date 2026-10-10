"""Onboard image quality screening and cloud detection.

Implements Stage 2 of the onboard pipeline:
Rejects images that are mostly cloud, blurry, badly exposed, or blank.
Computes a normalized quality metric q_i in [0, 1] used in base score B_i = q_i * m_i * c_i.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from src.config import HaloConfig, QualityFilterConfig, load_config


@dataclass
class QualityResult:
    """Outcome of quality inspection."""
    passes: bool
    quality_score: float  # q_i in [0, 1]
    reason: str
    cloud_fraction: float
    blur_score: float
    mean_luminance: float
    pixel_variance: float


class QualityFilter:
    """Evaluates satellite images for cloud cover, blur, exposure defects, and blankness."""

    def __init__(self, config: Optional[QualityFilterConfig] = None):
        """Initialize quality filter with parameters from config.

        Args:
            config: QualityFilterConfig instance. If None, loads from root config.
        """
        if config is None:
            config = load_config().quality_filter
        self.config = config

    def check_quality(self, pixels: np.ndarray) -> QualityResult:
        """Inspect image pixels and determine if the image meets quality standards.

        Args:
            pixels: RGB image array of shape (H, W, 3) in uint8.

        Returns:
            QualityResult: Filter determination, metric scores, and rejection reason if any.
        """
        if pixels is None or pixels.size == 0:
            return QualityResult(
                passes=False,
                quality_score=0.0,
                reason="empty_image",
                cloud_fraction=1.0,
                blur_score=0.0,
                mean_luminance=0.0,
                pixel_variance=0.0,
            )

        # 1. Convert to grayscale and HSV for analysis
        gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV)
        pixel_variance = float(np.var(gray))
        mean_luminance = float(np.mean(gray))

        # 2. Blank image check (sensor dropout / flat field with near-zero variance)
        if pixel_variance < min(self.config.min_variance, 4.0):
            return QualityResult(
                passes=False,
                quality_score=0.05,
                reason="blank_image",
                cloud_fraction=0.0,
                blur_score=pixel_variance,
                mean_luminance=mean_luminance,
                pixel_variance=pixel_variance,
            )

        # 3. Exposure check (extreme brightness saturation or darkness)
        if mean_luminance < self.config.min_mean_luminance:
            return QualityResult(
                passes=False,
                quality_score=0.1,
                reason="underexposed",
                cloud_fraction=0.0,
                blur_score=0.0,
                mean_luminance=mean_luminance,
                pixel_variance=pixel_variance,
            )
        if mean_luminance > self.config.max_mean_luminance:
            return QualityResult(
                passes=False,
                quality_score=0.1,
                reason="overexposed",
                cloud_fraction=1.0,
                blur_score=0.0,
                mean_luminance=mean_luminance,
                pixel_variance=pixel_variance,
            )

        # 4. Cloud fraction (bright + low saturation)
        # Value (brightness) > 200 and Saturation < 50
        val = hsv[:, :, 2]
        sat = hsv[:, :, 1]
        cloud_mask = (val > 195) & (sat < 55)
        cloud_fraction = float(np.mean(cloud_mask))

        if cloud_fraction > self.config.max_cloud_fraction:
            return QualityResult(
                passes=False,
                quality_score=max(0.05, 1.0 - cloud_fraction),
                reason="excessive_clouds",
                cloud_fraction=cloud_fraction,
                blur_score=0.0,
                mean_luminance=mean_luminance,
                pixel_variance=pixel_variance,
            )

        # 5. Blur check (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = float(np.var(laplacian))

        if blur_score < self.config.min_blur_score:
            return QualityResult(
                passes=False,
                quality_score=0.2,
                reason="blurry_image",
                cloud_fraction=cloud_fraction,
                blur_score=blur_score,
                mean_luminance=mean_luminance,
                pixel_variance=pixel_variance,
            )

        # Passed all filters: compute continuous quality score q_i in [0.35, 1.0]
        # Sharper with lower cloud fraction yields higher quality
        cloud_factor = 1.0 - (cloud_fraction / self.config.max_cloud_fraction) * 0.4
        sharpness_factor = min(1.0, 0.6 + 0.4 * (blur_score / (self.config.min_blur_score * 3.0)))
        quality_score = float(np.clip(cloud_factor * sharpness_factor, 0.1, 1.0))

        return QualityResult(
            passes=True,
            quality_score=quality_score,
            reason="passed",
            cloud_fraction=cloud_fraction,
            blur_score=blur_score,
            mean_luminance=mean_luminance,
            pixel_variance=pixel_variance,
        )
