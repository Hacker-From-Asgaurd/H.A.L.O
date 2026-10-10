"""Image streaming and event injection pipeline for HALO.

Generates reproducible timestamped image streams from Earth observation patches,
assigns tile IDs, simulation timestamps, and file sizes, and injects dynamic environmental events.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from src.config import HaloConfig, load_config


@dataclass
class ImageRecord:
    """Represents a single captured satellite image record in the onboard pipeline."""
    image_id: str
    timestamp: float  # Capture time in seconds since mission start
    tile_id: str  # Geographic tile / region identifier
    size_bytes: int  # Image payload size in bytes
    pixels: np.ndarray  # Image pixel data (H, W, 3) uint8
    ground_truth_label: str  # True land cover or event class
    ground_truth_importance: float  # Ground-truth value for evaluation
    is_event_image: bool = False  # Whether image coincides with an active event
    event_type: Optional[str] = None  # Event type name if an event image
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def timestamp_hours(self) -> float:
        """Capture time in hours."""
        return self.timestamp / 3600.0


def create_synthetic_patch(
    label: str,
    size: Tuple[int, int] = (64, 64),
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Generate a realistic synthetic Earth observation patch with spectral texture.

    Args:
        label: Scene category label.
        size: Spatial resolution (H, W).
        rng: Random state for reproducible synthesis.

    Returns:
        np.ndarray: Synthetic RGB image array of shape (H, W, 3) in uint8.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    h, w = size
    # Baseline land cover spectral color profiles (R, G, B)
    palette = {
        "annual_crop": (160, 190, 80),
        "forest": (34, 110, 45),
        "highway": (130, 130, 135),
        "industrial": (150, 140, 150),
        "residential": (180, 130, 120),
        "river": (40, 90, 170),
        "sea_lake": (25, 60, 145),
        "fire_or_smoke": (230, 70, 20),
        "flood": (50, 100, 160),
        "ship_or_vehicle": (120, 140, 160),
        "cloudy": (245, 245, 250),
        "blank": (15, 15, 15),
    }

    base_color = palette.get(label.lower(), (100, 130, 90))
    # Add spatial gradient / texture noise
    texture = rng.normal(loc=0.0, scale=18.0, size=(h, w, 3))
    patch = np.zeros((h, w, 3), dtype=np.float32)

    for c in range(3):
        patch[:, :, c] = base_color[c] + texture[:, :, c]

    # Add scene-specific features
    if "fire" in label.lower():
        # Add high-intensity hot spots and smoke gradient
        smoke = rng.uniform(80, 180, size=(h, w))
        patch[:, :, 0] = np.clip(patch[:, :, 0] + rng.exponential(40, (h, w)), 0, 255)
        patch[:, :, 1] = np.clip(patch[:, :, 1] * 0.7 + smoke * 0.3, 0, 255)
    elif "cloud" in label.lower():
        patch = np.clip(patch + rng.uniform(50, 100, (h, w, 3)), 0, 255)
    elif "blank" in label.lower():
        patch = np.full((h, w, 3), fill_value=5.0, dtype=np.float32)

    patch = np.clip(patch, 0, 255).astype(np.uint8)
    return patch


class ImageStreamGenerator:
    """Generates a reproducible discrete-event image stream for satellite simulation."""

    def __init__(self, config: HaloConfig, seed: Optional[int] = None):
        """Initialize the stream generator.

        Args:
            config: HaloConfig instance.
            seed: Random seed for deterministic generation. Defaults to config.system.seed.
        """
        self.config = config
        self.seed = seed if seed is not None else config.system.seed
        self.rng = np.random.RandomState(self.seed)

        # Tiles grid (e.g. 16 grid cells standing for regions)
        self.tiles = [f"TILE-{i:02d}" for i in range(16)]

        # Land cover categories
        self.background_classes = [
            "annual_crop",
            "forest",
            "highway",
            "industrial",
            "residential",
            "sea_lake",
        ]

    def _get_active_events(self, current_hour: float) -> List[Any]:
        """Check which scenario events are active at the given simulation hour.

        Args:
            current_hour: Time in hours from simulation start.

        Returns:
            List of active EventItemConfig instances.
        """
        active = []
        for evt in self.config.events.scenario_events:
            if evt.start_hour <= current_hour <= (evt.start_hour + evt.duration_hours):
                active.append(evt)
        return active

    def generate_stream(
        self,
        duration_hours: Optional[float] = None,
        capture_interval_sec: Optional[float] = None,
    ) -> List[ImageRecord]:
        """Generate a deterministic sequence of ImageRecords over the simulation duration.

        Args:
            duration_hours: Simulation duration. If None, uses config value.
            capture_interval_sec: Time between captures. If None, uses config value.

        Returns:
            List of ImageRecord objects in chronological capture order.
        """
        # Reset RNG with seed to ensure complete repeatability
        rng = np.random.RandomState(self.seed)

        duration = (
            duration_hours
            if duration_hours is not None
            else self.config.satellite.simulation_duration_hours
        )
        interval = (
            capture_interval_sec
            if capture_interval_sec is not None
            else self.config.satellite.capture_interval_sec
        )

        total_seconds = duration * 3600.0
        current_time_sec = 0.0
        image_index = 0
        stream: List[ImageRecord] = []

        # Previous image patch cache per tile for temporal continuity
        tile_base_patches: Dict[str, np.ndarray] = {}
        tile_classes: Dict[str, str] = {}

        for tile in self.tiles:
            bg_cls = self.background_classes[rng.choice(len(self.background_classes))]
            tile_classes[tile] = bg_cls
            tile_base_patches[tile] = create_synthetic_patch(bg_cls, rng=rng)

        while current_time_sec < total_seconds:
            current_hour = current_time_sec / 3600.0
            image_index += 1
            image_id = f"IMG-{image_index:05d}"

            # Orbit sweep selects tile under nadir track
            tile_idx = (image_index % len(self.tiles))
            tile_id = self.tiles[tile_idx]

            # Check if an event is currently active in this tile
            active_events = [
                evt for evt in self._get_active_events(current_hour)
                if evt.tile_id == tile_id
            ]

            is_event = len(active_events) > 0
            event_type = active_events[0].event_type if is_event else None
            gt_importance = (
                active_events[0].ground_truth_importance
                if is_event
                else self.config.model.get_usefulness(tile_classes[tile_id])
            )

            # Determine image class & synthesize pixels
            if is_event and event_type:
                cls_label = event_type
                pixels = create_synthetic_patch(cls_label, rng=rng)
            else:
                # Atmospheric and environmental variation across orbital revisits
                chance = rng.uniform(0.0, 1.0)
                if chance < 0.10:
                    cls_label = "cloudy"
                    pixels = create_synthetic_patch("cloudy", rng=rng)
                elif chance < 0.12:
                    cls_label = "blank"
                    pixels = create_synthetic_patch("blank", rng=rng)
                else:
                    cls_label = tile_classes[tile_id]
                    # Realistic orbital revisit dynamics: illumination angle + surface activity
                    base_patch = tile_base_patches[tile_id]
                    orbit_cycle = np.sin((current_hour / 1.6) * np.pi) * 12.0  # diurnally varying sun angle
                    # Random scene activity / cloud shadow / atmospheric turbulence
                    activity_noise = rng.normal(orbit_cycle, 9.0, base_patch.shape).astype(np.float32)
                    pixels = np.clip(base_patch.astype(np.float32) + activity_noise, 0, 255).astype(np.uint8)

            # Image size in bytes (e.g. realistic Sentinel-2 patch compressed: ~600KB - 1.8MB)
            # Section 4 spec allows variable image sizes
            base_size = int(rng.uniform(700_000, 1_400_000))
            if cls_label == "blank":
                size_bytes = int(rng.uniform(150_000, 300_000))  # Blank images compress smaller
            elif cls_label == "cloudy":
                size_bytes = int(rng.uniform(400_000, 800_000))
            else:
                size_bytes = base_size

            record = ImageRecord(
                image_id=image_id,
                timestamp=current_time_sec,
                tile_id=tile_id,
                size_bytes=size_bytes,
                pixels=pixels,
                ground_truth_label=cls_label,
                ground_truth_importance=gt_importance,
                is_event_image=is_event,
                event_type=event_type,
                metadata={"hour": current_hour, "tile_index": tile_idx},
            )
            stream.append(record)

            # Advance simulation time
            current_time_sec += interval

        return stream


def generate_image_stream(
    config: Optional[HaloConfig] = None,
    seed: Optional[int] = None,
    duration_hours: Optional[float] = None,
) -> List[ImageRecord]:
    """Convenience functional interface to produce an image stream.

    Args:
        config: HaloConfig instance. If None, loads root config.yaml.
        seed: Random seed.
        duration_hours: Optional simulation duration override.

    Returns:
        List of ImageRecord objects.
    """
    if config is None:
        config = load_config()
    generator = ImageStreamGenerator(config=config, seed=seed)
    return generator.generate_stream(duration_hours=duration_hours)
