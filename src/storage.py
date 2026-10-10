"""Onboard satellite storage management and eviction policy.

Implements Stage 8 & Section 4 storage rules:
When storage is full, delete already-downlinked images first, then the lowest score per byte.
Ensures total onboard memory strictly respects storage_capacity_bytes at all times.
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

from src.config import HaloConfig, load_config
from src.utility import UtilityQueue, UtilityQueueItem


class StorageManager:
    """Oversees onboard storage buffer and enforces eviction when capacity is exceeded."""

    def __init__(self, config: Optional[HaloConfig] = None):
        """Initialize StorageManager.

        Args:
            config: HaloConfig instance.
        """
        self.config = config or load_config()
        self.capacity_bytes: int = self.config.satellite.storage_capacity_bytes
        # Set of image IDs that have already been downlinked and acknowledged by ground
        self.already_downlinked_ids: Set[str] = set()

    def record_downlinked(self, image_id: str) -> None:
        """Mark an image as downlinked and acknowledged by the ground station.

        Args:
            image_id: Unique image identifier.
        """
        self.already_downlinked_ids.add(image_id)

    def enforce_storage_limit(
        self,
        queue: UtilityQueue,
        incoming_item: Optional[UtilityQueueItem] = None,
    ) -> List[UtilityQueueItem]:
        """Evict items when storage capacity is breached.

        Policy:
        1. Delete already-downlinked images first.
        2. Delete un-downlinked images with lowest score per byte (S_i / size_i).

        Args:
            queue: Onboard UtilityQueue.
            incoming_item: Optional newly captured item to accommodate.

        Returns:
            List of evicted UtilityQueueItem objects.
        """
        required_headroom = incoming_item.size_bytes if incoming_item else 0
        current_bytes = queue.total_bytes
        evicted: List[UtilityQueueItem] = []

        if current_bytes + required_headroom <= self.capacity_bytes:
            return evicted

        # Step 1: Delete already-downlinked images first
        candidates_downlinked = [
            item for item in queue.items
            if item.image.image_id in self.already_downlinked_ids
        ]

        for item in candidates_downlinked:
            queue.remove(item)
            evicted.append(item)
            current_bytes -= item.size_bytes
            if current_bytes + required_headroom <= self.capacity_bytes:
                return evicted

        # Step 2: Delete un-downlinked images with lowest score per byte
        # Sort ascending by score per byte
        sorted_untransmitted = sorted(
            queue.items,
            key=lambda it: (it.score_per_byte, it.current_score),
            reverse=False,
        )

        for item in sorted_untransmitted:
            # If incoming item itself has worse score per byte than the lowest in queue,
            # drop the incoming item instead of dropping higher-value buffered data
            if incoming_item and incoming_item.score_per_byte <= item.score_per_byte:
                # The incoming item is not worthy of evicting existing queue items
                evicted.append(incoming_item)
                return evicted

            queue.remove(item)
            evicted.append(item)
            current_bytes -= item.size_bytes
            if current_bytes + required_headroom <= self.capacity_bytes:
                break

        return evicted
