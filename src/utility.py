"""Utility model, delivery-time expiry evaluation, and bounded-wait aging for HALO.

Implements Section 4 equations:
- Base score: B_i = q_i * m_i * c_i
- Time-decaying utility: U_i(t) = B_i * A_i * e^(-lambda_k * (t - t_i))
- Expiry check at delivery time: U_i(t_next) < theta => delete image i
- Aging score: S_i(t) = U_i(t) + beta * min(w_i, w_max) * 1[U_i(t) >= theta]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.config import HaloConfig, UtilityModelConfig, load_config
from src.stream import ImageRecord


@dataclass
class UtilityQueueItem:
    """An image record managed inside the onboard utility queue."""
    image: ImageRecord
    base_score: float  # B_i in [0, 1]
    decay_rate: float  # lambda_k (1/hour)
    alert_boost: float = 1.0  # A_i
    is_covered: bool = False  # Marked covered by newer alert in region
    is_expired: bool = False
    current_utility: float = 0.0  # U_i(t)
    current_score: float = 0.0  # S_i(t)

    @property
    def size_bytes(self) -> int:
        """Size of image payload in bytes."""
        return self.image.size_bytes

    @property
    def score_per_byte(self) -> float:
        """Utility density S_i / size_i for knapsack/greedy scheduling."""
        if self.size_bytes <= 0:
            return 0.0
        return self.current_score / float(self.size_bytes)


class UtilityModel:
    """Computes time-decaying utility, aging bonuses, and delivery-time expiry."""

    def __init__(self, config: Optional[HaloConfig] = None):
        """Initialize utility model with parameters from config.

        Args:
            config: HaloConfig instance. If None, loads from root config.
        """
        self.config = config or load_config()
        self.um_cfg = self.config.utility_model
        self.decay_cfg = self.config.decay_rates

    def compute_decayed_utility(
        self,
        base_score: float,
        alert_boost: float,
        decay_rate: float,
        age_hours: float,
    ) -> float:
        """Compute U_i(t) = B_i * A_i * e^(-lambda_k * (t - t_i)).

        Args:
            base_score: B_i in [0, 1].
            alert_boost: A_i >= 1.0.
            decay_rate: lambda_k (1/hour).
            age_hours: Time elapsed since capture (t - t_i) in hours.

        Returns:
            Decayed utility U_i(t).
        """
        age = max(0.0, age_hours)
        decay_factor = math.exp(-decay_rate * age)
        return float(base_score * alert_boost * decay_factor)

    def is_expired_at_delivery(
        self,
        base_score: float,
        alert_boost: float,
        decay_rate: float,
        capture_time_sec: float,
        next_pass_time_sec: float,
    ) -> bool:
        """Evaluate expiry rule at delivery time: U_i(t_next) < theta.

        Args:
            base_score: B_i.
            alert_boost: A_i.
            decay_rate: lambda_k per hour.
            capture_time_sec: Image capture timestamp in seconds.
            next_pass_time_sec: Start timestamp t_next of next pass in seconds.

        Returns:
            bool: True if expected utility at t_next falls below theta (must delete).
        """
        expected_age_hours = max(0.0, (next_pass_time_sec - capture_time_sec) / 3600.0)
        expected_u = self.compute_decayed_utility(
            base_score=base_score,
            alert_boost=alert_boost,
            decay_rate=decay_rate,
            age_hours=expected_age_hours,
        )
        return expected_u < self.um_cfg.expiry_threshold

    def compute_aging_score(
        self,
        utility_u: float,
        wait_time_hours: float,
    ) -> float:
        """Compute aging score: S_i(t) = U_i(t) + beta * min(w_i, w_max) * 1[U_i(t) >= theta].

        Guarantees that aging never revives or raises an expired image.

        Args:
            utility_u: Current decayed utility U_i(t).
            wait_time_hours: Time waiting in queue w_i in hours.

        Returns:
            Priority score S_i(t).
        """
        theta = self.um_cfg.expiry_threshold
        # Indicator 1[U_i(t) >= theta]: bonus applies ONLY while above threshold
        if utility_u < theta:
            # Expired / stale: zero aging bonus
            return float(utility_u)

        w_capped = min(max(0.0, wait_time_hours), self.um_cfg.aging_w_max_hours)
        bonus = self.um_cfg.aging_beta * w_capped
        return float(utility_u + bonus)


class UtilityQueue:
    """Prioritized onboard image buffer with time decay, expiry pruning, and aging."""

    def __init__(self, config: Optional[HaloConfig] = None):
        """Initialize utility queue.

        Args:
            config: HaloConfig instance.
        """
        self.config = config or load_config()
        self.model = UtilityModel(self.config)
        self.items: List[UtilityQueueItem] = []

    def __len__(self) -> int:
        return len(self.items)

    @property
    def total_bytes(self) -> int:
        """Total memory consumed by items in queue."""
        return sum(item.size_bytes for item in self.items)

    def add(self, item: UtilityQueueItem) -> None:
        """Enqueue a new image item."""
        self.items.append(item)

    def remove(self, item: UtilityQueueItem) -> None:
        """Remove an item from the queue."""
        if item in self.items:
            self.items.remove(item)

    def update_scores(self, current_time_sec: float) -> None:
        """Refresh current utility U_i(t) and priority score S_i(t) across all images.

        Args:
            current_time_sec: Current simulation timestamp in seconds.
        """
        for item in self.items:
            age_hours = (current_time_sec - item.image.timestamp) / 3600.0
            u = self.model.compute_decayed_utility(
                base_score=item.base_score,
                alert_boost=item.alert_boost,
                decay_rate=item.decay_rate,
                age_hours=age_hours,
            )
            item.current_utility = u
            item.current_score = self.model.compute_aging_score(
                utility_u=u,
                wait_time_hours=age_hours,
            )

    def prune_expired(
        self,
        current_time_sec: float,
        next_pass_time_sec: float,
        enable_expiry: bool = True,
    ) -> List[UtilityQueueItem]:
        """Prune images whose projected value at next ground pass is below expiry threshold.

        Args:
            current_time_sec: Current simulation timestamp.
            next_pass_time_sec: Start time t_next of next ground pass.
            enable_expiry: Toggle for ablation studies (True by default).

        Returns:
            List of expired items deleted from queue.
        """
        if not enable_expiry:
            return []

        expired: List[UtilityQueueItem] = []
        retained: List[UtilityQueueItem] = []

        for item in self.items:
            is_stale = self.model.is_expired_at_delivery(
                base_score=item.base_score,
                alert_boost=item.alert_boost,
                decay_rate=item.decay_rate,
                capture_time_sec=item.image.timestamp,
                next_pass_time_sec=next_pass_time_sec,
            )
            if is_stale:
                item.is_expired = True
                expired.append(item)
            else:
                retained.append(item)

        self.items = retained
        return expired

    def get_items_sorted(self, by: str = "score_per_byte") -> List[UtilityQueueItem]:
        """Return items ordered by priority score per byte (descending).

        Args:
            by: Sorting key ('score_per_byte' or 'score').

        Returns:
            Sorted list of queue items.
        """
        if by == "score_per_byte":
            return sorted(self.items, key=lambda it: it.score_per_byte, reverse=True)
        elif by == "score":
            return sorted(self.items, key=lambda it: it.current_score, reverse=True)
        return list(self.items)
