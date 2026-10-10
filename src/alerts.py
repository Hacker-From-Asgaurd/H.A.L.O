"""Ground alert reprioritization and delivery management for HALO.

Implements Stage 5 & Section 4 ground alerts:
- Alerts are delivered strictly during ground passes (not upon creation).
- Active alerts apply boost A_i to matching images in the target region.
- Alert expires and ceases boosting after expiry_time.
- Older images in the alerted region are marked covered and penalized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from src.config import AlertItemConfig, HaloConfig, load_config
from src.utility import UtilityQueue, UtilityQueueItem


@dataclass
class GroundAlert:
    """A prioritized command uplinked to the satellite during a ground pass."""
    alert_id: str
    region_tile: str
    event_type: str
    boost_factor: float
    created_time_sec: float
    expiry_time_sec: float
    delivery_pass_index: int
    is_delivered: bool = False
    delivered_at_sec: Optional[float] = None

    def is_active(self, current_time_sec: float) -> bool:
        """Alert is active only after physical delivery and before expiry."""
        if not self.is_delivered or self.delivered_at_sec is None:
            return False
        return self.delivered_at_sec <= current_time_sec <= self.expiry_time_sec


class AlertManager:
    """Manages ground alerts schedule, delivery during passes, and utility boosts."""

    def __init__(self, config: Optional[HaloConfig] = None):
        """Initialize AlertManager.

        Args:
            config: HaloConfig instance.
        """
        self.config = config or load_config()
        self.alerts: List[GroundAlert] = []
        self._load_scenario_alerts()

    def _load_scenario_alerts(self) -> None:
        """Load configured scenario alerts from config."""
        for item in self.config.alerts.scenario_alerts:
            # Approximate pass time for delivery
            pass_start = item.delivery_pass_index * (self.config.ground_pass.pass_interval_min * 60.0)
            expiry_sec = pass_start + (item.duration_hours * 3600.0)
            self.alerts.append(
                GroundAlert(
                    alert_id=item.alert_id,
                    region_tile=item.region_tile,
                    event_type=item.event_type,
                    boost_factor=item.boost_factor,
                    created_time_sec=0.0,
                    expiry_time_sec=expiry_sec,
                    delivery_pass_index=item.delivery_pass_index,
                )
            )

    def add_alert(self, alert: GroundAlert) -> None:
        """Register a new ground alert."""
        self.alerts.append(alert)

    def on_pass_occurred(self, pass_index: int, pass_time_sec: float) -> List[GroundAlert]:
        """Deliver scheduled ground messages when satellite establishes contact during a pass.

        Args:
            pass_index: Index of current ground pass.
            pass_time_sec: Simulation time of pass contact.

        Returns:
            List of newly delivered GroundAlert objects.
        """
        newly_delivered = []
        for alert in self.alerts:
            if not alert.is_delivered and alert.delivery_pass_index <= pass_index:
                alert.is_delivered = True
                alert.delivered_at_sec = pass_time_sec
                newly_delivered.append(alert)
        return newly_delivered

    def get_active_alerts(self, current_time_sec: float) -> List[GroundAlert]:
        """Query currently active alerts at simulation timestamp.

        Args:
            current_time_sec: Current simulation time in seconds.

        Returns:
            List of currently active GroundAlert objects.
        """
        return [alert for alert in self.alerts if alert.is_active(current_time_sec)]

    def apply_to_queue(
        self,
        queue: UtilityQueue,
        current_time_sec: float,
        enable_alerts: bool = True,
    ) -> None:
        """Update alert boosts and apply covered-area penalties across queue items.

        Args:
            queue: Onboard UtilityQueue.
            current_time_sec: Current simulation timestamp.
            enable_alerts: Toggle for ablation studies (True by default).
        """
        if not enable_alerts:
            for item in queue.items:
                item.alert_boost = 1.0
            return

        active_alerts = self.get_active_alerts(current_time_sec)

        # Map active alerts by region and event type
        # Also track tiles with active alerts for the covered-area rule
        alerted_regions: Dict[str, GroundAlert] = {}
        for a in active_alerts:
            alerted_regions[a.region_tile] = a

        # Identify latest image per alerted tile to apply covered-area rule
        latest_timestamp_per_tile: Dict[str, float] = {}
        for item in queue.items:
            tile = item.image.tile_id
            if tile in alerted_regions:
                curr_ts = item.image.timestamp
                if tile not in latest_timestamp_per_tile or curr_ts > latest_timestamp_per_tile[tile]:
                    latest_timestamp_per_tile[tile] = curr_ts

        for item in queue.items:
            tile = item.image.tile_id
            if tile in alerted_regions:
                alert = alerted_regions[tile]
                label_match = (
                    alert.event_type in item.image.ground_truth_label
                    or (item.image.event_type and alert.event_type in item.image.event_type)
                )

                if label_match:
                    # Apply boost factor
                    item.alert_boost = alert.boost_factor

                    # Covered area rule: if this is an older image for the alerted tile, mark covered
                    if item.image.timestamp < latest_timestamp_per_tile.get(tile, item.image.timestamp):
                        item.is_covered = True
                        item.base_score = min(
                            item.base_score,
                            self.config.utility_model.covered_area_base_score,
                        )
                else:
                    item.alert_boost = 1.0
            else:
                # No active alert for this region
                item.alert_boost = 1.0
