"""Discrete-event satellite simulator for HALO.

Simulates onboard image capture, storage buffering, periodic ground passes with byte budgets,
and downlinks images according to configurable queue and scheduling policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from src.config import HaloConfig, load_config
from src.stream import ImageRecord, generate_image_stream


@dataclass
class GroundPass:
    """Represents a scheduled satellite-ground station communication pass."""
    pass_index: int
    start_time: float  # seconds
    end_time: float  # seconds
    budget_bytes: int  # maximum bytes allowed to downlink
    bytes_downlinked: int = 0
    images_downlinked: List[ImageRecord] = field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        """Pass contact duration in seconds."""
        return self.end_time - self.start_time

    @property
    def remaining_budget_bytes(self) -> int:
        """Unused downlink bandwidth in bytes."""
        return max(0, self.budget_bytes - self.bytes_downlinked)


@dataclass
class SimulationStats:
    """Summary statistics produced at the end of a simulation run."""
    total_captured_images: int
    total_captured_bytes: int
    total_sent_images: int
    total_sent_bytes: int
    total_dropped_images: int
    total_dropped_bytes: int
    total_passes: int
    storage_history_timestamps: List[float] = field(default_factory=list)
    storage_history_bytes: List[int] = field(default_factory=list)
    storage_capacity_bytes: int = 0
    records_sent: List[ImageRecord] = field(default_factory=list)
    records_dropped: List[ImageRecord] = field(default_factory=list)


def build_ground_pass_schedule(config: HaloConfig) -> List[GroundPass]:
    """Precompute all ground pass contact windows across the mission duration.

    Args:
        config: HaloConfig configuration.

    Returns:
        List of GroundPass objects sorted chronologically.
    """
    total_sec = config.satellite.simulation_duration_hours * 3600.0
    pass_duration_sec = config.ground_pass.pass_duration_min * 60.0
    pass_interval_sec = config.ground_pass.pass_interval_min * 60.0
    budget_bytes = config.ground_pass.downlink_budget_bytes

    passes = []
    current_start = pass_interval_sec  # First pass occurs after one orbit interval
    pass_idx = 0

    while current_start < total_sec:
        pass_idx += 1
        current_end = min(current_start + pass_duration_sec, total_sec)
        passes.append(
            GroundPass(
                pass_index=pass_idx,
                start_time=current_start,
                end_time=current_end,
                budget_bytes=budget_bytes,
            )
        )
        current_start += pass_interval_sec

    return passes


def get_next_pass_start_time(current_time: float, passes: List[GroundPass]) -> float:
    """Find start timestamp of the next upcoming ground pass relative to current_time.

    Args:
        current_time: Current simulation time in seconds.
        passes: List of scheduled GroundPass windows.

    Returns:
        Timestamp t_next (in seconds) of the start of the next pass.
    """
    for p in passes:
        if p.start_time > current_time:
            return p.start_time
    # If no upcoming pass remains, return end of current or future horizon
    return current_time + 3600.0


class SatelliteSimulator:
    """Core discrete-event simulator running onboard satellite storage and pass downlinks."""

    def __init__(self, config: Optional[HaloConfig] = None):
        """Initialize satellite simulator.

        Args:
            config: HaloConfig configuration. If None, loaded from root config.
        """
        self.config = config or load_config()

    def run_fifo_baseline(
        self,
        stream: Optional[List[ImageRecord]] = None,
        seed: Optional[int] = None,
    ) -> SimulationStats:
        """Run an end-to-end mission simulation using a First-In-First-Out (FIFO) transmission policy.

        Args:
            stream: Pre-generated ImageRecord stream. If None, generated using seed.
            seed: Random seed.

        Returns:
            SimulationStats: Full run summary including bytes sent, dropped, and history.
        """
        if stream is None:
            stream = generate_image_stream(self.config, seed=seed)

        passes = build_ground_pass_schedule(self.config)
        capacity_bytes = self.config.satellite.storage_capacity_bytes

        # Internal FIFO storage buffer: list of ImageRecord
        buffer: List[ImageRecord] = []
        current_storage_bytes = 0

        sent_records: List[ImageRecord] = []
        dropped_records: List[ImageRecord] = []

        storage_timestamps: List[float] = []
        storage_values: List[int] = []

        total_captured_bytes = sum(img.size_bytes for img in stream)

        # Merge capture events and pass events into chronologically ordered timeline
        pass_by_index = {p.pass_index: p for p in passes}
        pass_starts = {p.start_time: p for p in passes}

        # Event queue: capture events and pass start events
        timeline_events: List[Tuple[float, str, Any]] = []
        for img in stream:
            timeline_events.append((img.timestamp, "capture", img))
        for p in passes:
            timeline_events.append((p.start_time, "pass_start", p))

        # Sort events by simulation timestamp
        timeline_events.sort(key=lambda x: x[0])

        for event_time, event_type, event_obj in timeline_events:
            storage_timestamps.append(event_time)
            storage_values.append(current_storage_bytes)

            if event_type == "capture":
                img: ImageRecord = event_obj
                # Check if image fits in satellite storage
                if current_storage_bytes + img.size_bytes > capacity_bytes:
                    # Buffer full: drop image under plain FIFO policy
                    dropped_records.append(img)
                else:
                    buffer.append(img)
                    current_storage_bytes += img.size_bytes

            elif event_type == "pass_start":
                ground_pass: GroundPass = event_obj
                # FIFO Downlink: transmit images in order of arrival until budget depleted
                idx = 0
                while idx < len(buffer) and ground_pass.remaining_budget_bytes > 0:
                    candidate = buffer[idx]
                    if candidate.size_bytes <= ground_pass.remaining_budget_bytes:
                        # Downlink candidate
                        ground_pass.bytes_downlinked += candidate.size_bytes
                        ground_pass.images_downlinked.append(candidate)
                        sent_records.append(candidate)
                        current_storage_bytes -= candidate.size_bytes
                        # Remove from onboard buffer
                        buffer.pop(idx)
                    else:
                        # Cannot fit in remaining budget of this pass; leave for next pass
                        idx += 1

            storage_timestamps.append(event_time)
            storage_values.append(current_storage_bytes)

        total_sent_bytes = sum(img.size_bytes for img in sent_records)
        total_dropped_bytes = sum(img.size_bytes for img in dropped_records)

        return SimulationStats(
            total_captured_images=len(stream),
            total_captured_bytes=total_captured_bytes,
            total_sent_images=len(sent_records),
            total_sent_bytes=total_sent_bytes,
            total_dropped_images=len(dropped_records),
            total_dropped_bytes=total_dropped_bytes,
            total_passes=len(passes),
            storage_history_timestamps=storage_timestamps,
            storage_history_bytes=storage_values,
            storage_capacity_bytes=capacity_bytes,
            records_sent=sent_records,
            records_dropped=dropped_records,
        )
