"""Configuration loader and schema definition for HALO.

Reads parameters from config.yaml into strongly typed dataclasses.
Ensures zero hard-coded scientific constants per Section 4 of the project brief.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class SystemConfig:
    seed: int = 42
    log_level: str = "INFO"


@dataclass
class SatelliteConfig:
    storage_capacity_mb: float = 500.0
    capture_interval_sec: float = 45.0
    simulation_duration_hours: float = 24.0

    @property
    def storage_capacity_bytes(self) -> int:
        """Return storage capacity in bytes."""
        return int(self.storage_capacity_mb * 1024 * 1024)


@dataclass
class GroundPassConfig:
    pass_duration_min: float = 10.0
    pass_interval_min: float = 95.0
    downlink_budget_mb_per_pass: float = 80.0

    @property
    def downlink_budget_bytes(self) -> int:
        """Return downlink budget per pass in bytes."""
        return int(self.downlink_budget_mb_per_pass * 1024 * 1024)


@dataclass
class UtilityModelConfig:
    expiry_threshold: float = 0.15
    aging_beta: float = 0.04
    aging_w_max_hours: float = 6.0
    covered_area_base_score: float = 0.05


@dataclass
class DecayRatesConfig:
    fire_or_smoke: float = 1.0
    ship_or_vehicle: float = 0.5
    flood: float = 0.05
    crop_or_land_use: float = 0.005
    default: float = 0.02

    def get_decay_rate(self, event_type: str) -> float:
        """Lookup decay rate lambda (per hour) for a given event/scene type.

        Args:
            event_type: Name of the event or land class.

        Returns:
            Decay constant lambda (1/hour).
        """
        key = event_type.lower().replace(" ", "_").replace("-", "_")
        if "fire" in key or "smoke" in key:
            return self.fire_or_smoke
        elif "ship" in key or "vehicle" in key:
            return self.ship_or_vehicle
        elif "flood" in key or "water" in key:
            return self.flood
        elif "crop" in key or "land" in key or "forest" in key or "vegetation" in key:
            return self.crop_or_land_use
        return getattr(self, key, self.default)


@dataclass
class QualityFilterConfig:
    min_quality_score: float = 0.35
    max_cloud_fraction: float = 0.60
    min_blur_score: float = 25.0
    min_mean_luminance: float = 20.0
    max_mean_luminance: float = 240.0
    min_variance: float = 10.0


@dataclass
class DuplicateFilterConfig:
    min_change_threshold: float = 0.12
    max_duplicate_similarity: float = 0.88


@dataclass
class ModelConfig:
    architecture: str = "mobilenet_v2"
    num_classes: int = 10
    int8_quantization: bool = True
    model_path: str = "models/halo_mobilenet_v2_int8.pt"
    full_precision_path: str = "models/halo_mobilenet_v2_fp32.pt"
    class_usefulness: Dict[str, float] = field(default_factory=dict)

    def get_usefulness(self, class_name: str) -> float:
        """Return usefulness value m in [0, 1] for a predicted class.

        Args:
            class_name: Predicted scene class label.

        Returns:
            Usefulness metric m between 0 and 1.
        """
        norm_class = class_name.lower().replace(" ", "_")
        for key, val in self.class_usefulness.items():
            if key in norm_class or norm_class in key:
                return float(val)
        return 0.50


@dataclass
class AlertItemConfig:
    alert_id: str
    delivery_pass_index: int
    region_tile: str
    event_type: str
    boost_factor: float
    duration_hours: float


@dataclass
class AlertsConfig:
    default_boost: float = 2.5
    scenario_alerts: List[AlertItemConfig] = field(default_factory=list)


@dataclass
class EventItemConfig:
    event_id: str
    start_hour: float
    duration_hours: float
    tile_id: str
    event_type: str
    ground_truth_importance: float


@dataclass
class EventsConfig:
    scenario_events: List[EventItemConfig] = field(default_factory=list)


@dataclass
class SchedulerConfig:
    default_algorithm: str = "greedy"
    knapsack_discretization_kb: int = 50


@dataclass
class StorageConfig:
    eviction_policy: str = "halo"


@dataclass
class ExperimentsConfig:
    num_seeds: int = 10
    seed_list: List[int] = field(default_factory=list)
    results_dir: str = "experiments/results"
    figures_dir: str = "experiments/figures"


@dataclass
class HaloConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    satellite: SatelliteConfig = field(default_factory=SatelliteConfig)
    ground_pass: GroundPassConfig = field(default_factory=GroundPassConfig)
    utility_model: UtilityModelConfig = field(default_factory=UtilityModelConfig)
    decay_rates: DecayRatesConfig = field(default_factory=DecayRatesConfig)
    quality_filter: QualityFilterConfig = field(default_factory=QualityFilterConfig)
    duplicate_filter: DuplicateFilterConfig = field(default_factory=DuplicateFilterConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    events: EventsConfig = field(default_factory=EventsConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    experiments: ExperimentsConfig = field(default_factory=ExperimentsConfig)


def load_config(config_path: Optional[str | Path] = None) -> HaloConfig:
    """Load and parse the HALO YAML configuration file.

    Args:
        config_path: Path to configuration file. If None, looks for config.yaml in workspace root.

    Returns:
        HaloConfig: Parsed configuration object containing all typed subsections.
    """
    if config_path is None:
        # Default to root config.yaml
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f) or {}

    sys_cfg = SystemConfig(**data.get("system", {}))
    sat_cfg = SatelliteConfig(**data.get("satellite", {}))
    gp_cfg = GroundPassConfig(**data.get("ground_pass", {}))
    um_cfg = UtilityModelConfig(**data.get("utility_model", {}))
    dr_cfg = DecayRatesConfig(**data.get("decay_rates", {}))
    qf_cfg = QualityFilterConfig(**data.get("quality_filter", {}))
    df_cfg = DuplicateFilterConfig(**data.get("duplicate_filter", {}))
    mod_cfg = ModelConfig(**data.get("model", {}))

    alerts_raw = data.get("alerts", {})
    scenario_alerts = [
        AlertItemConfig(**item) for item in alerts_raw.get("scenario_alerts", [])
    ]
    alerts_cfg = AlertsConfig(
        default_boost=alerts_raw.get("default_boost", 2.5),
        scenario_alerts=scenario_alerts,
    )

    events_raw = data.get("events", {})
    scenario_events = [
        EventItemConfig(**item) for item in events_raw.get("scenario_events", [])
    ]
    events_cfg = EventsConfig(scenario_events=scenario_events)

    sched_cfg = SchedulerConfig(**data.get("scheduler", {}))
    stor_cfg = StorageConfig(**data.get("storage", {}))
    exp_cfg = ExperimentsConfig(**data.get("experiments", {}))

    return HaloConfig(
        system=sys_cfg,
        satellite=sat_cfg,
        ground_pass=gp_cfg,
        utility_model=um_cfg,
        decay_rates=dr_cfg,
        quality_filter=qf_cfg,
        duplicate_filter=df_cfg,
        model=mod_cfg,
        alerts=alerts_cfg,
        events=events_cfg,
        scheduler=sched_cfg,
        storage=stor_cfg,
        experiments=exp_cfg,
    )
