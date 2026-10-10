"""Simulation runners for baseline policies, ablations, and full HALO method.

Implements all methods compared in Section 6 & 9:
- FIFO
- Random
- Classifier-only priority
- Priority plus alerts plus aging, without expiry
- HALO (Greedy)
- HALO (Knapsack)
- Ablation configurations (no alerts, no aging, no expiry, no duplicate filter)
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.alerts import AlertManager, GroundAlert
from src.config import HaloConfig, load_config
from src.dedup import DuplicateFilter
from src.metrics import DownlinkedItemRecord, EvaluationMetrics, calculate_metrics
from src.quality import QualityFilter
from src.scheduler import greedy_schedule, knapsack_schedule
from src.scorer import HALOScorer, compute_base_score
from src.simulator import GroundPass, build_ground_pass_schedule, get_next_pass_start_time
from src.storage import StorageManager
from src.stream import ImageRecord, generate_image_stream
from src.utility import UtilityModel, UtilityQueue, UtilityQueueItem


class GroundStationEvaluator:
    """Evaluates the true ground utility delivered by any satellite transmission policy.

    Guarantees objective, symmetric evaluation across all tested methods:
    U_delivered = q_ground * m_ground * c_ground * A_ground * e^(-lambda * age)
    """

    def __init__(self, config: HaloConfig):
        self.config = config
        self.quality_filter = QualityFilter(config.quality_filter)
        self.last_received_pixels: Dict[str, np.ndarray] = {}

    def evaluate_delivery(
        self,
        img: ImageRecord,
        delivery_time_sec: float,
        active_alerts: List[GroundAlert],
    ) -> Tuple[float, bool]:
        """Compute objective useful information delivered to the ground station."""
        import cv2
        from skimage.metrics import structural_similarity as ssim

        # 1. Image Quality factor
        q_res = self.quality_filter.check_quality(img.pixels)
        if q_res.passes:
            quality_q = q_res.quality_score
        else:
            quality_q = 0.02 if q_res.reason == "blank_image" else 0.05

        # 2. Scene usefulness
        usefulness_m = img.ground_truth_importance

        # 3. Novelty / Change versus what ground already received for this tile
        tile = img.tile_id
        if tile not in self.last_received_pixels:
            change_c = 1.0
            self.last_received_pixels[tile] = img.pixels.copy()
        else:
            prev_pixels = self.last_received_pixels[tile]
            gray_curr = cv2.cvtColor(img.pixels, cv2.COLOR_RGB2GRAY)
            gray_prev = cv2.cvtColor(prev_pixels, cv2.COLOR_RGB2GRAY)
            diff = cv2.absdiff(gray_curr, gray_prev)
            mae = float(np.mean(diff)) / 255.0
            raw_change = float(np.clip(mae * 4.0, 0.0, 1.0))
            if raw_change < self.config.duplicate_filter.min_change_threshold:
                # Ground already has this scene: duplicates bring near-zero new information
                change_c = max(0.02, raw_change * 0.2)
            else:
                change_c = float(np.clip(0.40 + 0.60 * raw_change, 0.40, 1.0))
                self.last_received_pixels[tile] = img.pixels.copy()

        # 4. Alert boost at delivery time
        alert_boost = 1.0
        for alert in active_alerts:
            if alert.region_tile == tile and alert.event_type in img.ground_truth_label:
                alert_boost = alert.boost_factor
                break

        # 5. Decay factor
        age_hr = max(0.0, (delivery_time_sec - img.timestamp) / 3600.0)
        decay_rate = self.config.decay_rates.get_decay_rate(img.ground_truth_label)
        decay_factor = float(np.exp(-decay_rate * age_hr))

        u_delivered = float(quality_q * usefulness_m * change_c * alert_boost * decay_factor)
        is_stale = (u_delivered < self.config.utility_model.expiry_threshold)
        return u_delivered, is_stale


class SimulationRunner:
    """Executes mission simulations for any designated policy or ablation."""

    def __init__(self, config: Optional[HaloConfig] = None):
        """Initialize runner with config and onboard pipeline components.

        Args:
            config: HaloConfig instance.
        """
        self.config = config or load_config()
        self.quality_filter = QualityFilter(self.config.quality_filter)
        self.duplicate_filter = DuplicateFilter(self.config.duplicate_filter)
        self.scorer = HALOScorer(self.config)
        self.utility_model = UtilityModel(self.config)

    def run(
        self,
        method: str,
        seed: int = 42,
        stream: Optional[List[ImageRecord]] = None,
    ) -> Tuple[EvaluationMetrics, List[DownlinkedItemRecord], List[Tuple[float, int]]]:
        """Execute simulation run for the specified method.

        Args:
            method: Name of method.
            seed: Seed for reproducibility.
            stream: Pre-generated stream or None.

        Returns:
            Tuple of (EvaluationMetrics, downlinked_records, storage_timeseries).
        """
        rng = np.random.RandomState(seed)
        random.seed(seed)

        if stream is None:
            stream = generate_image_stream(self.config, seed=seed)

        # Reset onboard state and ground station evaluator
        self.duplicate_filter.reset()
        ground_evaluator = GroundStationEvaluator(self.config)
        alert_mgr = AlertManager(self.config)
        storage_mgr = StorageManager(self.config)
        queue = UtilityQueue(self.config)

        passes = build_ground_pass_schedule(self.config)

        # Ablation and behavior toggles
        is_fifo = (method == "fifo")
        is_random = (method == "random")
        is_classifier_only = (method == "classifier_only")

        enable_alerts = not (method in ["fifo", "random", "classifier_only", "ablation_no_alerts"])
        enable_aging = not (method in ["fifo", "random", "classifier_only", "ablation_no_aging"])
        enable_expiry = not (method in ["fifo", "random", "classifier_only", "priority_aging_no_expiry", "ablation_no_expiry"])
        enable_dedup = not (method in ["ablation_no_dedup"])
        use_knapsack = (method == "halo_knapsack")

        # Storage and metric trackers
        downlinked_records: List[DownlinkedItemRecord] = []
        storage_ts: List[float] = []
        storage_bytes: List[int] = []
        alert_delivery_times: Dict[str, float] = {}

        # Build chronological discrete-event timeline
        events: List[Tuple[float, str, Any]] = []
        for img in stream:
            events.append((img.timestamp, "capture", img))
        for p in passes:
            events.append((p.start_time, "pass_start", p))

        events.sort(key=lambda x: x[0])

        for event_time, event_type, event_obj in events:
            # 1. Update utility decay and aging for waiting images up to event_time
            if not is_fifo and not is_random and not is_classifier_only:
                queue.update_scores(event_time)
                if not enable_aging:
                    # Strip aging bonus if aging is ablated
                    for it in queue.items:
                        it.current_score = it.current_utility
            elif is_classifier_only:
                for it in queue.items:
                    it.current_score = it.base_score
                    it.current_utility = it.base_score

            storage_ts.append(event_time)
            storage_bytes.append(queue.total_bytes)

            # -------------------------------------------------------------
            # Capture Event: Ingest image through onboard pipeline
            # -------------------------------------------------------------
            if event_type == "capture":
                img_record: ImageRecord = event_obj

                # FIFO and Random buffer raw incoming frames up to storage capacity
                if is_fifo or is_random:
                    queue_item = UtilityQueueItem(
                        image=img_record,
                        base_score=img_record.ground_truth_importance,
                        decay_rate=self.config.decay_rates.get_decay_rate(img_record.ground_truth_label),
                    )
                    queue_item.current_score = 1.0
                    queue_item.current_utility = queue_item.base_score
                    if queue.total_bytes + img_record.size_bytes <= storage_mgr.capacity_bytes:
                        queue.add(queue_item)
                    continue

                # Stage 2: Quality filter
                q_res = self.quality_filter.check_quality(img_record.pixels)
                if not q_res.passes:
                    continue  # Dropped bad image (cloud/blur/blank)

                # Stage 3: Duplicate check
                if enable_dedup:
                    dup_res = self.duplicate_filter.check_duplicate(img_record.tile_id, img_record.pixels)
                    if dup_res.is_duplicate:
                        continue  # Dropped near-duplicate
                    change_c = dup_res.change_score
                    self.duplicate_filter.record_kept(img_record.tile_id, img_record.pixels)
                else:
                    change_c = 1.0

                # Stage 4: Onboard Scorer (INT8 MobileNet with stream memoization)
                if "scorer_result" in img_record.metadata:
                    pred_cls, usefulness_m, _ = img_record.metadata["scorer_result"]
                else:
                    pred_cls, usefulness_m, conf = self.scorer.score_image(img_record.pixels)
                    img_record.metadata["scorer_result"] = (pred_cls, usefulness_m, conf)

                base_b = compute_base_score(q_res.quality_score, usefulness_m, change_c)
                decay_rate = self.config.decay_rates.get_decay_rate(pred_cls)

                queue_item = UtilityQueueItem(
                    image=img_record,
                    base_score=base_b,
                    decay_rate=decay_rate,
                )
                queue_item.current_utility = base_b
                queue_item.current_score = base_b

                # Enforce storage limits via StorageManager
                storage_mgr.enforce_storage_limit(queue, incoming_item=queue_item)
                queue.add(queue_item)

            # -------------------------------------------------------------
            # Ground Pass Event: Downlink window opens
            # -------------------------------------------------------------
            elif event_type == "pass_start":
                ground_pass: GroundPass = event_obj

                # Uplink alert messages delivered during this pass
                new_alerts = alert_mgr.on_pass_occurred(ground_pass.pass_index, event_time)
                for a in new_alerts:
                    alert_delivery_times[a.event_type] = event_time

                # Apply active alert boosts & covered area rule
                alert_mgr.apply_to_queue(queue, event_time, enable_alerts=enable_alerts)
                queue.update_scores(event_time)

                # Stage 6: Delivery-time Expiry Check
                t_next = get_next_pass_start_time(event_time + 60.0, passes)
                if enable_expiry:
                    queue.prune_expired(event_time, t_next, enable_expiry=True)

                # Stage 7: Scheduler selection
                budget_d = ground_pass.budget_bytes

                if is_fifo:
                    selected_items: List[UtilityQueueItem] = []
                    curr_b = 0
                    for it in list(queue.items):
                        if curr_b + it.size_bytes <= budget_d:
                            selected_items.append(it)
                            curr_b += it.size_bytes
                elif is_random:
                    candidates = list(queue.items)
                    rng.shuffle(candidates)
                    selected_items = []
                    curr_b = 0
                    for it in candidates:
                        if curr_b + it.size_bytes <= budget_d:
                            selected_items.append(it)
                            curr_b += it.size_bytes
                elif is_classifier_only:
                    candidates = sorted(queue.items, key=lambda it: it.base_score, reverse=True)
                    selected_items = []
                    curr_b = 0
                    for it in candidates:
                        if curr_b + it.size_bytes <= budget_d:
                            selected_items.append(it)
                            curr_b += it.size_bytes
                elif use_knapsack:
                    selected_items, _, _ = knapsack_schedule(
                        queue.items,
                        budget_d,
                        discretization_kb=self.config.scheduler.knapsack_discretization_kb,
                    )
                else:
                    selected_items, _, _ = greedy_schedule(queue.items, budget_d)

                # Downlink selected items and acknowledge
                active_ground_alerts = alert_mgr.get_active_alerts(event_time)
                for it in selected_items:
                    ground_pass.bytes_downlinked += it.size_bytes
                    ground_pass.images_downlinked.append(it.image)
                    storage_mgr.record_downlinked(it.image.image_id)
                    queue.remove(it)

                    # Compute objective delivered utility on ground
                    u_delivered, is_stale = ground_evaluator.evaluate_delivery(
                        img=it.image,
                        delivery_time_sec=event_time,
                        active_alerts=active_ground_alerts,
                    )

                    downlinked_records.append(
                        DownlinkedItemRecord(
                            image=it.image,
                            pass_index=ground_pass.pass_index,
                            delivery_time_sec=event_time,
                            delivered_utility=u_delivered,
                            priority_score=it.current_score,
                            is_stale=is_stale,
                            was_alert_boosted=(it.alert_boost > 1.0),
                        )
                    )

            storage_ts.append(event_time)
            storage_bytes.append(queue.total_bytes)

        # Compute all 7 evaluation metrics for this run
        metrics = calculate_metrics(
            method_name=method,
            seed=seed,
            downlinked_records=downlinked_records,
            total_passes=len(passes),
            storage_history_timestamps=storage_ts,
            storage_history_bytes=storage_bytes,
            alert_delivery_times=alert_delivery_times,
            config=self.config,
        )

        storage_history = list(zip(storage_ts, storage_bytes))
        return metrics, downlinked_records, storage_history
