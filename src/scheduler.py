"""Downlink transmission scheduling algorithms for HALO.

Implements Section 4 scheduler:
max sum_i S_i * x_i  subject to  sum_i size_i * x_i <= D
Provides:
1. Greedy selection by score per byte (S_i / size_i).
2. Exact 0/1 Knapsack selection via dynamic programming with calibrated weight units.
"""

from __future__ import annotations

from typing import List, Tuple

from src.utility import UtilityQueueItem


def greedy_schedule(
    candidates: List[UtilityQueueItem],
    downlink_budget_bytes: int,
) -> Tuple[List[UtilityQueueItem], int, float]:
    """Greedily select images with highest priority score per byte within downlink budget.

    Args:
        candidates: Available waiting images in utility queue.
        downlink_budget_bytes: Downlink capacity D in bytes for the pass.

    Returns:
        Tuple of (selected_items, total_bytes_used, total_score_achieved).
    """
    # Sort descending by score per byte (S_i / size_i)
    sorted_items = sorted(
        candidates,
        key=lambda item: (item.score_per_byte, item.current_score),
        reverse=True,
    )

    selected: List[UtilityQueueItem] = []
    accumulated_bytes = 0
    accumulated_score = 0.0

    for item in sorted_items:
        if item.size_bytes <= 0:
            continue
        if accumulated_bytes + item.size_bytes <= downlink_budget_bytes:
            selected.append(item)
            accumulated_bytes += item.size_bytes
            accumulated_score += item.current_score

    return selected, accumulated_bytes, accumulated_score


def knapsack_schedule(
    candidates: List[UtilityQueueItem],
    downlink_budget_bytes: int,
    discretization_kb: int = 50,
) -> Tuple[List[UtilityQueueItem], int, float]:
    """Solve the 0/1 Knapsack problem to maximize total priority score within downlink budget.

    Args:
        candidates: Available waiting images in utility queue.
        downlink_budget_bytes: Downlink capacity D in bytes for the pass.
        discretization_kb: Size unit in KB for DP discretization to maintain fast solver runtime.

    Returns:
        Tuple of (selected_items, total_bytes_used, total_score_achieved).
    """
    if not candidates or downlink_budget_bytes <= 0:
        return [], 0, 0.0

    # Unit size in bytes
    unit_bytes = max(1024, discretization_kb * 1024)
    capacity_units = int(downlink_budget_bytes // unit_bytes)

    # Discretize weights (rounding up to ensure no budget violation)
    weights = [max(1, int(math_ceil(item.size_bytes, unit_bytes))) for item in candidates]
    # Values scaled to integer for precision DP
    values = [int(item.current_score * 10000.0) for item in candidates]

    n = len(candidates)
    # DP table: dp[i][w] = max value using subset of first i items with weight <= w
    dp = [[0] * (capacity_units + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        wt = weights[i - 1]
        val = values[i - 1]
        for w in range(capacity_units + 1):
            if wt <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - wt] + val)
            else:
                dp[i][w] = dp[i - 1][w]

    # Backtrack to find chosen items
    w = capacity_units
    chosen_indices = []
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            chosen_indices.append(i - 1)
            w -= weights[i - 1]

    # Verify actual exact byte sum does not exceed budget; if slight discretization round-up edge case, prune
    selected: List[UtilityQueueItem] = []
    total_bytes = 0
    total_score = 0.0

    for idx in chosen_indices:
        item = candidates[idx]
        if total_bytes + item.size_bytes <= downlink_budget_bytes:
            selected.append(item)
            total_bytes += item.size_bytes
            total_score += item.current_score

    return selected, total_bytes, total_score


def math_ceil(numerator: int, denominator: int) -> int:
    """Helper for integer ceiling division."""
    return (numerator + denominator - 1) // denominator
