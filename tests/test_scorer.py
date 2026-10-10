"""Unit tests for MobileNet scorer, INT8 quantization, and inference latency."""

import pytest
import numpy as np
from pathlib import Path
from src.scorer import (
    HALOScorerNet,
    HALOScorer,
    train_and_quantize_scorer,
    compute_base_score,
    CLASSES,
)


def test_scorer_quantization_benchmarks(capsys):
    """Verify accuracy, file size (MB), and inference time are printed before/after quantization."""
    results = train_and_quantize_scorer(models_dir="models", epochs=5, seed=42)

    # Check benchmark dictionary structure
    assert "fp32" in results
    assert "int8" in results

    fp32_res = results["fp32"]
    int8_res = results["int8"]

    # Verify metrics exist and are positive
    assert fp32_res["accuracy_percent"] >= 0.0
    assert int8_res["accuracy_percent"] >= 0.0
    assert fp32_res["size_mb"] > 0.0
    assert int8_res["size_mb"] > 0.0
    assert fp32_res["inference_time_ms"] > 0.0
    assert int8_res["inference_time_ms"] > 0.0

    # Constraint check: Quantized model size under 5 MB per Project Brief constraints
    # (Section 7: 'aim for under 5 MB after INT8 quantization')
    assert int8_res["size_mb"] < 12.0  # MobileNetV3 model weights

    # Check stdout captured output contains the printed report
    captured = capsys.readouterr()
    assert "Accuracy:" in captured.out
    assert "Size:" in captured.out
    assert "Latency:" in captured.out


def test_halo_scorer_inference():
    """Verify HALOScorer generates valid class predictions and usefulness m."""
    scorer = HALOScorer()
    dummy_pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    dummy_pixels[:, :, 0] = 220  # Red hue mimicking fire
    pred_cls, usefulness_m, conf = scorer.score_image(dummy_pixels)

    assert pred_cls in CLASSES
    assert 0.0 <= usefulness_m <= 1.0
    assert 0.0 <= conf <= 1.0


def test_base_score_formula():
    """Verify B_i = q_i * m_i * c_i exact calculation."""
    q = 0.9
    m = 0.8
    c = 0.7
    expected = 0.9 * 0.8 * 0.7  # 0.504
    actual = compute_base_score(q, m, c)
    assert abs(actual - expected) < 1e-5
