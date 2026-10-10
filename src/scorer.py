"""Onboard image classification and usefulness scoring using quantized MobileNet.

Implements Stage 4 of the onboard pipeline:
Runs INT8 quantized MobileNet on captured patches to predict scene class and usefulness value m.
Reports accuracy, model file size (MB), and inference time (ms) before and after quantization.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image

from src.config import HaloConfig, load_config


CLASSES = [
    "fire_or_smoke",
    "flood",
    "ship_or_vehicle",
    "highway",
    "industrial",
    "residential",
    "forest",
    "annual_crop",
    "sea_lake",
    "cloudy_or_blank",
]

CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASSES)}


class HALOScorerNet(nn.Module):
    """MobileNet-based neural network for scene classification and usefulness scoring."""

    def __init__(self, num_classes: int = len(CLASSES)):
        """Initialize the model backbone and classification heads.

        Args:
            num_classes: Number of target scene classes.
        """
        super().__init__()
        # Use MobileNetV3-Small as it fits strict embedded satellite power/storage constraints
        backbone = models.mobilenet_v3_small(weights=None)
        in_features = backbone.classifier[0].in_features
        # Replace classifier with custom dual head (class logits and calibrated usefulness)
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )
        self.usefulness_head = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            Tuple of (class_logits, usefulness_scores).
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        usefulness = self.usefulness_head(features).squeeze(-1)
        return logits, usefulness


def preprocess_image(pixels: np.ndarray, target_size: Tuple[int, int] = (64, 64)) -> torch.Tensor:
    """Preprocess a uint8 numpy RGB array into a normalized PyTorch tensor.

    Args:
        pixels: uint8 image array (H, W, 3).
        target_size: Target dimensions (H, W).

    Returns:
        torch.Tensor of shape (1, 3, H, W) normalized to ImageNet mean/std.
    """
    img = Image.fromarray(pixels).resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    # Transpose to (3, H, W)
    arr = np.transpose(arr, (2, 0, 1))
    # Normalize with standard satellite/ImageNet statistics
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    norm = (arr - mean) / std
    return torch.from_numpy(norm).unsqueeze(0)


def create_training_data(
    num_samples_per_class: int = 40,
    rng_seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate representative multi-spectral satellite training samples with ground-truth.

    Args:
        num_samples_per_class: Number of samples per scene class.
        rng_seed: Deterministic seed.

    Returns:
        Tuple of (X_tensor, y_class_tensor, y_usefulness_tensor).
    """
    from src.stream import create_synthetic_patch
    config = load_config()
    rng = np.random.RandomState(rng_seed)

    x_list = []
    y_cls_list = []
    y_use_list = []

    for cls_idx, cls_name in enumerate(CLASSES):
        usefulness = config.model.get_usefulness(cls_name)
        for _ in range(num_samples_per_class):
            patch = create_synthetic_patch(cls_name, size=(64, 64), rng=rng)
            tensor = preprocess_image(patch).squeeze(0)
            x_list.append(tensor)
            y_cls_list.append(cls_idx)
            # Add slight perturbation to usefulness ground truth
            jittered_use = np.clip(usefulness + rng.normal(0, 0.02), 0.0, 1.0)
            y_use_list.append(float(jittered_use))

    x_t = torch.stack(x_list)
    y_cls_t = torch.tensor(y_cls_list, dtype=torch.long)
    y_use_t = torch.tensor(y_use_list, dtype=torch.float32)
    return x_t, y_cls_t, y_use_t


def train_and_quantize_scorer(
    models_dir: str = "models",
    epochs: int = 15,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train MobileNet scorer, apply INT8 quantization, and report accuracy/size/speed.

    Args:
        models_dir: Directory where model weights are stored.
        epochs: Training epochs.
        seed: Random seed.

    Returns:
        Dict containing benchmark metrics before and after quantization.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(models_dir, exist_ok=True)
    fp32_path = Path(models_dir) / "halo_mobilenet_v3_fp32.pt"
    int8_path = Path(models_dir) / "halo_mobilenet_v3_int8.pt"

    # 1. Prepare data
    x_train, y_train_cls, y_train_use = create_training_data(num_samples_per_class=35, rng_seed=seed)
    x_test, y_test_cls, y_test_use = create_training_data(num_samples_per_class=15, rng_seed=seed + 100)

    # 2. Build model and optimizer
    model_fp32 = HALOScorerNet(num_classes=len(CLASSES))
    optimizer = torch.optim.Adam(model_fp32.parameters(), lr=1e-3, weight_decay=1e-4)
    cls_criterion = nn.CrossEntropyLoss()
    use_criterion = nn.MSELoss()

    model_fp32.train()
    dataset = torch.utils.data.TensorDataset(x_train, y_train_cls, y_train_use)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        for bx, by_cls, by_use in loader:
            optimizer.zero_grad()
            logits, use_pred = model_fp32(bx)
            loss = cls_criterion(logits, by_cls) + 2.0 * use_criterion(use_pred, by_use)
            loss.backward()
            optimizer.step()

    # 3. Evaluate FP32 model
    model_fp32.eval()
    with torch.no_grad():
        test_logits, _ = model_fp32(x_test)
        pred_cls = torch.argmax(test_logits, dim=-1)
        accuracy_fp32 = (pred_cls == y_test_cls).float().mean().item() * 100.0

        # Benchmark FP32 latency (ms per image)
        latencies = []
        for _ in range(50):
            sample = x_test[0:1]
            t0 = time.perf_counter()
            _ = model_fp32(sample)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        latency_fp32_ms = float(np.median(latencies))

    torch.save(model_fp32.state_dict(), fp32_path)
    size_fp32_mb = fp32_path.stat().st_size / (1024 * 1024)

    # 4. INT8 Quantization (Dynamic Quantization for mobile/satellite deployment)
    model_int8 = torch.ao.quantization.quantize_dynamic(
        model_fp32,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    model_int8.eval()

    with torch.no_grad():
        test_logits_int8, _ = model_int8(x_test)
        pred_cls_int8 = torch.argmax(test_logits_int8, dim=-1)
        accuracy_int8 = (pred_cls_int8 == y_test_cls).float().mean().item() * 100.0

        latencies_int8 = []
        for _ in range(50):
            sample = x_test[0:1]
            t0 = time.perf_counter()
            _ = model_int8(sample)
            latencies_int8.append((time.perf_counter() - t0) * 1000.0)
        latency_int8_ms = float(np.median(latencies_int8))

    torch.save(model_int8, int8_path)
    size_int8_mb = int8_path.stat().st_size / (1024 * 1024)

    results = {
        "fp32": {
            "accuracy_percent": accuracy_fp32,
            "size_mb": size_fp32_mb,
            "inference_time_ms": latency_fp32_ms,
            "path": str(fp32_path),
        },
        "int8": {
            "accuracy_percent": accuracy_int8,
            "size_mb": size_int8_mb,
            "inference_time_ms": latency_int8_ms,
            "path": str(int8_path),
        },
    }

    # Print required task confirmation
    print("=" * 60)
    print("HALO Onboard Scorer Model Quantization Benchmark:")
    print(f"FP32 Model  -> Accuracy: {accuracy_fp32:.2f}%, Size: {size_fp32_mb:.2f} MB, Latency: {latency_fp32_ms:.2f} ms")
    print(f"INT8 Model  -> Accuracy: {accuracy_int8:.2f}%, Size: {size_int8_mb:.2f} MB, Latency: {latency_int8_ms:.2f} ms")
    print(f"Size Reduction: {((size_fp32_mb - size_int8_mb) / size_fp32_mb) * 100.0:.1f}%")
    print("=" * 60)

    return results


class HALOScorer:
    """Inference engine for the onboard scorer stage."""

    def __init__(self, config: Optional[HaloConfig] = None, use_int8: bool = True):
        """Initialize the onboard scorer.

        Args:
            config: HaloConfig instance.
            use_int8: Whether to load quantized INT8 weights.
        """
        self.config = config or load_config()
        self.use_int8 = use_int8
        self.model = None
        self._load_or_train_model()

    def _load_or_train_model(self) -> None:
        """Load quantized model or train if not already on disk."""
        models_dir = Path("models")
        int8_file = models_dir / "halo_mobilenet_v3_int8.pt"
        if not int8_file.exists():
            train_and_quantize_scorer(models_dir=str(models_dir))

        if self.use_int8 and int8_file.exists():
            try:
                self.model = torch.load(int8_file, map_location="cpu", weights_only=False)
                self.model.eval()
                return
            except Exception:
                pass

        # Fallback to FP32 architecture
        model = HALOScorerNet(num_classes=len(CLASSES))
        fp32_file = models_dir / "halo_mobilenet_v3_fp32.pt"
        if fp32_file.exists():
            model.load_state_dict(torch.load(fp32_file, map_location="cpu", weights_only=True))
        model.eval()
        self.model = model

    def score_image(self, pixels: np.ndarray) -> Tuple[str, float, float]:
        """Classify image and return predicted class, usefulness value m, and confidence.

        Args:
            pixels: RGB image uint8 array.

        Returns:
            Tuple of (predicted_class_name, usefulness_m, confidence).
        """
        tensor = preprocess_image(pixels)
        with torch.no_grad():
            logits, use_score = self.model(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            pred_idx = int(torch.argmax(probs).item())
            confidence = float(probs[pred_idx].item())
            predicted_class = IDX_TO_CLASS.get(pred_idx, "crop_or_land_use")
            calibrated_m = float(use_score.squeeze().item())

            # Map through config class usefulness table for guaranteed policy consistency
            config_usefulness = self.config.model.get_usefulness(predicted_class)
            # Fuse calibrated network output with config domain policy
            fused_m = 0.5 * calibrated_m + 0.5 * config_usefulness

        return predicted_class, fused_m, confidence


def compute_base_score(quality_q: float, model_usefulness_m: float, change_c: float) -> float:
    """Compute base score B_i = q_i * m_i * c_i per Section 4 of brief.

    Args:
        quality_q: Image quality factor in [0, 1].
        model_usefulness_m: Onboard model usefulness in [0, 1].
        change_c: Change score versus previous tile image in [0, 1].

    Returns:
        Base score B_i bounded in [0, 1].
    """
    base = float(quality_q * model_usefulness_m * change_c)
    return float(np.clip(base, 0.0, 1.0))
