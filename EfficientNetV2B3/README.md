# EfficientNetV2B3

**Framework:** TensorFlow / Keras  **Input:** 300×300×3  **Params:** ~14.4M  **Top-1:** ~82.0%

## Architecture

EfficientNetV2-B3 is the largest B-variant. Width coefficient 1.2× widens ALL filter sizes
(using `round_filters` with min-depth guard) and also widens the stem to 40 channels.
Depth coefficient 1.4× deepens all stages.

| Stage | Type | Grid | n | Out-ch | Expand | SE |
|-------|------|------|---|--------|--------|----|
| 0 | Fused-MBConv | 150×150 | 2 | 24 | 1 | — |
| 1 | Fused-MBConv | 75×75 | 3 | 40 | 4 | — |
| 2 | Fused-MBConv | 38×38 | 3 | 56 | 4 | — |
| 3 | MBConv | 19×19 | 5 | 112 | 4 | 0.25 |
| 4 | MBConv | 19×19 | 7 | 136 | 6 | 0.25 |
| 5 | MBConv | 10×10 | 12 | 232 | 6 | 0.25 |
| Head | Conv1×1 | 10×10 | — | 1280 | — | — |

**Stem:** 40 filters (round_filters(32, 1.2)=40) — wider than B0/B1/B2 which use 32.

## Compound Scaling vs B0

| Coefficient | B0 | B1 | B2 | B3 |
|-------------|----|----|----|----|
| Width | 1.0 | 1.0 | 1.1 | **1.2** |
| Depth | 1.0 | 1.1 | 1.2 | **1.4** |
| Resolution | 224 | 240 | 260 | **300** |
| Params | 7.2M | 8.2M | 10.1M | **14.4M** |

## Preprocessing
```python
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
# [0,255] -> [-1,1]  (same preprocess_input for all B0-B3)
```
## Fine-Tuning

| Phase | Layers | LR | Epochs |
|-------|--------|----|--------|
| 1 | Head only | 1e-3 | 10 |
| 2 | Last 40% backbone (stages 4-5 + head conv) | 1e-5 | 20 |
