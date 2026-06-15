# EfficientNetV2B0

**Framework:** TensorFlow / Keras  **Input:** 224×224×3  **Params:** ~7.2M  **Top-1:** ~78.7%

## Architecture

EfficientNetV2-B0 is the base mobile variant (compound coefficients φ=0: width=1.0, depth=1.0).
Stage 0 reduces the 32-channel stem to 16 channels via a single Fused-MBConv (expand=1, no residual).

| Stage | Type | Grid | n | Out-ch | Expand | SE |
|-------|------|------|---|--------|--------|----|
| 0 | Fused-MBConv | 112×112 | 1 | 16 | 1 | — |
| 1 | Fused-MBConv | 56×56 | 2 | 32 | 4 | — |
| 2 | Fused-MBConv | 28×28 | 2 | 48 | 4 | — |
| 3 | MBConv | 14×14 | 3 | 96 | 4 | 0.25 |
| 4 | MBConv | 14×14 | 5 | 112 | 6 | 0.25 |
| 5 | MBConv | 7×7 | 8 | 192 | 6 | 0.25 |
| Head | Conv1×1 | 7×7 | — | 1280 | — | — |

## Preprocessing
```python
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
# [0,255] -> [-1,1]  (same for all V2B variants)
```
## Fine-Tuning

| Phase | Layers | LR | Epochs |
|-------|--------|----|--------|
| 1 | Head only | 1e-3 | 10 |
| 2 | Last 40% backbone (stages 4-5 + head conv) | 1e-5 | 20 |
