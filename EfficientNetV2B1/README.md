# EfficientNetV2B1

**Framework:** TensorFlow / Keras  **Input:** 240×240×3  **Params:** ~8.2M  **Top-1:** ~79.8%

## Architecture

EfficientNetV2-B1 applies depth coefficient 1.1× to B0 (repeat counts rounded up with `ceil`).
Same filters as B0; deeper stages catch more complex patterns at higher resolution.

| Stage | Type | Grid | n | Out-ch | Expand | SE |
|-------|------|------|---|--------|--------|----|
| 0 | Fused-MBConv | 120×120 | 2 | 16 | 1 | — |
| 1 | Fused-MBConv | 60×60 | 3 | 32 | 4 | — |
| 2 | Fused-MBConv | 30×30 | 3 | 48 | 4 | — |
| 3 | MBConv | 15×15 | 4 | 96 | 4 | 0.25 |
| 4 | MBConv | 15×15 | 6 | 112 | 6 | 0.25 |
| 5 | MBConv | 8×8 | 9 | 192 | 6 | 0.25 |
| Head | Conv1×1 | 8×8 | — | 1280 | — | — |

## Preprocessing
```python
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
# [0,255] -> [-1,1]
```
## Fine-Tuning

| Phase | Layers | LR | Epochs |
|-------|--------|----|--------|
| 1 | Head only | 1e-3 | 10 |
| 2 | Last 40% backbone (stages 4-5 + head conv) | 1e-5 | 20 |
