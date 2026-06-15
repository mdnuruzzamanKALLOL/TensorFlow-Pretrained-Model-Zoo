# EfficientNetV2B2

**Framework:** TensorFlow / Keras  **Input:** 260×260×3  **Params:** ~10.1M  **Top-1:** ~80.3%

## Architecture

EfficientNetV2-B2 applies both width (1.1×) and depth (1.2×) scaling over B0.
Width scaling uses `round_filters` (nearest multiple of 8, ensures ≥ 0.9× of original).

| Stage | Type | Grid | n | Out-ch | Expand | SE |
|-------|------|------|---|--------|--------|----|
| 0 | Fused-MBConv | 130×130 | 2 | 16 | 1 | — |
| 1 | Fused-MBConv | 65×65 | 3 | 32 | 4 | — |
| 2 | Fused-MBConv | 33×33 | 3 | 56 | 4 | — |
| 3 | MBConv | 17×17 | 4 | 104 | 4 | 0.25 |
| 4 | MBConv | 17×17 | 6 | 120 | 6 | 0.25 |
| 5 | MBConv | 9×9 | 10 | 208 | 6 | 0.25 |
| Head | Conv1×1 | 9×9 | — | 1280 | — | — |

**Width changes vs B0:** 48→56, 96→104, 112→120, 192→208.

## Preprocessing
```python
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
```
## Fine-Tuning

| Phase | Layers | LR | Epochs |
|-------|--------|----|--------|
| 1 | Head only | 1e-3 | 10 |
| 2 | Last 40% backbone (stages 4-5 + head conv) | 1e-5 | 20 |
