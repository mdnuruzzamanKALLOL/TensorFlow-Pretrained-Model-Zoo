# EfficientNetV2L

**Framework:** TensorFlow / Keras
**Input size:** 480 × 480 × 3
**Parameters:** ~119.0 M
**ImageNet Top-1:** ~85.7%

## Architecture

EfficientNetV2-L is the largest variant, using a wider stem (32 vs 24 filters)
and significantly deeper stages. Stage 5 alone has 25 MBConv blocks.

| Stage | Type | Grid | Blocks | Out-ch | Expand | SE |
|-------|------|------|--------|--------|--------|----|
| 0 | Fused-MBConv | 240×240 | 4 | 32 | 1 | — |
| 1 | Fused-MBConv | 120×120 | 7 | 64 | 4 | — |
| 2 | Fused-MBConv | 60×60 | 7 | 96 | 4 | — |
| 3 | MBConv | 30×30 | 10 | 192 | 4 | 0.25 |
| 4 | MBConv | 30×30 | 19 | 224 | 6 | 0.25 |
| 5 | MBConv | 15×15 | 25 | 384 | 6 | 0.25 |
| 6 | MBConv | 15×15 | 7 | 640 | 6 | 0.25 |
| Head | Conv1×1 | 15×15 | — | 1280 | — | — |

**Activation:** SiLU (Swish) throughout. **SE ratio:** input_channels × 0.25.
**Stem:** 32 filters (wider than V2S/V2M which use 24).

## Preprocessing

```python
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
# Maps [0, 255] -> [-1, 1]  (x / 127.5 - 1.0)
# Same preprocess_input for V2S, V2M, and V2L
```

## File Structure

```
EfficientNetV2L/
├── NoteBook/EfficientNetV2L.ipynb
├── Python Scripts/
│   ├── EfficientNetV2L.py   # architecture from scratch
│   ├── train.py
│   ├── inference.py
│   └── How to run.txt
├── Using Weight File/
│   ├── load_pretrained.py
│   ├── feature_extraction.py
│   ├── fine_tuning.py
│   └── How to run.txt
└── README.md
```

## Fine-Tuning Strategy

| Phase | Layers | LR | Epochs |
|-------|--------|----|--------|
| 1 | Head only | 1e-3 | 10 |
| 2 | Last 30% of backbone (stage 6 + tail of stage 5 + head conv) | 1e-5 | 20 |

V2L uses 30% (vs 40% for S/M) because stage 5 has 25 deep MBConv blocks —
unfreezing 40% would include too many parameters and risk overfitting on small datasets.

## Memory Tips

```python
# Mixed precision (reduces VRAM ~40%):
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Or reduce batch size to 2 with gradient accumulation
```
