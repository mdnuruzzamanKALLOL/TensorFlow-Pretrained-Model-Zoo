# EfficientNetV2M

**Framework:** TensorFlow / Keras
**Input size:** 384 × 384 × 3
**Parameters:** ~54.4 M
**ImageNet Top-1:** ~85.3%

## Architecture

EfficientNetV2-M adds a 7th stage compared to V2-S, increases depths in all stages,
and widens early channels from 64→80 in stage 2.

| Stage | Type | Grid | Blocks | Out-ch | Expand | SE |
|-------|------|------|--------|--------|--------|----|
| 0 | Fused-MBConv | 192×192 | 3 | 24 | 1 | — |
| 1 | Fused-MBConv | 96×96 | 5 | 48 | 4 | — |
| 2 | Fused-MBConv | 48×48 | 5 | 80 | 4 | — |
| 3 | MBConv | 24×24 | 7 | 160 | 4 | 0.25 |
| 4 | MBConv | 24×24 | 14 | 176 | 6 | 0.25 |
| 5 | MBConv | 12×12 | 18 | 304 | 6 | 0.25 |
| 6 | MBConv | 12×12 | 5 | 512 | 6 | 0.25 |
| Head | Conv1×1 | 12×12 | — | 1280 | — | — |

**Activation:** SiLU (Swish) throughout. **SE ratio:** input_channels × 0.25.

## Preprocessing

```python
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
# Maps [0, 255] -> [-1, 1]  (x / 127.5 - 1.0)
```

## File Structure

```
EfficientNetV2M/
├── NoteBook/EfficientNetV2M.ipynb
├── Python Scripts/
│   ├── EfficientNetV2M.py   # architecture from scratch
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
| 2 | Last 40% of backbone (stages 5-6 + head conv) | 1e-5 | 20 |
