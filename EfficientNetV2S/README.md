# EfficientNetV2S

**Framework:** TensorFlow / Keras
**Input size:** 300 × 300 × 3
**Parameters:** ~21.6 M
**ImageNet Top-1:** ~83.9%

## Architecture

EfficientNetV2-S (Tan & Le, 2021) introduces **Fused-MBConv** for early stages and
**progressive learning** during training. Key differences from EfficientNetV1:

| Stage | Type | Grid | Blocks | Out-ch | Expand | SE |
|-------|------|------|--------|--------|--------|----|
| 0 | Fused-MBConv | 150×150 | 2 | 24 | 1 | — |
| 1 | Fused-MBConv | 75×75 | 4 | 48 | 4 | — |
| 2 | Fused-MBConv | 38×38 | 4 | 64 | 4 | — |
| 3 | MBConv | 19×19 | 6 | 128 | 4 | 0.25 |
| 4 | MBConv | 19×19 | 9 | 160 | 6 | 0.25 |
| 5 | MBConv | 10×10 | 15 | 256 | 6 | 0.25 |
| Head | Conv1×1 | 10×10 | — | 1280 | — | — |

**Fused-MBConv** replaces the depthwise conv + 1×1 expand of MBConv with a single
3×3 convolution. For expand=1 stages, this is equivalent to a standard conv residual block.
**Activation:** SiLU (Swish) throughout. **SE ratio:** input_channels × 0.25.

## Preprocessing

```python
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
# Maps [0, 255] -> [-1, 1]  (x / 127.5 - 1.0)
```

## File Structure

```
EfficientNetV2S/
├── NoteBook/EfficientNetV2S.ipynb
├── Python Scripts/
│   ├── EfficientNetV2S.py   # architecture from scratch
│   ├── train.py             # training
│   ├── inference.py         # single-image inference
│   └── How to run.txt
├── Using Weight File/
│   ├── load_pretrained.py   # head-only transfer learning
│   ├── feature_extraction.py
│   ├── fine_tuning.py       # two-phase fine-tuning
│   └── How to run.txt
└── README.md
```

## Fine-Tuning Strategy

| Phase | Layers | LR | Epochs |
|-------|--------|----|--------|
| 1 | Head only | 1e-3 | 10 |
| 2 | Last 40% of backbone (MBConv stages 4-5 + head conv) | 1e-5 | 20 |
