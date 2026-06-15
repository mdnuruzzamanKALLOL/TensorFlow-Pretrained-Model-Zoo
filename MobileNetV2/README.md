# MobileNetV2 — Inverted Residuals and Linear Bottlenecks (TensorFlow / Keras)

**Paper:** MobileNetV2: Inverted Residuals and Linear Bottlenecks
**Authors:** Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
**Conference:** CVPR 2018

---

## Overview

MobileNetV2 introduces two key ideas on top of MobileNetV1:

1. **Inverted Residuals** — the shortcut connects narrow (low-channel) layers while
   the internal expansion happens in between (opposite of ResNet bottlenecks).

2. **Linear Bottleneck** — the final projection in each block uses **no activation**
   (linear output), preventing ReLU from destroying information in low-dimensional
   manifolds.

Result: ~3.5M parameters, ~71.3% Top-1 — *fewer params AND higher accuracy* than V1.

---

## Inverted Residual Block

```
                   ┌── Residual connection ──────────────────┐
Input (H×W×C_in)   │                                         │
        │          │                                         ↓
        ├── [Expand] Conv1×1(C_in × t) + BN + ReLU6         │  (skipped if t=1)
        ├── DWConv3×3(stride)           + BN + ReLU6         │
        └── [Project] Conv1×1(C_out)   + BN   (NO activation)│
Output (H/s × W/s × C_out) ─────── Add ───────────────────→ │
                                  (only if stride=1 and C_in==C_out)
```

**Why linear bottleneck?** Low-dimensional manifolds lose information when passed
through ReLU (negative values zeroed). Removing activation at the bottleneck
preserves the manifold intact for the residual to add onto.

---

## Architecture

```
Input (224×224×3)
│
├── Stem: Conv3×3/2(32) + BN + ReLU6        → 112×112×32
│
├── expanded_conv: t=1, c=16,  n=1, s=1     → 112×112×16
├── block_1..2:    t=6, c=24,  n=2, s=2     →  56×56×24
├── block_3..5:    t=6, c=32,  n=3, s=2     →  28×28×32
├── block_6..9:    t=6, c=64,  n=4, s=2     →  14×14×64
├── block_10..12:  t=6, c=96,  n=3, s=1     →  14×14×96
├── block_13..15:  t=6, c=160, n=3, s=2     →   7×7×160
├── block_16:      t=6, c=320, n=1, s=1     →   7×7×320
├── Conv_1: Conv1×1(1280) + BN + ReLU6      →   7×7×1280
│
└── GlobalAvgPool → Dense(num_classes, softmax)
```

---

## V1 vs V2 Comparison

| | MobileNetV1 | MobileNetV2 |
|--|-------------|-------------|
| Block | DWConv + PWConv | Expand + DWConv + Linear Project |
| Residual | No | Yes (stride=1, same dims) |
| Activation | ReLU6 everywhere | ReLU6 except last projection |
| Head | GAP directly | Conv1×1(1280) → GAP |
| Params | ~4.2M | ~3.5M |
| Top-1 | ~70.6% | ~71.3% |

---

## Key Stats

| Property | Value |
|----------|-------|
| Parameters | ~3.5M |
| Top-1 (ImageNet) | ~71.3% |
| Input size | 224×224 |
| Total IR blocks | 17 |
| Head channels | 1280 |

---

## Transfer Learning

```python
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

base_model = keras.applications.MobileNetV2(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = keras.layers.GlobalAveragePooling2D()(base_model.output)  # 1280-dim
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = keras.Model(base_model.input, outputs)

datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input)  # x/127.5 - 1.0
```

---

## Folder Structure

```
MobileNetV2/
├── README.md
├── NoteBook/
│   └── mobilenetv2.ipynb
├── Python Scripts/
│   ├── mobilenetv2.py        — build_mobilenetv2(alpha=1.0) from scratch
│   ├── train.py
│   ├── inference.py
│   └── How to run.txt
└── Using Weight File/
    ├── load_pretrained.py
    ├── feature_extraction.py
    ├── fine_tuning.py        — unfreeze block_13-16 + Conv_1
    └── How to run.txt
```

---

## Citation

```bibtex
@inproceedings{sandler2018mobilenetv2,
  title     = {MobileNetV2: Inverted Residuals and Linear Bottlenecks},
  author    = {Sandler, Mark and Howard, Andrew and Zhu, Menglong and
               Zhmoginov, Andrey and Chen, Liang-Chieh},
  booktitle = {CVPR},
  year      = {2018}
}
```
