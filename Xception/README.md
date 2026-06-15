# Xception — Extreme Inception via Depthwise Separable Convolutions

**Paper:** Xception: Deep Learning with Depthwise Separable Convolutions
**Author:** François Chollet
**Conference:** CVPR 2017

---

## Overview

Xception (Extreme Inception) is based on the hypothesis that cross-channel correlations
and spatial correlations in feature maps can be **fully decoupled**. This leads to
replacing Inception modules entirely with **depthwise separable convolutions** — a
depthwise spatial convolution (one filter per input channel) followed by a pointwise
(1×1) convolution.

Xception is built on the same principles as Inception V3 but takes the idea to the
extreme. The result is a model that achieves better accuracy with fewer parameters.

---

## Depthwise Separable Convolution

A standard Conv2D(in=32, out=64, 3×3) costs: `32 × 64 × 3 × 3` = 18,432 multiplications per spatial location.

A depthwise separable equivalent:
- DepthwiseConv2D(32 channels, 3×3): `32 × 3 × 3` = 288 multiplications
- PointwiseConv2D(32→64, 1×1):  `32 × 64` = 2,048 multiplications
- **Total: 2,336** — roughly 8× fewer than standard conv

---

## Architecture

```
Input (3 × 299 × 299)
│
├── Entry Flow
│   ├── Conv2D(32, 3×3/2) + BN + ReLU          → 32  × 150×150
│   ├── Conv2D(64, 3×3/1) + BN + ReLU          → 64  × 150×150
│   ├── SepBlock(128) [stride-2, residual]       → 128 ×  75×75
│   ├── SepBlock(256) [stride-2, residual]       → 256 ×  38×38
│   └── SepBlock(728) [stride-2, residual]       → 728 ×  19×19
│
├── Middle Flow (×8)
│   └── SepBlock(728) [identity residual]        → 728 ×  19×19
│
├── Exit Flow
│   ├── SepBlock(728→1024) [stride-2, residual]  → 1024 × 10×10
│   ├── SepConv(1536) + ReLU                     → 1536 × 10×10
│   └── SepConv(2048) + ReLU                     → 2048 × 10×10
│
└── GlobalAveragePool → Dense(num_classes, softmax)
```

Each SepBlock: `[ReLU →] SeparableConv2D → BN → ReLU → SeparableConv2D → BN → MaxPool + residual`

---

## Key Stats

| Property | Value |
|----------|-------|
| Parameters | ~22.9M |
| Top-1 (ImageNet) | ~79.0% |
| Top-5 (ImageNet) | ~94.5% |
| Input size | 299×299 |
| Framework | TensorFlow / Keras |

---

## Training Configuration (From Scratch)

| Setting | Value |
|---------|-------|
| Input size | 299×299 |
| Batch size | 32 |
| Optimizer | Adam (lr=1e-3) |
| Scheduler | ReduceLROnPlateau (factor=0.1, patience=5) |
| Loss | categorical_crossentropy |
| Epochs | 30 |

---

## Transfer Learning

```python
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.xception import preprocess_input

base_model = keras.applications.Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(299, 299, 3),
)
x       = layers.GlobalAveragePooling2D()(base_model.output)
x       = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model   = keras.Model(inputs=base_model.input, outputs=outputs)

# Preprocessing (important — scales to [-1, 1])
datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input
)
```

### Two-Phase Fine-Tuning

```python
# Phase 1 — freeze backbone, train head
base_model.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-3), ...)
model.fit(train_gen, epochs=10, ...)

# Phase 2 — unfreeze, fine-tune with lower lr
base_model.trainable = True
model.compile(optimizer=keras.optimizers.Adam(1e-5), ...)
model.fit(train_gen, initial_epoch=10, epochs=30, ...)
```

---

## Folder Structure

```
Xception/
├── README.md
├── NoteBook/
│   └── xception.ipynb         — 17-cell notebook (arch + train + ROC AUC)
├── Python Scripts/
│   ├── xception.py            — build_xception() from scratch
│   ├── train.py               — training loop (Adam + ReduceLROnPlateau)
│   ├── inference.py           — top-K single-image prediction
│   └── How to run.txt
└── Using Weight File/
    ├── load_pretrained.py     — load Keras Applications Xception
    ├── feature_extraction.py  — frozen backbone, head only
    ├── fine_tuning.py         — two-phase fine-tuning
    └── How to run.txt
```

---

## Citation

```bibtex
@inproceedings{chollet2017xception,
  title     = {Xception: Deep Learning with Depthwise Separable Convolutions},
  author    = {Chollet, Fran{\c{c}}ois},
  booktitle = {CVPR},
  year      = {2017}
}
```
