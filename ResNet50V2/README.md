# ResNet50V2 — Identity Mappings in Deep Residual Networks (TensorFlow / Keras)

**Paper:** Identity Mappings in Deep Residual Networks
**Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
**Conference:** ECCV 2016

---

## Overview

ResNet50V2 introduces **pre-activation** residual blocks — moving BatchNorm and ReLU
to *before* each convolution rather than after. This creates a **true identity shortcut
path**: the gradient flows from output to input with no transformation in between,
enabling better optimisation and slightly higher accuracy than ResNet50.

The key insight: in the original ResNet, the shortcut passes through a ReLU, which
blocks negative gradient signals. Pre-activation removes this bottleneck.

---

## Pre-activation vs Post-activation

| | ResNet50 (V1) | ResNet50V2 (V2) |
|--|--|--|
| Conv order | Conv → BN → ReLU | **BN → ReLU → Conv** |
| Shortcut path | Passes through Add → ReLU | **True identity (Add only)** |
| Stem | Conv → BN → ReLU → Pool | **Conv → Pool** (no BN/ReLU) |
| Final layer | Last block has BN+ReLU | **Extra post-BN+ReLU before GAP** |
| Preprocessing | Subtract BGR mean (~caffe) | **Scale to [-1, 1] (~tf mode)** |
| Top-1 ImageNet | ~74.9% | **~75.6%** |

---

## Pre-activation Bottleneck Block

```
Input (x)
  |
  +-- BN + ReLU  (= preact)
      |
      +-- [if conv_shortcut] Conv1x1(filters*4, stride) -> shortcut
      |   [else             ] x  (identity, or MaxPool if stride>1)
      |
      Conv1x1(filters)   -> BN -> ReLU
      Conv3x3(filters, stride) -> BN -> ReLU
      Conv1x1(filters*4)          [no BN/ReLU — next block's preact handles it]
      |
      Add([conv_path, shortcut]) -> output
```

The last Conv1×1 has **no** BN or ReLU after it; the next block's BN+ReLU handles
normalisation, keeping the shortcut path clean.

---

## Architecture

```
Input (224×224×3)
│
├── Stem  : Conv7×7/2 [no BN/ReLU] + MaxPool3×3/2   →   64 × 56×56
│
├── Stage 1 (conv2): 3 × PreActBottleneck(64)  s=1   →  256 × 56×56
├── Stage 2 (conv3): 4 × PreActBottleneck(128) s=2   →  512 × 28×28
├── Stage 3 (conv4): 6 × PreActBottleneck(256) s=2   → 1024 × 14×14
├── Stage 4 (conv5): 3 × PreActBottleneck(512) s=2   → 2048 ×  7×7
│
├── Post BN + ReLU  [unique to V2]
└── GlobalAvgPool → Dense(num_classes, softmax)
```

---

## Key Stats

| Property | Value |
|----------|-------|
| Parameters | ~25.6M |
| Top-1 (ImageNet) | ~75.6% |
| Top-5 (ImageNet) | ~92.8% |
| Input size | 224×224 |
| Framework | TensorFlow / Keras |

---

## Training Configuration (From Scratch)

| Setting | Value |
|---------|-------|
| Input size | 224×224 |
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
from tensorflow.keras.applications.resnet_v2 import preprocess_input

base_model = keras.applications.ResNet50V2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3),
)
x       = layers.GlobalAveragePooling2D()(base_model.output)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model   = keras.Model(inputs=base_model.input, outputs=outputs)

# IMPORTANT: ResNet50V2 uses "tf" mode preprocessing — scale to [-1, 1]
# This is DIFFERENT from ResNet50 (which subtracts BGR mean)
datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input   # x / 127.5 - 1.0
)
```

### Two-Phase Fine-Tuning

```python
# Phase 1 — freeze full backbone
base_model.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-3), ...)
model.fit(train_gen, epochs=10, ...)

# Phase 2 — unfreeze conv4, conv5 stages + post_bn/relu
base_model.trainable = True
for layer in base_model.layers:
    keep = layer.name.startswith('conv4') or layer.name.startswith('conv5') \
           or layer.name in ('post_bn', 'post_relu')
    layer.trainable = keep
model.compile(optimizer=keras.optimizers.Adam(1e-5), ...)
model.fit(train_gen, initial_epoch=10, epochs=30, ...)
```

---

## Folder Structure

```
ResNet50V2/
├── README.md
├── NoteBook/
│   └── resnet50v2.ipynb         — 17-cell notebook (arch + train + ROC AUC)
├── Python Scripts/
│   ├── resnet50v2.py            — build_resnet50v2() from scratch
│   ├── train.py                 — Adam + ReduceLROnPlateau training loop
│   ├── inference.py             — top-K single-image prediction
│   └── How to run.txt
└── Using Weight File/
    ├── load_pretrained.py       — load Keras Applications ResNet50V2
    ├── feature_extraction.py    — frozen backbone, GAP + Dense head
    ├── fine_tuning.py           — two-phase fine-tuning (conv4+5 unfreeze)
    └── How to run.txt
```

---

## Citation

```bibtex
@inproceedings{he2016identity,
  title     = {Identity Mappings in Deep Residual Networks},
  author    = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle = {ECCV},
  year      = {2016}
}
```
