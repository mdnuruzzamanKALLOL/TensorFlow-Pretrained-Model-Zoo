# ResNet50 — Deep Residual Learning (TensorFlow / Keras)

**Paper:** Deep Residual Learning for Image Recognition
**Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
**Conference:** CVPR 2016 (Best Paper Award)

---

## Overview

ResNet introduced **residual (skip) connections** that let gradients flow directly
across layers, solving the vanishing gradient problem that prevented training of
very deep networks. The core formula:

```
output = F(x) + x
```

Instead of learning the full mapping `H(x)`, the network learns the **residual**
`F(x) = H(x) - x`. If the identity is optimal, it is easier to drive F toward
zero than to parameterize an exact identity mapping.

ResNet-50 uses **Bottleneck** blocks (1×1 → 3×3 → 1×1) with `expansion=4`,
making it far more parameter-efficient than a naive stack of 3×3 convolutions.

---

## Bottleneck Block

```
Input
  |
  +-- Conv1x1(in -> filters)       -> BN -> ReLU   (compress channels)
      Conv3x3(filters, stride)     -> BN -> ReLU   (spatial processing)
      Conv1x1(filters -> filters*4)-> BN            (expand channels)
  |
  +-- shortcut: Identity  OR  Conv1x1(in -> filters*4, stride) + BN
  |
  Add + ReLU -> output
```

Projection shortcut is used only when input channels ≠ output channels **or** stride > 1.

---

## Architecture

```
Input (224×224×3)
│
├── Stem  : Conv7×7/2 + BN + ReLU + MaxPool3×3/2    →   64 × 56×56
│
├── Stage 1 (conv2): 3 × Bottleneck(64)   stride=1  →  256 × 56×56
├── Stage 2 (conv3): 4 × Bottleneck(128)  stride=2  →  512 × 28×28
├── Stage 3 (conv4): 6 × Bottleneck(256)  stride=2  → 1024 × 14×14
├── Stage 4 (conv5): 3 × Bottleneck(512)  stride=2  → 2048 ×  7×7
│
└── GlobalAvgPool → Dense(num_classes, softmax)
```

---

## Key Stats

| Property | Value |
|----------|-------|
| Parameters | ~25.6M |
| Top-1 (ImageNet) | ~74.9% |
| Top-5 (ImageNet) | ~92.1% |
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
from tensorflow.keras.applications.resnet50 import preprocess_input

base_model = keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3),
)
x       = layers.GlobalAveragePooling2D()(base_model.output)  # 2048-dim
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model   = keras.Model(inputs=base_model.input, outputs=outputs)

# IMPORTANT: ResNet50 preprocessing — subtracts BGR mean, does NOT /255
datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input
)
```

### Two-Phase Fine-Tuning

```python
# Phase 1 — freeze full backbone
base_model.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-3), ...)
model.fit(train_gen, epochs=10, ...)

# Phase 2 — unfreeze conv4 + conv5 stages only
base_model.trainable = True
for layer in base_model.layers:
    layer.trainable = layer.name.startswith('conv4') or layer.name.startswith('conv5')
model.compile(optimizer=keras.optimizers.Adam(1e-5), ...)
model.fit(train_gen, initial_epoch=10, epochs=30, ...)
```

---

## Folder Structure

```
ResNet50/
├── README.md
├── NoteBook/
│   └── resnet50.ipynb           — 17-cell notebook (arch + train + ROC AUC)
├── Python Scripts/
│   ├── resnet50.py              — build_resnet50() from scratch
│   ├── train.py                 — Adam + ReduceLROnPlateau training loop
│   ├── inference.py             — top-K single-image prediction
│   └── How to run.txt
└── Using Weight File/
    ├── load_pretrained.py       — load Keras Applications ResNet50
    ├── feature_extraction.py    — frozen backbone, GAP + Dense head
    ├── fine_tuning.py           — two-phase fine-tuning (conv4+5 unfreeze)
    └── How to run.txt
```

---

## Citation

```bibtex
@inproceedings{he2016deep,
  title     = {Deep Residual Learning for Image Recognition},
  author    = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle = {CVPR},
  year      = {2016}
}
```
