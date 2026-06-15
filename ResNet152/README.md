# ResNet152 — Deep Residual Learning (TensorFlow / Keras)

**Paper:** Deep Residual Learning for Image Recognition
**Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
**Conference:** CVPR 2016 (Best Paper Award)

---

## Overview

ResNet152 is the deepest standard ResNet variant, with stage depths of **3-8-36-3**.
Compared to ResNet101 (3-4-23-3), Stage 2 grows from 4 to **8 blocks** and Stage 3
grows from 23 to **36 blocks**, adding ~15.7M parameters for a modest +0.2% Top-1
gain. ResNet152 is useful when squeezing out every bit of accuracy matters and
inference speed is less of a constraint.

---

## ResNet Family — Stage Depth Comparison

| Model | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Params | Top-1 |
|-------|---------|---------|---------|---------|--------|-------|
| ResNet50 | 3 | 4 | 6 | 3 | ~25.6M | ~74.9% |
| ResNet101 | 3 | 4 | 23 | 3 | ~44.5M | ~76.4% |
| **ResNet152** | 3 | **8** | **36** | 3 | ~60.2M | ~76.6% |

---

## Architecture

```
Input (224×224×3)
│
├── Stem  : Conv7×7/2 + BN + ReLU + MaxPool3×3/2     →   64 × 56×56
│
├── Stage 1 (conv2): 3  × Bottleneck(64)   s=1        →  256 × 56×56
├── Stage 2 (conv3): 8  × Bottleneck(128)  s=2        →  512 × 28×28
├── Stage 3 (conv4): 36 × Bottleneck(256)  s=2        → 1024 × 14×14  ← deepest
├── Stage 4 (conv5): 3  × Bottleneck(512)  s=2        → 2048 ×  7×7
│
└── GlobalAvgPool → Dense(num_classes, softmax)
```

---

## Key Stats

| Property | Value |
|----------|-------|
| Parameters | ~60.2M |
| Top-1 (ImageNet) | ~76.6% |
| Top-5 (ImageNet) | ~93.1% |
| Input size | 224×224 |
| Framework | TensorFlow / Keras |

---

## Training Configuration (From Scratch)

| Setting | Value |
|---------|-------|
| Input size | 224×224 |
| Batch size | 16 |
| Optimizer | Adam (lr=1e-3) |
| Scheduler | ReduceLROnPlateau (factor=0.1, patience=5) |
| Loss | categorical_crossentropy |
| Epochs | 30 |

---

## Transfer Learning

```python
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet import preprocess_input

base_model = keras.applications.ResNet152(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3),
)
x       = layers.GlobalAveragePooling2D()(base_model.output)  # 2048-dim
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model   = keras.Model(inputs=base_model.input, outputs=outputs)

# Caffe-mode: subtract BGR mean (same as ResNet50 / ResNet101)
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

# Phase 2 — unfreeze conv4 (36 blocks) + conv5
base_model.trainable = True
for layer in base_model.layers:
    layer.trainable = (
        layer.name.startswith('conv4') or
        layer.name.startswith('conv5')
    )
model.compile(optimizer=keras.optimizers.Adam(1e-5), ...)
model.fit(train_gen, initial_epoch=10, epochs=30, ...)
```

---

## Folder Structure

```
ResNet152/
├── README.md
├── NoteBook/
│   └── resnet152.ipynb          — 17-cell notebook (arch + train + ROC AUC)
├── Python Scripts/
│   ├── resnet152.py             — build_resnet152() from scratch
│   ├── train.py                 — Adam + ReduceLROnPlateau, batch=16
│   ├── inference.py             — top-K single-image prediction
│   └── How to run.txt
└── Using Weight File/
    ├── load_pretrained.py       — load Keras Applications ResNet152
    ├── feature_extraction.py    — frozen backbone, GAP + Dense head
    ├── fine_tuning.py           — two-phase (conv4 x36 + conv5 unfreeze)
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
