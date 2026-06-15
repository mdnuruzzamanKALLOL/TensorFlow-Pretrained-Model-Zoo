# ResNet152V2 — Identity Mappings in Deep Residual Networks (TensorFlow / Keras)

**Paper:** Identity Mappings in Deep Residual Networks
**Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
**Conference:** ECCV 2016

---

## Overview

ResNet152V2 is the deepest standard ResNetV2 variant, with stage depths of **3-8-36-3**.
It applies the pre-activation redesign (BN+ReLU before each convolution) to ResNet152,
enabling pure identity shortcuts that improve gradient flow. This yields **~78.0% Top-1**
on ImageNet — the highest accuracy in the standard ResNet family.

---

## ResNetV2 Family — Stage Depth Comparison

| Model | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Params | Top-1 |
|-------|---------|---------|---------|---------|--------|-------|
| ResNet50V2 | 3 | 4 | 6 | 3 | ~25.6M | ~75.6% |
| ResNet101V2 | 3 | 4 | 23 | 3 | ~44.7M | ~77.2% |
| **ResNet152V2** | 3 | **8** | **36** | 3 | ~60.4M | ~78.0% |

---

## Architecture

```
Input (224×224×3)
│
├── Stem : Conv7×7/2  [NO BN/ReLU] + MaxPool3×3/2        →   64 × 56×56
│
├── Stage 1 (conv2): 3  × PreActBottleneck(64)   s=1      →  256 × 56×56
├── Stage 2 (conv3): 8  × PreActBottleneck(128)  s=2      →  512 × 28×28
├── Stage 3 (conv4): 36 × PreActBottleneck(256)  s=2      → 1024 × 14×14  ← deepest
├── Stage 4 (conv5): 3  × PreActBottleneck(512)  s=2      → 2048 ×  7×7
│
├── post_bn  (BatchNorm)
├── post_relu (ReLU)
│
└── GlobalAvgPool → Dense(num_classes, softmax)
```

---

## Key Stats

| Property | Value |
|----------|-------|
| Parameters | ~60.4M |
| Top-1 (ImageNet) | ~78.0% |
| Top-5 (ImageNet) | ~94.0% |
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
from tensorflow.keras.applications.resnet_v2 import preprocess_input

base_model = keras.applications.ResNet152V2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3),
)
x       = layers.GlobalAveragePooling2D()(base_model.output)  # 2048-dim
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model   = keras.Model(inputs=base_model.input, outputs=outputs)

# V2 preprocessing: x/127.5 - 1.0  ->  [-1, 1]
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

# Phase 2 — unfreeze conv4 (36 blocks) + conv5 + post_bn + post_relu
base_model.trainable = True
for layer in base_model.layers:
    layer.trainable = (
        layer.name.startswith('conv4') or
        layer.name.startswith('conv5') or
        layer.name in ('post_bn', 'post_relu')
    )
model.compile(optimizer=keras.optimizers.Adam(1e-5), ...)
model.fit(train_gen, initial_epoch=10, epochs=30, ...)
```

---

## Preprocessing Comparison (full ResNet family)

| Model | Mode | Operation | Range |
|-------|------|-----------|-------|
| ResNet50 / 101 / 152 (V1) | caffe | subtract BGR mean [103.939, 116.779, 123.68] | ~[-128, +128] |
| ResNet50V2 / 101V2 / **152V2** | tf | `x / 127.5 - 1.0` | [-1, 1] |

---

## Folder Structure

```
ResNet152V2/
├── README.md
├── NoteBook/
│   └── resnet152v2.ipynb          — 17-cell notebook (arch + train + ROC AUC)
├── Python Scripts/
│   ├── resnet152v2.py             — build_resnet152v2() from scratch
│   ├── train.py                   — Adam + ReduceLROnPlateau, batch=16
│   ├── inference.py               — top-K single-image prediction
│   └── How to run.txt
└── Using Weight File/
    ├── load_pretrained.py         — load Keras Applications ResNet152V2
    ├── feature_extraction.py      — frozen backbone, GAP + Dense head
    ├── fine_tuning.py             — two-phase (conv4 x36 + conv5 + post_bn/relu)
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
