# ResNet101V2 — Identity Mappings in Deep Residual Networks (TensorFlow / Keras)

**Paper:** Identity Mappings in Deep Residual Networks
**Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
**Conference:** ECCV 2016

---

## Overview

ResNet101V2 is the V2 variant of ResNet101, redesigned so that the shortcut
connection carries a pure identity signal. The key change is **pre-activation**:
Batch Normalization and ReLU are applied *before* each convolution rather than
after the addition, creating a clean gradient highway through skip connections.

ResNet101V2 has the same stage depths (3-4-**23**-3) as ResNet101 but achieves
higher Top-1 accuracy (~77.2% vs ~76.4%) due to the improved gradient flow.

---

## ResNetV2 Family — Stage Depth Comparison

| Model | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Params | Top-1 |
|-------|---------|---------|---------|---------|--------|-------|
| ResNet50V2 | 3 | 4 | **6** | 3 | ~25.6M | ~75.6% |
| **ResNet101V2** | 3 | 4 | **23** | 3 | ~44.7M | ~77.2% |
| ResNet152V2 | 3 | 8 | 36 | 3 | ~60.4M | ~78.0% |

---

## Pre-activation Bottleneck Block (V2 vs V1)

```
V1 (ResNet101)                    V2 (ResNet101V2)
──────────────────────────        ──────────────────────────
Input                             Input
  |                                 |
  +─── shortcut ───────+            +─── shortcut ────────+
  |   (proj if needed) |            |   (identity / proj) |
  Conv1x1 -> BN -> ReLU            BN -> ReLU -> Conv1x1
  Conv3x3 -> BN -> ReLU            BN -> ReLU -> Conv3x3
  Conv1x1 -> BN                    BN -> ReLU -> Conv1x1
  |                                 |   (NO BN/ReLU here)
  Add ─────────────────+            Add ─────────────────+
  ReLU                              (output, no activation)
```

In V2, the shortcut path is a **true identity**: no activation, no scaling.
This allows gradients to flow unmodified from any layer back to the input.

---

## Architecture

```
Input (224×224×3)
│
├── Stem : Conv7×7/2  [NO BN/ReLU] + MaxPool3×3/2       →   64 × 56×56
│
├── Stage 1 (conv2): 3  × PreActBottleneck(64)   s=1     →  256 × 56×56
├── Stage 2 (conv3): 4  × PreActBottleneck(128)  s=2     →  512 × 28×28
├── Stage 3 (conv4): 23 × PreActBottleneck(256)  s=2     → 1024 × 14×14  ← deep
├── Stage 4 (conv5): 3  × PreActBottleneck(512)  s=2     → 2048 ×  7×7
│
├── post_bn  (BatchNorm)      ← required because last block has no post-activation
├── post_relu (ReLU)
│
└── GlobalAvgPool → Dense(num_classes, softmax)
```

---

## Key Stats

| Property | Value |
|----------|-------|
| Parameters | ~44.7M |
| Top-1 (ImageNet) | ~77.2% |
| Top-5 (ImageNet) | ~93.8% |
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

base_model = keras.applications.ResNet101V2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3),
)
x       = layers.GlobalAveragePooling2D()(base_model.output)  # 2048-dim
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model   = keras.Model(inputs=base_model.input, outputs=outputs)

# V2 preprocessing: x/127.5 - 1.0  ->  [-1, 1]   (NOT BGR mean subtraction)
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

# Phase 2 — unfreeze conv4 (23 blocks) + conv5 + post_bn + post_relu
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

**Why unfreeze `post_bn` and `post_relu`?**
The final block in `conv5` ends with a raw convolution (no BN/ReLU), so
`post_bn` and `post_relu` are the activation gate for that stage's output.
If `conv5` weights change but `post_bn` stays frozen, its running statistics
become mismatched, degrading performance. Always unfreeze them together.

---

## Preprocessing Comparison

| Model | Mode | Operation | Range |
|-------|------|-----------|-------|
| ResNet101 (V1) | caffe | subtract BGR mean [103.939, 116.779, 123.68] | ~[-128, +128] |
| **ResNet101V2** | tf | `x / 127.5 - 1.0` | [-1, 1] |

---

## Folder Structure

```
ResNet101V2/
├── README.md
├── NoteBook/
│   └── resnet101v2.ipynb          — 17-cell notebook (arch + train + ROC AUC)
├── Python Scripts/
│   ├── resnet101v2.py             — build_resnet101v2() from scratch
│   ├── train.py                   — Adam + ReduceLROnPlateau, batch=16
│   ├── inference.py               — top-K single-image prediction
│   └── How to run.txt
└── Using Weight File/
    ├── load_pretrained.py         — load Keras Applications ResNet101V2
    ├── feature_extraction.py      — frozen backbone, GAP + Dense head
    ├── fine_tuning.py             — two-phase (conv4 x23 + conv5 + post_bn/relu)
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
