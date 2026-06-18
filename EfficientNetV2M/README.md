# EfficientNetV2-M — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** EfficientNetV2-M TensorFlow pretrained 54M 480px 85.2% ImageNet Keras high-accuracy fast training transfer learning classification 2021

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

EfficientNetV2-M delivers 85.2% ImageNet top-1 at 54.1 M parameters and 480² input. It surpasses EfficientNet-B7 (84.3%) while training 11× faster, representing the efficiency breakthrough of the V2 family. It is widely used as the accuracy anchor for high-performance classification systems.

---

<div align="center">

## Model Specifications

| Property | Value |
|----------|-------|
| **Parameters** | 54.1 M |
| **Input Resolution** | 480×480 |
| **ImageNet Top-1** | 85.2% |
| **ImageNet Top-5** | 97.4% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.EfficientNetV2M` |
| **Year** | 2021 |
| **Venue** | ICML 2021 |

</div>

---

## Architecture Highlights

- 480×480 input with NAS-optimized 6-stage block distribution
- 54.1 M parameters with 85.2% top-1 — better than B7 with 11× faster training
- Fused-MBConv in early stages removes expensive per-channel depthwise operations
- Progressive learning with image size from 300 to 480 during training
- Supports tf.keras mixed precision (float16) out of the box

---

<div align="center">

## ImageNet Performance — EfficientNetV2 Family

| Variant | Params | Input | Top-1 | Top-5 |
|---------|:------:|:-----:|:-----:|:-----:|
| EfficientNetV2B0 | 7.1 M | 224² | 78.7% | 94.3% |
| EfficientNetV2B1 | 8.1 M | 240² | 79.8% | 95.0% |
| EfficientNetV2B2 | 10.1 M | 260² | 80.5% | 95.1% |
| EfficientNetV2B3 | 14.4 M | 300² | 82.0% | 95.8% |
| EfficientNetV2S | 21.5 M | 384² | 83.9% | 96.7% |
| EfficientNetV2M | 54.1 M | 480² | 85.2% | 97.4% |
| EfficientNetV2L | 119 M | 480² | 86.8% | 97.9% |

</div>

---

## When to Use EfficientNetV2-M

Use V2-M when 85%+ accuracy is required and training budget permits 54 M parameters. The efficiency improvement over B7 is dramatic — always prefer V2-M over B7 in new work.

---

## Real-World Use Cases

- High-stakes classification requiring 85%+ accuracy on server GPUs
- Foundation for EfficientDetV2-M object detection
- Multi-class disease classification in clinical AI systems
- Satellite imagery semantic labelling with rich feature maps

---

## Folder Structure

```
EfficientNetV2M/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.EfficientNetV2M(
    include_top=True,
    weights="imagenet",
    input_shape=(480, 480, 3),
    classes=1000,
)
model.summary()
```

---

## Transfer Learning

```python
import tensorflow as tf

NUM_CLASSES = 10  # replace with your number of classes

base = tf.keras.applications.EfficientNetV2M(
    include_top=False,
    weights="imagenet",
    pooling="avg",
)
base.trainable = False  # freeze for feature extraction

x = tf.keras.layers.Dense(256, activation="relu")(base.output)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = tf.keras.Model(base.input, output)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
```

---

## Fine-Tuning (Progressive Unfreeze)

```python
# Step 1: train the head with frozen base (see Transfer Learning above)
model.fit(train_ds, epochs=5, validation_data=val_ds)

# Step 2: unfreeze the base and fine-tune with a lower learning rate
base.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(train_ds, epochs=10, validation_data=val_ds)
```

---

## Citation

```bibtex
@inproceedings{tan2021efficientnetv2,
  title={{EfficientNetV2}: Smaller Models and Faster Training},
  author={Tan, Mingxing and Le, Quoc V},
  booktitle={ICML},
  pages={10096--10106},
  year={2021}
}
```

**Paper:** [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
**Authors:** Mingxing Tan, Quoc V. Le
**Venue:** ICML 2021

---

<div align="center">
<sub>Part of the <a href="../README.md">TensorFlow Pretrained Model Zoo</a> — 38 models, 10 families, ready-to-run notebooks and scripts</sub>
</div>

---

<div align="center">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TensorFlow-Pretrained-Model-Zoo.EfficientNetV2M&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
