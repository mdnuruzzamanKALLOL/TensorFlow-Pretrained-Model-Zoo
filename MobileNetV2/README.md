# MobileNet V2 — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** MobileNetV2 TensorFlow pretrained 3.4M 71.3% ImageNet mobile edge TFLite Keras inverted residuals linear bottleneck real-time classification 2018

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

MobileNetV2 introduces Inverted Residual blocks with Linear Bottlenecks, achieving 71.3% ImageNet top-1 with only 3.4 M parameters — 20% fewer than MobileNet V1 while being more accurate. Its bottleneck feature maps are preserved without non-linearity, preventing information loss in low-dimensional representations.

---

## Model Specifications

| Property | Value |
|----------|-------|
| **Parameters** | 3.4 M |
| **Input Resolution** | 224×224 |
| **ImageNet Top-1** | 71.3% |
| **ImageNet Top-5** | 90.1% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.MobileNetV2` |
| **Year** | 2018 |
| **Venue** | CVPR 2018 |

---

## Architecture Highlights

- Inverted residuals: expands channels in bottleneck then projects back to narrow output
- Linear bottleneck output layer (no ReLU) preserves manifold structure
- Expansion factor t=6 in each block: channels expand 6× before depthwise conv
- Residual shortcuts between bottleneck layers for gradient flow
- Better accuracy than MobileNetV1 with 20% fewer parameters

---

## ImageNet Performance — MobileNet Family

| Model | Params | Input | Top-1 | Top-5 |
|-------|:------:|:-----:|:-----:|:-----:|
| MobileNet | 4.2 M | 224² | 70.4% | 89.5% |
| MobileNetV2 | 3.4 M | 224² | 71.3% | 90.1% |

---

## When to Use MobileNet V2

MobileNetV2 is the standard recommendation for mobile deployment requiring 70–72% accuracy. More accurate than V1 with fewer parameters. Use EfficientNet-B0 when > 77% accuracy is required with similar latency budget.

---

## Real-World Use Cases

- Production mobile apps on iOS and Android with TFLite quantization
- DeepLab V3+ semantic segmentation backbone for mobile
- SSDLite object detection for real-time mobile inference
- Edge AI on Coral Edge TPU and Jetson Nano
- Transfer learning baseline for agricultural and retail edge apps

---

## Folder Structure

```
MobileNetV2/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.MobileNetV2(
    include_top=True,
    weights="imagenet",
    input_shape=(224, 224, 3),
    classes=1000,
)
model.summary()
```

---

## Transfer Learning

```python
import tensorflow as tf

NUM_CLASSES = 10  # replace with your number of classes

base = tf.keras.applications.MobileNetV2(
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
@inproceedings{sandler2018mobilenetv2,
  title={{MobileNetV2}: Inverted Residuals and Linear Bottlenecks},
  author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
  booktitle={CVPR},
  pages={4510--4520},
  year={2018}
}
```

**Paper:** [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
**Authors:** Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
**Venue:** CVPR 2018

---

<div align="center">
<sub>Part of the <a href="../README.md">TensorFlow Pretrained Model Zoo</a> — 38 models, 10 families, ready-to-run notebooks and scripts</sub>
</div>
