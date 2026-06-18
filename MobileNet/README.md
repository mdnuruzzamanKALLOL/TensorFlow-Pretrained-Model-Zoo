# MobileNet V1 — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** MobileNet V1 TensorFlow pretrained 4.2M 70.4% ImageNet mobile edge IoT Keras depthwise separable TFLite real-time inference classification 2017

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

MobileNet V1 introduced depthwise separable convolutions to dramatically reduce the computational cost of CNNs for mobile and embedded vision. With only 4.2 M parameters and 569 M multiply-adds, it achieves 70.4% ImageNet top-1 — enabling real-time inference on smartphones and IoT devices for the first time.

---

## Model Specifications

<div align="center">

| Property | Value |
|----------|-------|
| **Parameters** | 4.2 M |
| **Input Resolution** | 224×224 |
| **ImageNet Top-1** | 70.4% |
| **ImageNet Top-5** | 89.5% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.MobileNet` |
| **Year** | 2017 |
| **Venue** | arXiv 2017 |

</div>

---

## Architecture Highlights

- Depthwise separable convolutions: factorizes standard conv into depthwise + pointwise
- Width multiplier α ∈ {0.25, 0.5, 0.75, 1.0} to trade accuracy for speed
- Resolution multiplier ρ to reduce input resolution at inference
- 8–9× fewer multiply-adds than VGG-16 with only 1% lower accuracy
- Pioneered the mobile AI era — basis for MobileNetV2, V3, EfficientNet

---

## ImageNet Performance — MobileNet Family

<div align="center">

| Model | Params | Input | Top-1 | Top-5 |
|-------|:------:|:-----:|:-----:|:-----:|
| MobileNet | 4.2 M | 224² | 70.4% | 89.5% |
| MobileNetV2 | 3.4 M | 224² | 71.3% | 90.1% |

</div>

---

## When to Use MobileNet V1

Use MobileNet V1 when targeting very constrained hardware (< 2 GB RAM, no GPU) or when reproducing older mobile AI literature. For new projects, MobileNetV2 or EfficientNet-B0 provide better accuracy at similar or lower compute.

---

## Real-World Use Cases

- On-device inference on Android and iOS (< 5 ms latency target)
- IoT and embedded systems: Raspberry Pi, Arduino with Edge AI
- Real-time object classification in robotics pipelines
- TFLite deployment for offline-first mobile applications
- Heritage architecture for understanding mobile CNN design evolution

---

## Folder Structure

```
MobileNet/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.MobileNet(
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

base = tf.keras.applications.MobileNet(
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
@article{howard2017mobilenets,
  title={{MobileNets}: Efficient Convolutional Neural Networks for Mobile Vision Applications},
  author={Howard, Andrew G and Zhu, Menglong and Chen, Bo and Kalenichenko, Dmitry and Wang, Weijun and Weyand, Tobias and Andreetto, Marco and Adam, Hartwig},
  journal={arXiv preprint arXiv:1704.04861},
  year={2017}
}
```

**Paper:** [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
**Authors:** Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
**Venue:** arXiv 2017

---

<div align="center">
<sub>Part of the <a href="../README.md">TensorFlow Pretrained Model Zoo</a> — 38 models, 10 families, ready-to-run notebooks and scripts</sub>
</div>

---

<div align="center">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TensorFlow-Pretrained-Model-Zoo.MobileNet&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
