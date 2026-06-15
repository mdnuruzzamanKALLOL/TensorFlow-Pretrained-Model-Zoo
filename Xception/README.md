# Xception — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** Xception TensorFlow pretrained 22.9M 299px 79% ImageNet depthwise separable Keras Chollet inception extreme transfer learning classification 2017

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

Xception (Extreme Inception) replaces all Inception modules with depthwise separable convolutions, achieving 79.0% ImageNet top-1 at 22.9 M parameters — surpassing Inception V3 (77.9%) with fewer parameters. Created by François Chollet (Keras author), it demonstrates that cross-channel and spatial correlations can be fully decoupled.

---

## Model Specifications

| Property | Value |
|----------|-------|
| **Parameters** | 22.9 M |
| **Input Resolution** | 299×299 |
| **ImageNet Top-1** | 79.0% |
| **ImageNet Top-5** | 94.5% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.Xception` |
| **Year** | 2017 |
| **Venue** | CVPR 2017 |

---

## Architecture Highlights

- All inception modules replaced with depthwise separable convolutions
- Extreme hypothesis: cross-channel and spatial correlations fully decoupled
- Residual connections between depthwise separable conv blocks
- Entry/Middle/Exit flow architecture with 36 convolutional layers total
- 22.9 M parameters outperforming Inception V3 (23.8 M) by 1.1% top-1

---

## ImageNet Performance — Xception Family

| Model | Params | Input | Top-1 | Top-5 |
|-------|:------:|:-----:|:-----:|:-----:|
| Xception | 22.9 M | 299² | 79.0% | 94.5% |

---

## When to Use Xception

Use Xception when working in the 299² input space and you want to outperform Inception V3 with fewer parameters. It is the best performing single 299² model in the TF Keras zoo. For new projects, EfficientNetV2-S at 384² is more accurate.

---

## Real-World Use Cases

- Texture-rich domain classification (art, fabric, satellite imagery)
- Transfer learning onto datasets where depthwise features generalize well
- Lightweight alternative to Inception-ResNetV2 at similar resolution (299²)
- Research studying depthwise separable convolution generalization
- Image quality assessment (IQA) and perceptual metric computation

---

## Folder Structure

```
Xception/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.Xception(
    include_top=True,
    weights="imagenet",
    input_shape=(299, 299, 3),
    classes=1000,
)
model.summary()
```

---

## Transfer Learning

```python
import tensorflow as tf

NUM_CLASSES = 10  # replace with your number of classes

base = tf.keras.applications.Xception(
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
@inproceedings{chollet2017xception,
  title={Xception: Deep Learning with Depthwise Separable Convolutions},
  author={Chollet, Fran\c{c}ois},
  booktitle={CVPR},
  pages={1251--1258},
  year={2017}
}
```

**Paper:** [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
**Authors:** François Chollet
**Venue:** CVPR 2017

---

<div align="center">
<sub>Part of the <a href="../README.md">TensorFlow Pretrained Model Zoo</a> — 38 models, 10 families, ready-to-run notebooks and scripts</sub>
</div>
