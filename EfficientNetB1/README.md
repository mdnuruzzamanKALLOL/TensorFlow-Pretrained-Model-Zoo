# EfficientNet-B1 — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** EfficientNet B1 TensorFlow pretrained 7.8M 240px 79.1% ImageNet Keras compound scaling transfer learning mobile classification

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

EfficientNet-B1 compounds B0 by scaling depth × 1.1, width × 1.1, resolution to 240². With 7.8 M parameters, it achieves 79.1% ImageNet top-1 — a 2% gain over B0 for only 2.5 M extra parameters. The 240² input improves spatial detail capture over B0's 224², making it better for tasks with small-object discrimination.

---

## Model Specifications

| Property | Value |
|----------|-------|
| **Parameters** | 7.8 M |
| **Input Resolution** | 240×240 |
| **ImageNet Top-1** | 79.1% |
| **ImageNet Top-5** | 94.4% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.EfficientNetB1` |
| **Year** | 2019 |
| **Venue** | ICML 2019 |

---

## Architecture Highlights

- Compound scaling φ=1: uniformly scaled from B0 baseline
- 240×240 resolution input captures finer spatial details than B0
- 7.8 M parameters maintain lightweight profile for constrained environments
- SE attention in every MBConv block at ratio 0.25
- Top choice in Kaggle competitions for accuracy/speed balance

---

## ImageNet Performance — EfficientNet Family

| Variant | Params | Input | Top-1 | Top-5 |
|---------|:------:|:-----:|:-----:|:-----:|
| EfficientNetB0 | 5.3 M | 224² | 77.1% | 93.3% |
| EfficientNetB1 | 7.8 M | 240² | 79.1% | 94.4% |
| EfficientNetB2 | 9.1 M | 260² | 80.1% | 94.9% |
| EfficientNetB3 | 12 M | 300² | 81.6% | 95.7% |
| EfficientNetB4 | 19 M | 380² | 82.9% | 96.4% |
| EfficientNetB5 | 30 M | 456² | 83.6% | 96.7% |
| EfficientNetB6 | 43 M | 528² | 84.0% | 96.9% |
| EfficientNetB7 | 66 M | 600² | 84.3% | 97.0% |

---

## When to Use EfficientNet-B1

Use B1 when B0 doesn't quite meet accuracy requirements but you must stay lightweight. Adds ~50% parameters over B0 for a 2% accuracy gain — often the best trade-off for mobile inference targets with moderate accuracy requirements.

---

## Real-World Use Cases

- Mobile applications where B0's 77.1% accuracy is insufficient
- Food recognition and fine-grained visual classification
- Wildlife monitoring with camera trap images
- Product defect detection on manufacturing lines

---

## Folder Structure

```
EfficientNetB1/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.EfficientNetB1(
    include_top=True,
    weights="imagenet",
    input_shape=(240, 240, 3),
    classes=1000,
)
model.summary()
```

---

## Transfer Learning

```python
import tensorflow as tf

NUM_CLASSES = 10  # replace with your number of classes

base = tf.keras.applications.EfficientNetB1(
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
@inproceedings{tan2019efficientnet,
  title={{EfficientNet}: Rethinking Model Scaling for Convolutional Neural Networks},
  author={Tan, Mingxing and Le, Quoc V},
  booktitle={ICML},
  pages={6105--6114},
  year={2019}
}
```

**Paper:** [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
**Authors:** Mingxing Tan, Quoc V. Le
**Venue:** ICML 2019

---

<div align="center">
<sub>Part of the <a href="../README.md">TensorFlow Pretrained Model Zoo</a> — 38 models, 10 families, ready-to-run notebooks and scripts</sub>
</div>
