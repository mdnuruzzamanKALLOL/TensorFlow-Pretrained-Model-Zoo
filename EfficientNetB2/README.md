# EfficientNet-B2 — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** EfficientNet B2 TensorFlow pretrained 9.1M 260px 80.1% ImageNet Keras transfer learning MBConv compound scaling classification

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

EfficientNet-B2 is the first EfficientNet variant to cross 80% ImageNet top-1 accuracy at just 9.1 M parameters with 260² input. It delivers state-of-the-art accuracy-per-parameter efficiency and serves as a popular trade-off point between lightweight B0/B1 and the heavier B3–B7 models.

---

## Model Specifications

| Property | Value |
|----------|-------|
| **Parameters** | 9.1 M |
| **Input Resolution** | 260×260 |
| **ImageNet Top-1** | 80.1% |
| **ImageNet Top-5** | 94.9% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.EfficientNetB2` |
| **Year** | 2019 |
| **Venue** | ICML 2019 |

---

## Architecture Highlights

- 260×260 input resolution for enhanced spatial feature representation
- Dropout rate 0.3 — stronger regularization than B1
- Crosses the 80% top-1 threshold with only 9.1 M parameters
- MBConv blocks with SE channels: more robust feature selection
- Often outperforms ResNet-50 at 3× fewer parameters

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

## When to Use EfficientNet-B2

Use B2 as the first 80%+ accuracy EfficientNet for tasks where crossing the 80% barrier is required but resources remain constrained. Slightly heavier than B1 but significantly more accurate — a common production sweet spot.

---

## Real-World Use Cases

- Medical image classification crossing 80% accuracy threshold
- Drone aerial scene understanding
- Consumer app image recognition on modern smartphone hardware
- Remote sensing binary classification (building detection, road extraction)

---

## Folder Structure

```
EfficientNetB2/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.EfficientNetB2(
    include_top=True,
    weights="imagenet",
    input_shape=(260, 260, 3),
    classes=1000,
)
model.summary()
```

---

## Transfer Learning

```python
import tensorflow as tf

NUM_CLASSES = 10  # replace with your number of classes

base = tf.keras.applications.EfficientNetB2(
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


---

<div align="center">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TensorFlow-Pretrained-Model-Zoo.EfficientNetB2&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
