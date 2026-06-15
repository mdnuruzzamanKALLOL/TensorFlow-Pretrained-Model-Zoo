# EfficientNet-B5 — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** EfficientNet B5 TensorFlow pretrained 30M 456px 83.6% ImageNet Keras high-resolution transfer learning pathology aerial classification

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

EfficientNet-B5 achieves 83.6% ImageNet top-1 at 456² resolution with 30 M parameters. Its high-resolution input makes it the best EfficientNet variant for tasks involving small objects or fine spatial structure. Training requires 24+ GB VRAM with standard batch sizes, making it the boundary between accessible and resource-intensive EfficientNets.

---

## Model Specifications

| Property | Value |
|----------|-------|
| **Parameters** | 30 M |
| **Input Resolution** | 456×456 |
| **ImageNet Top-1** | 83.6% |
| **ImageNet Top-5** | 96.7% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.EfficientNetB5` |
| **Year** | 2019 |
| **Venue** | ICML 2019 |

---

## Architecture Highlights

- 456×456 input — captures micro-level texture and small-object detail
- 30 M parameters provide ample capacity without reaching B6/B7 memory overhead
- Strong baseline for object detection with high-resolution input requirements
- Dropout rate 0.4 with label smoothing in pretrained training recipe

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

## When to Use EfficientNet-B5

Use B5 when 456² input resolution provides meaningful accuracy gains over B4 for your domain. Ideal for tasks with small discriminative regions. Requires 24 GB VRAM for fine-tuning — consider gradient checkpointing on smaller GPUs.

---

## Real-World Use Cases

- Pathology whole-slide image patch classification at high magnification
- Aerial object detection where input resolution is critical
- Remote sensing: tree crown detection, vehicle counting from drone images
- Astronomy image classification with fine spatial features

---

## Folder Structure

```
EfficientNetB5/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.EfficientNetB5(
    include_top=True,
    weights="imagenet",
    input_shape=(456, 456, 3),
    classes=1000,
)
model.summary()
```

---

## Transfer Learning

```python
import tensorflow as tf

NUM_CLASSES = 10  # replace with your number of classes

base = tf.keras.applications.EfficientNetB5(
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

![Profile Views](https://komarev.com/ghpvc/?username=mdnuruzzamanKALLOL&label=Profile%20Views&color=FF6F00&style=flat-square)

</div>
