# EfficientNet-B3 — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** EfficientNet B3 TensorFlow pretrained 12M 300px 81.6% ImageNet Kaggle Keras compound scaling transfer learning EfficientDet classification

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

EfficientNet-B3 reaches 81.6% ImageNet top-1 with 12 M parameters at 300² input — matching ResNet-152 (76.6%) with 5× fewer parameters. It is the most popular EfficientNet variant in academic research and Kaggle competitions, offering an excellent balance between accuracy, training speed, and GPU memory consumption.

---

## Model Specifications

<div align="center">

| Property | Value |
|----------|-------|
| **Parameters** | 12 M |
| **Input Resolution** | 300×300 |
| **ImageNet Top-1** | 81.6% |
| **ImageNet Top-5** | 95.7% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.EfficientNetB3` |
| **Year** | 2019 |
| **Venue** | ICML 2019 |

</div>

---

## Architecture Highlights

- 300×300 input enhances detection of medium-scale objects and textures
- 12 M parameters — comparable to MobileNetV2 but 10% more accurate on ImageNet
- Dropout 0.3 and aggressive data augmentation in pretrained recipe
- Common backbone for EfficientDet object detection framework
- Strong transfer performance on fine-grained 200-class benchmarks (CUB-200, FGVC)

---

## ImageNet Performance — EfficientNet Family

<div align="center">

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

</div>

---

## When to Use EfficientNet-B3

B3 is the most versatile EfficientNet — use it as the default starting point for classification tasks on A100/V100 GPUs when accuracy > 81% is required without going to very large models. Often best accuracy/speed trade-off in competitions.

---

## Real-World Use Cases

- Kaggle image classification competitions (a community favorite)
- EfficientDet object detection backbone for COCO-style benchmarks
- Fine-grained visual recognition: birds, cars, aircraft species
- Retail product recognition and visual search
- Plant disease and crop health classification from field images

---

## Folder Structure

```
EfficientNetB3/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.EfficientNetB3(
    include_top=True,
    weights="imagenet",
    input_shape=(300, 300, 3),
    classes=1000,
)
model.summary()
```

---

## Transfer Learning

```python
import tensorflow as tf

NUM_CLASSES = 10  # replace with your number of classes

base = tf.keras.applications.EfficientNetB3(
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

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TensorFlow-Pretrained-Model-Zoo.EfficientNetB3&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
