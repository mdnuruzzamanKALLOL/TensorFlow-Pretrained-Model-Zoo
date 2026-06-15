# Inception V3 — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** Inception V3 TensorFlow pretrained 23.8M 299px 77.9% ImageNet Keras medical imaging multi-scale transfer learning classification 2016

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

Inception V3 modernizes the original GoogLeNet by introducing factorized convolutions (7×7 → two 1×7 + 7×1), grid reduction modules, label smoothing, and auxiliary classifiers. With 23.8 M parameters at 299² input, it achieves 77.9% ImageNet top-1 and remains a widely-used backbone in medical imaging and multi-scale feature extraction tasks.

---

## Model Specifications

| Property | Value |
|----------|-------|
| **Parameters** | 23.8 M |
| **Input Resolution** | 299×299 |
| **ImageNet Top-1** | 77.9% |
| **ImageNet Top-5** | 93.7% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.InceptionV3` |
| **Year** | 2016 |
| **Venue** | CVPR 2016 |

---

## Architecture Highlights

- Factorized convolutions: n×n replaced by 1×n + n×1 for computation savings
- Inception modules capture multi-scale features with parallel 1×1, 3×3, 5×5 branches
- Auxiliary classifiers at intermediate layers for gradient regularization
- Label smoothing (ε=0.1) in training — later adopted universally
- Grid reduction modules avoid representational bottlenecks during downsampling

---

## ImageNet Performance — Inception Family

| Model | Params | Input | Top-1 | Top-5 |
|-------|:------:|:-----:|:-----:|:-----:|
| InceptionV3 | 23.8 M | 299² | 77.9% | 93.7% |
| InceptionResNetV2 | 55.8 M | 299² | 80.1% | 95.3% |

---

## When to Use Inception V3

Use Inception V3 when domain literature uses it as the reference architecture (medical imaging papers especially). Its multi-scale inception modules often transfer better than ResNets on texture-rich domains like histopathology.

---

## Real-World Use Cases

- Medical imaging: X-ray, MRI, CT scan classification (heavily cited benchmark)
- Multi-scale feature extraction where inception modules excel
- Dermatology: skin lesion classification (Stanford dermatologist-level AI paper used Inception V3)
- Diabetic retinopathy screening (Google Health paper used Inception V3)
- Satellite image classification from multi-spectral sensors

---

## Folder Structure

```
InceptionV3/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.InceptionV3(
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

base = tf.keras.applications.InceptionV3(
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
@inproceedings{szegedy2016rethinking,
  title={Rethinking the Inception Architecture for Computer Vision},
  author={Szegedy, Christian and Vanhoucke, Vincent and Ioffe, Sergey and Shlens, Jon and Wojna, Zbigniew},
  booktitle={CVPR},
  pages={2818--2826},
  year={2016}
}
```

**Paper:** [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)
**Authors:** Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathan Shlens, Zbigniew Wojna
**Venue:** CVPR 2016

---

<div align="center">
<sub>Part of the <a href="../README.md">TensorFlow Pretrained Model Zoo</a> — 38 models, 10 families, ready-to-run notebooks and scripts</sub>
</div>


---

<div align="center">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TensorFlow-Pretrained-Model-Zoo.InceptionV3&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
