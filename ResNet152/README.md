# ResNet-152 — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** ResNet-152 TensorFlow pretrained 60.2M 76.6% ImageNet deepest ResNet Keras transfer learning residual ensemble classification 2016

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

ResNet-152 is the deepest standard ResNet at 60.2 M parameters, with stage configuration [3, 8, 36, 3]. It achieves 76.6% ImageNet top-1 — only marginally above ResNet-101 (76.4%) — suggesting diminishing returns from depth alone in the V1 formulation. It is primarily used for ensembling and historical benchmarking.

---

## Model Specifications

<div align="center">

| Property | Value |
|----------|-------|
| **Parameters** | 60.2 M |
| **Input Resolution** | 224×224 |
| **ImageNet Top-1** | 76.6% |
| **ImageNet Top-5** | 93.1% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.ResNet152` |
| **Year** | 2016 |
| **Venue** | CVPR 2016 |

</div>

---

## Architecture Highlights

- Stage-3 has 36 bottleneck blocks — the deepest stage in any standard ResNet
- [3, 8, 36, 3] block configuration across 4 stages
- 60.2 M parameters — largest standard ResNet-V1 variant
- Demonstrates depth saturation in post-activation residual design
- Strong ensemble diversity member due to unique depth distribution

---

## ImageNet Performance — ResNet Family

<div align="center">

| Variant | Params | Input | Top-1 | Top-5 |
|---------|:------:|:-----:|:-----:|:-----:|
| ResNet50 | 25.6 M | 224² | 74.9% | 92.1% |
| ResNet50V2 | 25.6 M | 224² | 75.6% | 92.8% |
| ResNet101 | 44.7 M | 224² | 76.4% | 92.8% |
| ResNet101V2 | 44.7 M | 224² | 77.2% | 93.8% |
| ResNet152 | 60.2 M | 224² | 76.6% | 93.1% |
| ResNet152V2 | 60.2 M | 224² | 78.0% | 94.2% |

</div>

---

## When to Use ResNet-152

Use ResNet-152 only when you need the deepest V1 ResNet for ensemble diversity or exact baseline reproduction. For new work, ResNet-152V2 (+1.4% top-1) or EfficientNet/ConvNeXt are far more efficient choices.

---

## Real-World Use Cases

- Ensemble member for diversity in large-scale classification systems
- Historical baseline reproducing 2016-era ILSVRC results
- Feature extraction for long-tail recognition requiring the richest ResNet features

---

## Folder Structure

```
ResNet152/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.ResNet152(
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

base = tf.keras.applications.ResNet152(
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
@inproceedings{he2016deep,
  title={Deep Residual Learning for Image Recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={CVPR},
  pages={770--778},
  year={2016}
}
```

**Paper:** [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
**Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
**Venue:** CVPR 2016

---

<div align="center">
<sub>Part of the <a href="../README.md">TensorFlow Pretrained Model Zoo</a> — 38 models, 10 families, ready-to-run notebooks and scripts</sub>
</div>

---

<div align="center">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TensorFlow-Pretrained-Model-Zoo.ResNet152&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
