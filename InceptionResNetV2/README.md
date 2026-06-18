# Inception-ResNet V2 — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** Inception-ResNet V2 TensorFlow pretrained 55.8M 80.1% ImageNet Keras residual inception multi-scale medical imaging classification 2017

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

Inception-ResNet V2 combines the multi-scale feature extraction of Inception modules with the identity shortcut connections of ResNets, achieving 80.1% ImageNet top-1 at 55.8 M parameters. It is the highest-accuracy Inception-family model and excels at tasks requiring both multi-scale and deep residual features.

---

## Model Specifications

<div align="center">

| Property | Value |
|----------|-------|
| **Parameters** | 55.8 M |
| **Input Resolution** | 299×299 |
| **ImageNet Top-1** | 80.1% |
| **ImageNet Top-5** | 95.3% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.InceptionResNetV2` |
| **Year** | 2017 |
| **Venue** | AAAI 2017 |

</div>

---

## Architecture Highlights

- Inception-ResNet-A/B/C blocks: inception branches merged with residual addition
- Residual scaling (scale ≈ 0.1–0.3) prevents training instability at depth
- Stem block with factorized convolutions before inception-residual stages
- 299×299 input preserves fine-grained spatial information through deep network
- Reduction-A/B modules for spatial downsampling without bottlenecks

---

## ImageNet Performance — Inception Family

<div align="center">

| Model | Params | Input | Top-1 | Top-5 |
|-------|:------:|:-----:|:-----:|:-----:|
| InceptionV3 | 23.8 M | 299² | 77.9% | 93.7% |
| InceptionResNetV2 | 55.8 M | 299² | 80.1% | 95.3% |

</div>

---

## When to Use Inception-ResNet V2

Use Inception-ResNet V2 when the target domain benefits from both multi-scale inception features AND deep residual connectivity. 55.8 M parameters require ~16 GB VRAM for fine-tuning. Consider EfficientNetV2-S for similar accuracy with faster training.

---

## Real-World Use Cases

- High-accuracy medical imaging classification (CT, MRI, fundus images)
- Fine-grained recognition where both multi-scale and deep features matter
- Remote sensing multi-label classification
- Ensemble member alongside EfficientNet or ConvNeXt for diversity
- Transfer learning onto large proprietary medical imaging datasets

---

## Folder Structure

```
InceptionResNetV2/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.InceptionResNetV2(
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

base = tf.keras.applications.InceptionResNetV2(
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
@inproceedings{szegedy2017inception,
  title={Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning},
  author={Szegedy, Christian and Ioffe, Sergey and Vanhoucke, Vincent and Alemi, Alexander},
  booktitle={AAAI},
  volume={31},
  year={2017}
}
```

**Paper:** [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)
**Authors:** Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alexander Alemi
**Venue:** AAAI 2017

---

<div align="center">
<sub>Part of the <a href="../README.md">TensorFlow Pretrained Model Zoo</a> — 38 models, 10 families, ready-to-run notebooks and scripts</sub>
</div>

---

<div align="center">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TensorFlow-Pretrained-Model-Zoo.InceptionResNetV2&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
