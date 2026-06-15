# DenseNet-201 — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** DenseNet-201 TensorFlow pretrained 20M 77.3% ImageNet Keras transfer learning dense connections medical imaging classification fine-tuning

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

DenseNet-201 is the largest standard DenseNet variant at 20 M parameters with 201 layers total. Its final dense block contains 48 layers, providing extraordinarily rich feature reuse. It achieves 77.3% ImageNet top-1 — competitive with ResNet-50 (74.9%) and ResNet-101 (76.4%) while using fewer parameters.

---

## Model Specifications

| Property | Value |
|----------|-------|
| **Parameters** | 20 M |
| **Input Resolution** | 224×224 |
| **ImageNet Top-1** | 77.3% |
| **ImageNet Top-5** | 93.6% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.DenseNet201` |
| **Year** | 2017 |
| **Venue** | CVPR 2017 |

---

## Architecture Highlights

- Dense block configuration [6, 12, 48, 32] — deepest variant with 201 total layers
- 48-layer final dense block captures the richest hierarchical features
- Maintains parameter efficiency: 20 M vs ResNet-101's 44.7 M for similar accuracy
- Excellent regularization effect from dense connectivity (implicit ensemble)
- Optimal for transfer learning to high-resolution domain-specific datasets

---

## ImageNet Performance — DenseNet Family

| Variant | Params | Input | Top-1 | Top-5 |
|---------|:------:|:-----:|:-----:|:-----:|
| DenseNet121 | 8 M | 224² | 75.0% | 92.2% |
| DenseNet169 | 14 M | 224² | 76.2% | 93.2% |
| DenseNet201 | 20 M | 224² | 77.3% | 93.6% |

---

## When to Use DenseNet-201

Use DenseNet-201 as the strongest DenseNet baseline when accuracy must maximize within the DenseNet architecture. For most tasks, it outperforms DenseNet-121 and DenseNet-169; use it when training time is not a bottleneck.

---

## Real-World Use Cases

- Kaggle competition baselines for medical and scientific imaging
- COVID-19 / pneumonia detection from chest CT and X-ray images
- Satellite scene classification requiring rich multi-scale features
- Transfer learning anchor when dataset size is 10 k–100 k images
- Ensemble member with EfficientNet or ResNet variants

---

## Folder Structure

```
DenseNet201/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.DenseNet201(
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

base = tf.keras.applications.DenseNet201(
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
@inproceedings{huang2017densely,
  title={Densely Connected Convolutional Networks},
  author={Huang, Gao and Liu, Zhuang and Van Der Maaten, Laurens and Weinberger, Kilian Q},
  booktitle={CVPR},
  pages={4700--4708},
  year={2017}
}
```

**Paper:** [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
**Authors:** Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
**Venue:** CVPR 2017

---

<div align="center">
<sub>Part of the <a href="../README.md">TensorFlow Pretrained Model Zoo</a> — 38 models, 10 families, ready-to-run notebooks and scripts</sub>
</div>


---

<div align="center">

![Profile Views](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FmdnuruzzamanKALLOL%2FTensorFlow-Pretrained-Model-Zoo%2Ftree%2Fmaster%2FDenseNet201&count_bg=%23FF6F00&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Profile%20Views&edge_flat=false)

</div>
