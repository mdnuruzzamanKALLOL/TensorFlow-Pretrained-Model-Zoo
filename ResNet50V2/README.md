# ResNet-50 V2 — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** ResNet-50V2 TensorFlow pretrained 25.6M 75.6% ImageNet pre-activation Keras transfer learning residual ECCV 2016 classification

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

ResNet-50 V2 reorders operations to pre-activation (BN→ReLU→Conv instead of Conv→BN→ReLU), creating cleaner identity paths that improve gradient flow and generalization. It achieves 75.6% ImageNet top-1 — 0.7% better than ResNet-50 with identical parameters — and is generally preferred when starting from ImageNet weights for transfer learning.

---

## Model Specifications

<div align="center">

| Property | Value |
|----------|-------|
| **Parameters** | 25.6 M |
| **Input Resolution** | 224×224 |
| **ImageNet Top-1** | 75.6% |
| **ImageNet Top-5** | 92.8% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.ResNet50V2` |
| **Year** | 2016 |
| **Venue** | ECCV 2016 |

</div>

---

## Architecture Highlights

- Pre-activation order: BatchNorm → ReLU → Conv (vs post-activation in V1)
- Full pre-activation residual unit enables true identity shortcuts
- Improved gradient flow and regularization from pre-activation BatchNorm
- 0.7% top-1 improvement over ResNet-50 with same architecture and parameters
- Better generalization on small downstream datasets due to cleaner gradient paths

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

## When to Use ResNet-50 V2

Prefer ResNet-50V2 over ResNet-50 for transfer learning — same parameters, 0.7% better top-1, cleaner identity paths. Use ResNet-50 only when V1 architecture is specifically required for reproducibility or framework compatibility.

---

## Real-World Use Cases

- Transfer learning where V2's improved generalization matters
- Fine-tuning on domain-specific datasets with limited data
- Backbone for detection heads when marginal accuracy improvement is valued
- Research requiring controlled comparison between V1 and V2 residual formulations

---

## Folder Structure

```
ResNet50V2/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.ResNet50V2(
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

base = tf.keras.applications.ResNet50V2(
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
@inproceedings{he2016identity,
  title={Identity Mappings in Deep Residual Networks},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={ECCV},
  pages={630--645},
  year={2016}
}
```

**Paper:** [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
**Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
**Venue:** ECCV 2016

---

<div align="center">
<sub>Part of the <a href="../README.md">TensorFlow Pretrained Model Zoo</a> — 38 models, 10 families, ready-to-run notebooks and scripts</sub>
</div>

---

<div align="center">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TensorFlow-Pretrained-Model-Zoo.ResNet50V2&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
