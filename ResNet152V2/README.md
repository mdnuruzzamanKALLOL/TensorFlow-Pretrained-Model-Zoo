# ResNet-152 V2 — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** ResNet-152V2 TensorFlow pretrained 60.2M 78% ImageNet pre-activation ECCV Keras deepest residual transfer learning classification

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

ResNet-152 V2 applies pre-activation identity mappings to the 152-layer architecture, achieving 78.0% ImageNet top-1 — a 1.4% gain over ResNet-152 V1. This is the largest improvement from V1→V2 across the ResNet family, demonstrating that pre-activation benefits are amplified in deeper networks. It is the strongest V1/V2 standard ResNet.

---

## Model Specifications

| Property | Value |
|----------|-------|
| **Parameters** | 60.2 M |
| **Input Resolution** | 224×224 |
| **ImageNet Top-1** | 78.0% |
| **ImageNet Top-5** | 94.2% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.ResNet152V2` |
| **Year** | 2016 |
| **Venue** | ECCV 2016 |

---

## Architecture Highlights

- Pre-activation at all 152 layers amplifies gradient flow benefits over shallower V2s
- 1.4% top-1 improvement over ResNet-152 — largest V1→V2 gain in the family
- 36 pre-activation blocks in stage 3 for the richest deep feature hierarchy
- 78.0% top-1 competitive with VGG-16 equivalent-depth models at 2.3× fewer parameters

---

## ImageNet Performance — ResNet Family

| Variant | Params | Input | Top-1 | Top-5 |
|---------|:------:|:-----:|:-----:|:-----:|
| ResNet50 | 25.6 M | 224² | 74.9% | 92.1% |
| ResNet50V2 | 25.6 M | 224² | 75.6% | 92.8% |
| ResNet101 | 44.7 M | 224² | 76.4% | 92.8% |
| ResNet101V2 | 44.7 M | 224² | 77.2% | 93.8% |
| ResNet152 | 60.2 M | 224² | 76.6% | 93.1% |
| ResNet152V2 | 60.2 M | 224² | 78.0% | 94.2% |

---

## When to Use ResNet-152 V2

Use ResNet-152V2 as the best standard ResNet baseline. The 1.4% V1→V2 improvement is the highest in the family — clearly prefer V2 at this depth. For new research, ConvNeXt-S achieves 83.1% vs 78.0% at comparable parameters.

---

## Real-World Use Cases

- Strongest pure-ResNet baseline for classification benchmarks
- Transfer learning when maximum ResNet representation quality is required
- Historical benchmark for very deep pre-activation networks

---

## Folder Structure

```
ResNet152V2/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.ResNet152V2(
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

base = tf.keras.applications.ResNet152V2(
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

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TensorFlow-Pretrained-Model-Zoo.ResNet152V2&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
