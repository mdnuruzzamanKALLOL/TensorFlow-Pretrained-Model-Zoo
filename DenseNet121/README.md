# DenseNet-121 — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** DenseNet-121 TensorFlow pretrained model ImageNet 75% chest X-ray medical imaging Keras transfer learning dense connections feature reuse classification

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

DenseNet-121 connects each layer to every subsequent layer in a feed-forward fashion, achieving strong feature reuse with only 8 M parameters. It won Best Paper at CVPR 2017 and remains one of the most parameter-efficient models for medical imaging, where its dense skip connections enable fine-grained gradient flow to all layers.

---

<div align="center">

## Model Specifications

| Property | Value |
|----------|-------|
| **Parameters** | 8 M |
| **Input Resolution** | 224×224 |
| **ImageNet Top-1** | 75.0% |
| **ImageNet Top-5** | 92.2% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.DenseNet121` |
| **Year** | 2017 |
| **Venue** | CVPR 2017 |

</div>

---

## Architecture Highlights

- Dense connectivity: each layer receives feature maps from all preceding layers
- Growth rate k=32: each layer adds 32 feature maps, keeping capacity controlled
- Bottleneck layers (1×1 + 3×3 conv) in dense blocks reduce computation
- Transition layers with 0.5 compression ratio halve feature map count
- Excellent gradient flow — effectively mitigates vanishing gradient problem

---

<div align="center">

## ImageNet Performance — DenseNet Family

| Variant | Params | Input | Top-1 | Top-5 |
|---------|:------:|:-----:|:-----:|:-----:|
| DenseNet121 | 8 M | 224² | 75.0% | 92.2% |
| DenseNet169 | 14 M | 224² | 76.2% | 93.2% |
| DenseNet201 | 20 M | 224² | 77.3% | 93.6% |

</div>

---

## When to Use DenseNet-121

Best choice for medical imaging and low-resource domains due to strong feature reuse with very few parameters. Preferred over ResNet-50 when GPU memory is limited and multi-scale feature aggregation matters (e.g., lesion segmentation).

---

## Real-World Use Cases

- Chest X-ray pathology detection (CheXNet benchmark uses DenseNet-121)
- Skin lesion classification (ISIC melanoma challenge)
- Retinal fundus image grading for diabetic retinopathy
- Low-data regimes: achieves competitive accuracy with < 10 k training images
- Multi-label classification tasks with sigmoid output head

---

## Folder Structure

```
DenseNet121/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.DenseNet121(
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

base = tf.keras.applications.DenseNet121(
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

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TensorFlow-Pretrained-Model-Zoo.DenseNet121&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
