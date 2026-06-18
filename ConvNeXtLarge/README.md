# ConvNeXt Large — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** ConvNeXt Large TensorFlow pretrained 198M ImageNet 84.3% top accuracy CNN fine-tuning Keras transfer learning classification 2022

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

ConvNeXt Large pushes CNN accuracy to 84.3% on ImageNet-1K with 198 M parameters. It matches or exceeds ViT-L/16 performance in transfer learning scenarios while remaining a pure convolutional model — no attention mechanisms, fully compatible with standard CNN deployment pipelines including TFLite and TensorRT.

---

## Model Specifications

<div align="center">

| Property | Value |
|----------|-------|
| **Parameters** | 198 M |
| **Input Resolution** | 224×224 |
| **ImageNet Top-1** | 84.3% |
| **ImageNet Top-5** | 96.9% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.ConvNeXtLarge` |
| **Year** | 2022 |
| **Venue** | CVPR 2022 |

</div>

---

## Architecture Highlights

- 192-channel base width at stage 1, scaling to 1536 at the final stage
- Highly over-parameterized for rich feature representations on complex domains
- Stochastic depth with rate 0.4 for strong regularization
- Excellent for ImageNet-22K → ImageNet-1K two-stage fine-tuning
- Compatible with mixed-precision (FP16) training for memory efficiency

---

## ImageNet Performance — ConvNeXt Family

<div align="center">

| Variant | Params | Input | Top-1 | Top-5 |
|---------|:------:|:-----:|:-----:|:-----:|
| ConvNeXtTiny | 28 M | 224² | 81.3% | 95.6% |
| ConvNeXtSmall | 50 M | 224² | 83.1% | 96.4% |
| ConvNeXtBase | 89 M | 224² | 83.8% | 96.7% |
| ConvNeXtLarge | 198 M | 224² | 84.3% | 96.9% |
| ConvNeXtXLarge | 350 M | 224² | 85.4% | 97.4% |

</div>

---

## When to Use ConvNeXt Large

Use ConvNeXt Large when accuracy is the primary objective and you have access to multi-GPU training (24+ GB VRAM total). Not recommended for edge or mobile deployment.

---

## Real-World Use Cases

- High-stakes medical AI: radiology report assistance, pathology grading
- Large-vocabulary fine-grained recognition tasks
- Remote sensing with very high resolution imagery
- Foundation model feature extractor for downstream adapters
- Research on CNN vs Transformer scaling behaviour

---

## Folder Structure

```
ConvNeXtLarge/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.ConvNeXtLarge(
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

base = tf.keras.applications.ConvNeXtLarge(
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
@inproceedings{liu2022convnet,
  title={A ConvNet for the 2020s},
  author={Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
  booktitle={CVPR},
  pages={11976--11986},
  year={2022}
}
```

**Paper:** [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
**Authors:** Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie
**Venue:** CVPR 2022

---

<div align="center">
<sub>Part of the <a href="../README.md">TensorFlow Pretrained Model Zoo</a> — 38 models, 10 families, ready-to-run notebooks and scripts</sub>
</div>

---

<div align="center">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TensorFlow-Pretrained-Model-Zoo.ConvNeXtLarge&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
