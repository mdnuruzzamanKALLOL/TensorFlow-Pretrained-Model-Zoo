# ConvNeXt XLarge — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** ConvNeXt XLarge TensorFlow pretrained 350M ImageNet 85.4% CNN largest model Keras transfer learning image classification state-of-the-art 2022

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

ConvNeXt XLarge is the largest variant in the ConvNeXt family at 350 M parameters, achieving 85.4% ImageNet top-1 accuracy (87.0% at 384² input after ImageNet-22K pretraining). It represents the state-of-the-art for pure CNN architectures and is intended for research and high-resource production systems.

---

## Model Specifications

| Property | Value |
|----------|-------|
| **Parameters** | 350 M |
| **Input Resolution** | 224×224 |
| **ImageNet Top-1** | 85.4% |
| **ImageNet Top-5** | 97.4% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.ConvNeXtXLarge` |
| **Year** | 2022 |
| **Venue** | CVPR 2022 |

---

## Architecture Highlights

- 256-channel base width scaling to 2048 channels at the final stage
- Designed primarily for two-stage pretraining: ImageNet-22K then ImageNet-1K fine-tune
- Demonstrates CNNs can scale competitively with Vision Transformers at large capacity
- Stochastic depth rate 0.5 — essential for training stability at this scale
- Supports gradient checkpointing for memory-efficient training

---

## ImageNet Performance — ConvNeXt Family

| Variant | Params | Input | Top-1 | Top-5 |
|---------|:------:|:-----:|:-----:|:-----:|
| ConvNeXtTiny | 28 M | 224² | 81.3% | 95.6% |
| ConvNeXtSmall | 50 M | 224² | 83.1% | 96.4% |
| ConvNeXtBase | 89 M | 224² | 83.8% | 96.7% |
| ConvNeXtLarge | 198 M | 224² | 84.3% | 96.9% |
| ConvNeXtXLarge | 350 M | 224² | 85.4% | 97.4% |

---

## When to Use ConvNeXt XLarge

Use ConvNeXt XLarge only when you have ImageNet-22K pretraining data or a very large domain-specific dataset (> 1 M images) and need maximum accuracy. Requires 40+ GB VRAM for full fine-tuning; use frozen feature extraction for smaller compute budgets.

---

## Real-World Use Cases

- State-of-the-art benchmark submissions requiring maximum CNN accuracy
- Scientific imaging: astronomy, microscopy, material science
- Large-scale pretraining on proprietary domain datasets
- Multi-label classification on extremely diverse class spaces

---

## Folder Structure

```
ConvNeXtXLarge/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.ConvNeXtXLarge(
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

base = tf.keras.applications.ConvNeXtXLarge(
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

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TensorFlow-Pretrained-Model-Zoo.ConvNeXtXLarge&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
