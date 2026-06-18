# NASNet Large — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** NASNetLarge TensorFlow pretrained 88.9M 331px 82.7% ImageNet NAS SOTA 2018 Keras neural architecture search transfer learning classification

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

NASNet Large uses the same NAS-discovered cell design as NASNetMobile but scaled to 88.9 M parameters and 331² input, achieving 82.7% ImageNet top-1. When introduced in 2018, it was the highest-accuracy model on ImageNet, demonstrating that NAS could discover architectures surpassing all hand-designed models.

---

## Model Specifications

<div align="center">

| Property | Value |
|----------|-------|
| **Parameters** | 88.9 M |
| **Input Resolution** | 331×331 |
| **ImageNet Top-1** | 82.7% |
| **ImageNet Top-5** | 96.2% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.NASNetLarge` |
| **Year** | 2018 |
| **Venue** | CVPR 2018 |

</div>

---

## Architecture Highlights

- N=6 Normal Cells per stage scaled from NASNetMobile's N=4
- F=168 filter multiplier (vs F=44 in Mobile) — 4× wider channels
- 331×331 input resolution for fine-grained spatial feature capture
- First NAS model to achieve SOTA on ImageNet in 2018
- Cell transferability: same cells from CIFAR-10 NAS transferred to ImageNet

---

## ImageNet Performance — NASNet Family

<div align="center">

| Model | Params | Input | Top-1 | Top-5 |
|-------|:------:|:-----:|:-----:|:-----:|
| NASNetMobile | 5.3 M | 224² | 74.4% | 91.9% |
| NASNetLarge | 88.9 M | 331² | 82.7% | 96.2% |

</div>

---

## When to Use NASNet Large

Use NASNetLarge primarily for reproducing NAS research baselines. For new high-accuracy projects, ConvNeXt-S/B or EfficientNetV2-S achieve similar or better accuracy with much faster training. NASNetLarge is architecturally complex and slower to train than modern alternatives.

---

## Real-World Use Cases

- Historical NAS benchmark reference in architecture search papers
- High-accuracy feature extraction when parameter count is not a concern
- Transfer learning onto large labeled datasets (> 100 k images)
- Ensemble diversity member due to unique NAS-designed feature distribution

---

## Folder Structure

```
NASNetLarge/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.NASNetLarge(
    include_top=True,
    weights="imagenet",
    input_shape=(331, 331, 3),
    classes=1000,
)
model.summary()
```

---

## Transfer Learning

```python
import tensorflow as tf

NUM_CLASSES = 10  # replace with your number of classes

base = tf.keras.applications.NASNetLarge(
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
@inproceedings{zoph2018learning,
  title={Learning Transferable Architectures for Scalable Image Recognition},
  author={Zoph, Barret and Vasudevan, Vijay and Shlens, Jonathon and Le, Quoc V},
  booktitle={CVPR},
  pages={8697--8710},
  year={2018}
}
```

**Paper:** [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012)
**Authors:** Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le
**Venue:** CVPR 2018

---

<div align="center">
<sub>Part of the <a href="../README.md">TensorFlow Pretrained Model Zoo</a> — 38 models, 10 families, ready-to-run notebooks and scripts</sub>
</div>

---

<div align="center">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TensorFlow-Pretrained-Model-Zoo.NASNetLarge&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
