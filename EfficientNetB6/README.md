# EfficientNet-B6 — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** EfficientNet B6 TensorFlow pretrained 43M 528px 84% ImageNet Keras competition ensemble high-resolution transfer learning

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

EfficientNet-B6 reaches 84.0% ImageNet top-1 with 43 M parameters at 528² input. It is a research-grade model requiring multi-GPU training for efficient fine-tuning. As part of ensemble systems it consistently contributes accuracy diversity due to its unique 528² resolution perspective on input images.

---

## Model Specifications

| Property | Value |
|----------|-------|
| **Parameters** | 43 M |
| **Input Resolution** | 528×528 |
| **ImageNet Top-1** | 84.0% |
| **ImageNet Top-5** | 96.9% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.EfficientNetB6` |
| **Year** | 2019 |
| **Venue** | ICML 2019 |

---

## Architecture Highlights

- 528×528 input — one of the highest resolutions in the B-series
- 43 M parameters with compound-scaled depth, width, and resolution
- Dropout 0.5 — maximum within the B-series for heavy regularization
- Commonly used in top-tier competition ensemble stacks

---

## ImageNet Performance — EfficientNet Family

| Variant | Params | Input | Top-1 | Top-5 |
|---------|:------:|:-----:|:-----:|:-----:|
| EfficientNetB0 | 5.3 M | 224² | 77.1% | 93.3% |
| EfficientNetB1 | 7.8 M | 240² | 79.1% | 94.4% |
| EfficientNetB2 | 9.1 M | 260² | 80.1% | 94.9% |
| EfficientNetB3 | 12 M | 300² | 81.6% | 95.7% |
| EfficientNetB4 | 19 M | 380² | 82.9% | 96.4% |
| EfficientNetB5 | 30 M | 456² | 83.6% | 96.7% |
| EfficientNetB6 | 43 M | 528² | 84.0% | 96.9% |
| EfficientNetB7 | 66 M | 600² | 84.3% | 97.0% |

---

## When to Use EfficientNet-B6

Use B6 in ensemble settings or when maximizing resolution is important and B5 accuracy is insufficient. Not recommended as a standalone model for most production use cases due to memory and latency overhead — consider EfficientNetV2-M instead.

---

## Real-World Use Cases

- Competition ensemble member for image classification leaderboards
- Crop disease analysis at extreme image resolution
- Industrial quality control with very high-resolution imaging

---

## Folder Structure

```
EfficientNetB6/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.EfficientNetB6(
    include_top=True,
    weights="imagenet",
    input_shape=(528, 528, 3),
    classes=1000,
)
model.summary()
```

---

## Transfer Learning

```python
import tensorflow as tf

NUM_CLASSES = 10  # replace with your number of classes

base = tf.keras.applications.EfficientNetB6(
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
@inproceedings{tan2019efficientnet,
  title={{EfficientNet}: Rethinking Model Scaling for Convolutional Neural Networks},
  author={Tan, Mingxing and Le, Quoc V},
  booktitle={ICML},
  pages={6105--6114},
  year={2019}
}
```

**Paper:** [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
**Authors:** Mingxing Tan, Quoc V. Le
**Venue:** ICML 2019

---

<div align="center">
<sub>Part of the <a href="../README.md">TensorFlow Pretrained Model Zoo</a> — 38 models, 10 families, ready-to-run notebooks and scripts</sub>
</div>


---

<div align="center">

![Profile Views](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FmdnuruzzamanKALLOL%2FTensorFlow-Pretrained-Model-Zoo%2Ftree%2Fmaster%2FEfficientNetB6&count_bg=%23FF6F00&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Profile%20Views&edge_flat=false)

</div>
