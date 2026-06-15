# VGG-19 — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** VGG-19 TensorFlow pretrained 143.7M 71.3% ImageNet style transfer perceptual loss GAN SRGAN Keras feature extraction classification 2014

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

VGG-19 extends VGG-16 with three additional convolutional layers (16 conv layers total), reaching 143.7 M parameters at identical 71.3% ImageNet top-1 accuracy. Despite no classification gain, VGG-19 feature maps at deeper layers provide richer perceptual representations, making it the preferred choice for style transfer and image generation tasks.

---

## Model Specifications

| Property | Value |
|----------|-------|
| **Parameters** | 143.7 M |
| **Input Resolution** | 224×224 |
| **ImageNet Top-1** | 71.3% |
| **ImageNet Top-5** | 90.0% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.VGG19` |
| **Year** | 2014 |
| **Venue** | ICLR 2015 |

---

## Architecture Highlights

- 16 convolutional layers vs VGG-16's 13 — three extra conv layers in stages 3–5
- Same 4096-dim fully-connected head as VGG-16
- Deeper block3/block4/block5 feature maps preferred for style loss computation
- Marginal parameter increase (5.7 M) over VGG-16 with equivalent classification accuracy
- block1/2 features for content loss; block3/4 for style loss in standard recipes

---

## ImageNet Performance — VGG Family

| Model | Params | Input | Top-1 | Top-5 |
|-------|:------:|:-----:|:-----:|:-----:|
| VGG16 | 138 M | 224² | 71.3% | 90.1% |
| VGG19 | 143.7 M | 224² | 71.3% | 90.0% |

---

## When to Use VGG-19

Use VGG-19 over VGG-16 for style transfer and perceptual loss — the extra conv layers in blocks 3–5 produce richer style representations. For classification, both achieve identical accuracy; use ResNet instead.

---

## Real-World Use Cases

- Neural style transfer — preferred over VGG-16 for deeper style feature maps
- Image synthesis quality assessment with perceptual (VGG) loss
- GAN training perceptual loss (pix2pix, CycleGAN, SRGAN)
- Low-light image enhancement perceptual loss (e.g., EnlightenGAN)
- Texture synthesis and texture transfer research

---

## Folder Structure

```
VGG19/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.VGG19(
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

base = tf.keras.applications.VGG19(
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
@article{simonyan2014very,
  title={Very Deep Convolutional Networks for Large-Scale Image Recognition},
  author={Simonyan, Karen and Zisserman, Andrew},
  journal={arXiv preprint arXiv:1409.1556},
  year={2014}
}
```

**Paper:** [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
**Authors:** Karen Simonyan, Andrew Zisserman
**Venue:** ICLR 2015

---

<div align="center">
<sub>Part of the <a href="../README.md">TensorFlow Pretrained Model Zoo</a> — 38 models, 10 families, ready-to-run notebooks and scripts</sub>
</div>
