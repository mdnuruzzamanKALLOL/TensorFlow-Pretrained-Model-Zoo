# VGG-16 — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** VGG-16 TensorFlow pretrained 138M 71.3% ImageNet style transfer perceptual loss Keras feature extraction classification 2014

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

VGG-16 uses a uniform architecture of 3×3 convolutions stacked in 13 convolutional layers and 3 fully-connected layers, demonstrating that network depth with small filters is the key factor in achieving strong performance. Despite 138 M parameters, it achieves 71.3% ImageNet top-1 and remains a foundational model for style transfer and perceptual loss functions.

---

## Model Specifications

| Property | Value |
|----------|-------|
| **Parameters** | 138 M |
| **Input Resolution** | 224×224 |
| **ImageNet Top-1** | 71.3% |
| **ImageNet Top-5** | 90.1% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.VGG16` |
| **Year** | 2014 |
| **Venue** | ICLR 2015 |

---

## Architecture Highlights

- Uniform 3×3 convolutional filters throughout all 13 conv layers
- 5 max-pooling stages for progressive spatial downsampling
- 3 fully-connected layers at the top (4096→4096→1000)
- Homogeneous architecture makes it easy to analyze and modify
- fc6/fc7 layers commonly extracted as 4096-dim image embeddings

---

## ImageNet Performance — VGG Family

| Model | Params | Input | Top-1 | Top-5 |
|-------|:------:|:-----:|:-----:|:-----:|
| VGG16 | 138 M | 224² | 71.3% | 90.1% |
| VGG19 | 143.7 M | 224² | 71.3% | 90.0% |

---

## When to Use VGG-16

Use VGG-16 specifically for style transfer and perceptual loss computation — these applications were designed around VGG's feature maps. For classification or transfer learning, ResNet-50 and EfficientNet are far more efficient.

---

## Real-World Use Cases

- Neural style transfer (Gatys et al. uses VGG-16/19 perceptual loss)
- Perceptual loss for image super-resolution and image-to-image translation
- Feature extraction for classic computer vision systems
- Educational: clear, simple architecture for teaching CNNs
- Content-based image retrieval using fc7 embeddings

---

## Folder Structure

```
VGG16/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.VGG16(
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

base = tf.keras.applications.VGG16(
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
