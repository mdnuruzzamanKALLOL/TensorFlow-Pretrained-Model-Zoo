# EfficientNetV2-B1 — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** EfficientNetV2 B1 TensorFlow pretrained 8.1M 240px 79.8% ImageNet Keras progressive learning Fused-MBConv transfer learning 2021

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

EfficientNetV2-B1 scales B0 in depth and resolution to 240², achieving 79.8% ImageNet top-1 at 8.1 M parameters. It trains significantly faster than the equivalent EfficientNet-B1 while exceeding its accuracy, making it the recommended compact model when accuracy in the 79–80% range is the target.

---

## Model Specifications

| Property | Value |
|----------|-------|
| **Parameters** | 8.1 M |
| **Input Resolution** | 240×240 |
| **ImageNet Top-1** | 79.8% |
| **ImageNet Top-5** | 95.0% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.EfficientNetV2B1` |
| **Year** | 2021 |
| **Venue** | ICML 2021 |

---

## Architecture Highlights

- 240×240 input with Fused-MBConv early stages for efficient feature extraction
- Progressive learning curriculum: image size ramps from 128 to 240 during training
- 8.1 M parameters with 79.8% accuracy — better than EfficientNet-B1 (79.1%) at similar size
- RandAugment + Mixup augmentation in the V2 training recipe

---

## ImageNet Performance — EfficientNetV2 Family

| Variant | Params | Input | Top-1 | Top-5 |
|---------|:------:|:-----:|:-----:|:-----:|
| EfficientNetV2B0 | 7.1 M | 224² | 78.7% | 94.3% |
| EfficientNetV2B1 | 8.1 M | 240² | 79.8% | 95.0% |
| EfficientNetV2B2 | 10.1 M | 260² | 80.5% | 95.1% |
| EfficientNetV2B3 | 14.4 M | 300² | 82.0% | 95.8% |
| EfficientNetV2S | 21.5 M | 384² | 83.9% | 96.7% |
| EfficientNetV2M | 54.1 M | 480² | 85.2% | 97.4% |
| EfficientNetV2L | 119 M | 480² | 86.8% | 97.9% |

---

## When to Use EfficientNetV2-B1

Use V2-B1 when accuracy in the 79–80% range is needed with minimal resource overhead. Trains 2–4× faster than EfficientNet-B1 with marginally better accuracy. The default lightweight V2 model when B0 accuracy is insufficient.

---

## Real-World Use Cases

- Lightweight classification tasks requiring > 79% accuracy
- Footwear, fashion, and e-commerce product recognition
- Document and scene text image classification
- Video frame classification for action recognition backbones

---

## Folder Structure

```
EfficientNetV2B1/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.EfficientNetV2B1(
    include_top=True,
    weights="imagenet",
    input_shape=(240, 240, 3),
    classes=1000,
)
model.summary()
```

---

## Transfer Learning

```python
import tensorflow as tf

NUM_CLASSES = 10  # replace with your number of classes

base = tf.keras.applications.EfficientNetV2B1(
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
@inproceedings{tan2021efficientnetv2,
  title={{EfficientNetV2}: Smaller Models and Faster Training},
  author={Tan, Mingxing and Le, Quoc V},
  booktitle={ICML},
  pages={10096--10106},
  year={2021}
}
```

**Paper:** [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
**Authors:** Mingxing Tan, Quoc V. Le
**Venue:** ICML 2021

---

<div align="center">
<sub>Part of the <a href="../README.md">TensorFlow Pretrained Model Zoo</a> — 38 models, 10 families, ready-to-run notebooks and scripts</sub>
</div>
