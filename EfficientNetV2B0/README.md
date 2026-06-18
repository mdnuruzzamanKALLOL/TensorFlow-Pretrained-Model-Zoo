# EfficientNetV2-B0 — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** EfficientNetV2 B0 TensorFlow pretrained 7.1M 78.7% ImageNet Fused-MBConv progressive learning faster training Keras transfer learning 2021

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

EfficientNetV2-B0 combines Fused-MBConv blocks in early stages with MBConv+SE in later stages, achieving 78.7% ImageNet top-1 at 224² with 7.1 M parameters. It trains 5–11× faster than EfficientNet-B0 under progressive learning while exceeding its accuracy — making it the modern recommended replacement for B0 in new projects.

---

## Model Specifications

<div align="center">

| Property | Value |
|----------|-------|
| **Parameters** | 7.1 M |
| **Input Resolution** | 224×224 |
| **ImageNet Top-1** | 78.7% |
| **ImageNet Top-5** | 94.3% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.EfficientNetV2B0` |
| **Year** | 2021 |
| **Venue** | ICML 2021 |

</div>

---

## Architecture Highlights

- Fused-MBConv in stages 1–3: fuses expansion and depthwise conv into a single 3×3 conv
- Progressive learning: trains with smaller images/weaker augmentation first, then scales up
- NAS-optimized for training speed (FLOPs/accuracy/training-time) rather than inference only
- Achieves 78.7% at 7.1 M — 1.6% better than EfficientNet-B0 at similar size
- Supports tf.keras.applications directly with ImageNet weights

---

## ImageNet Performance — EfficientNetV2 Family

<div align="center">

| Variant | Params | Input | Top-1 | Top-5 |
|---------|:------:|:-----:|:-----:|:-----:|
| EfficientNetV2B0 | 7.1 M | 224² | 78.7% | 94.3% |
| EfficientNetV2B1 | 8.1 M | 240² | 79.8% | 95.0% |
| EfficientNetV2B2 | 10.1 M | 260² | 80.5% | 95.1% |
| EfficientNetV2B3 | 14.4 M | 300² | 82.0% | 95.8% |
| EfficientNetV2S | 21.5 M | 384² | 83.9% | 96.7% |
| EfficientNetV2M | 54.1 M | 480² | 85.2% | 97.4% |
| EfficientNetV2L | 119 M | 480² | 86.8% | 97.9% |

</div>

---

## When to Use EfficientNetV2-B0

Always prefer EfficientNetV2-B0 over EfficientNet-B0 for new projects. Faster to train, slightly more accurate, and architecturally more modern. Use EfficientNet-B0 only when you need to reproduce older baselines exactly.

---

## Real-World Use Cases

- Mobile and embedded deployment requiring faster training and modern architecture
- Production systems where both inference efficiency and training speed matter
- Edge AI on TFLite-compatible hardware with accuracy > 78%
- Baseline for V2 family architecture ablation studies

---

## Folder Structure

```
EfficientNetV2B0/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.EfficientNetV2B0(
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

base = tf.keras.applications.EfficientNetV2B0(
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

---

<div align="center">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TensorFlow-Pretrained-Model-Zoo.EfficientNetV2B0&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
