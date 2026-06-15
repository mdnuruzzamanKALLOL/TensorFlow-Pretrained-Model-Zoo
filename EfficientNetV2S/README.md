# EfficientNetV2-S — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** EfficientNetV2-S TensorFlow pretrained 21.5M 384px 83.9% ImageNet Keras large-scale transfer learning fast training classification 2021

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

EfficientNetV2-S is the first large-scale V2 variant, achieving 83.9% ImageNet top-1 at 21.5 M parameters and 384² input. Originally designed for ImageNet-21K pretraining, it produces exceptionally transferable features and trains up to 6.8× faster than EfficientNet-B7 (84.3%) with comparable accuracy.

---

## Model Specifications

| Property | Value |
|----------|-------|
| **Parameters** | 21.5 M |
| **Input Resolution** | 384×384 |
| **ImageNet Top-1** | 83.9% |
| **ImageNet Top-5** | 96.7% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.EfficientNetV2S` |
| **Year** | 2021 |
| **Venue** | ICML 2021 |

---

## Architecture Highlights

- 384×384 input with 6-stage architecture mixing Fused-MBConv and MBConv+SE
- Designed for ImageNet-21K pretraining → ImageNet-1K fine-tuning pipeline
- Achieves 83.9% — matching EfficientNet-B6 (84.0%) at half the parameters
- 6.8× faster training than EfficientNet-B7 for comparable accuracy
- Recommended starting point for large-domain transfer learning

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

## When to Use EfficientNetV2-S

Use V2-S as the go-to large model when 83–84% ImageNet accuracy is required. Far faster to train than EfficientNet-B6/B7 at similar accuracy. The default V2 model for server-side production deployments.

---

## Real-World Use Cases

- Transfer learning starting point for complex domains (medical, satellite, scientific)
- EfficientDetV2-S object detection backbone
- High-accuracy classification APIs where inference is on server-side GPUs
- ImageNet-21K→domain two-stage fine-tuning pipeline
- Benchmark reference for comparing CNN vs ViT scaling at 384²

---

## Folder Structure

```
EfficientNetV2S/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.EfficientNetV2S(
    include_top=True,
    weights="imagenet",
    input_shape=(384, 384, 3),
    classes=1000,
)
model.summary()
```

---

## Transfer Learning

```python
import tensorflow as tf

NUM_CLASSES = 10  # replace with your number of classes

base = tf.keras.applications.EfficientNetV2S(
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

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TensorFlow-Pretrained-Model-Zoo.EfficientNetV2S&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
