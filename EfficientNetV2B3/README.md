# EfficientNetV2-B3 — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** EfficientNetV2 B3 TensorFlow pretrained 14.4M 300px 82% ImageNet Keras compact transfer learning classification 2021

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

EfficientNetV2-B3 reaches 82.0% ImageNet top-1 at 14.4 M parameters and 300² input, exceeding EfficientNet-B3 (81.6%) while training faster. It serves as the accuracy bridge between compact V2-B0/B1/B2 models and the large V2-S/M/L variants, providing a strong production backbone for demanding tasks.

---

## Model Specifications

| Property | Value |
|----------|-------|
| **Parameters** | 14.4 M |
| **Input Resolution** | 300×300 |
| **ImageNet Top-1** | 82.0% |
| **ImageNet Top-5** | 95.8% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.EfficientNetV2B3` |
| **Year** | 2021 |
| **Venue** | ICML 2021 |

---

## Architecture Highlights

- 300×300 input with mixed Fused-MBConv + MBConv+SE block structure
- 14.4 M parameters: strongest compact V2 variant before V2-S
- Achieves 82.0% — 0.4% better than EfficientNet-B3 with faster training
- Highly competitive with ResNet-50 and MobileNetV3-Large at much lower cost

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

## When to Use EfficientNetV2-B3

Use V2-B3 as the best compact V2 model for tasks requiring 82%+ accuracy. Prefer over EfficientNet-B3 in all new projects — same resolution, more accurate, faster training.

---

## Real-World Use Cases

- Production image classification APIs requiring 82%+ accuracy
- Research comparing V1 vs V2 EfficientNet training speed at 300² resolution
- EfficientDetV2 object detection backbone
- Medical domain fine-tuning where compact but accurate models are preferred

---

## Folder Structure

```
EfficientNetV2B3/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.EfficientNetV2B3(
    include_top=True,
    weights="imagenet",
    input_shape=(300, 300, 3),
    classes=1000,
)
model.summary()
```

---

## Transfer Learning

```python
import tensorflow as tf

NUM_CLASSES = 10  # replace with your number of classes

base = tf.keras.applications.EfficientNetV2B3(
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

![Profile Views](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FmdnuruzzamanKALLOL%2FTensorFlow-Pretrained-Model-Zoo%2Ftree%2Fmaster%2FEfficientNetV2B3&count_bg=%23FF6F00&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Profile%20Views&edge_flat=false)

</div>
