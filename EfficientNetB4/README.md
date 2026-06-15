# EfficientNet-B4 — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** EfficientNet B4 TensorFlow pretrained 19M 380px 82.9% ImageNet Keras satellite medical fine-grained transfer learning classification

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

EfficientNet-B4 achieves 82.9% ImageNet top-1 at 380² input with 19 M parameters. The higher resolution enables detailed texture and edge discrimination, making it particularly effective for satellite imagery, medical imaging, and fine-grained tasks. It matches EfficientNetV2-B3 performance while being architecturally simpler.

---

## Model Specifications

| Property | Value |
|----------|-------|
| **Parameters** | 19 M |
| **Input Resolution** | 380×380 |
| **ImageNet Top-1** | 82.9% |
| **ImageNet Top-5** | 96.4% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.EfficientNetB4` |
| **Year** | 2019 |
| **Venue** | ICML 2019 |

---

## Architecture Highlights

- 380×380 input resolution — highest among the compact B0–B4 sub-family
- 19 M parameters provide rich feature capacity for complex domains
- SE blocks throughout MBConv stages with learnable channel recalibration
- Dropout 0.4 — strong regularization for smaller domain-specific datasets
- Common component of ensemble systems targeting 83–84% ImageNet accuracy

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

## When to Use EfficientNet-B4

Use B4 when high-resolution input is critical (> 300 px) and you want to stay within the EfficientNet family before jumping to V2 models. Significantly outperforms B3 on fine-grained tasks where 380² input resolves discriminating details.

---

## Real-World Use Cases

- Satellite image scene classification (EuroSAT, AID, NWPU-RESISC45 benchmarks)
- Medical imaging with high-resolution pathology slides
- Document image classification and OCR preprocessing
- Low-light image quality assessment after restoration
- Species identification from high-resolution camera trap photos

---

## Folder Structure

```
EfficientNetB4/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.EfficientNetB4(
    include_top=True,
    weights="imagenet",
    input_shape=(380, 380, 3),
    classes=1000,
)
model.summary()
```

---

## Transfer Learning

```python
import tensorflow as tf

NUM_CLASSES = 10  # replace with your number of classes

base = tf.keras.applications.EfficientNetB4(
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

![Profile Views](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FmdnuruzzamanKALLOL%2FTensorFlow-Pretrained-Model-Zoo%2Ftree%2Fmaster%2FEfficientNetB4&count_bg=%23FF6F00&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Profile%20Views&edge_flat=false)

</div>
