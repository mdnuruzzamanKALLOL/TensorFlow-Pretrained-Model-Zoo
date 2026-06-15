# ConvNeXt Base — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** ConvNeXt Base TensorFlow pretrained 89M parameters ImageNet 83.8% CNN backbone Keras transfer learning fine-tuning classification 2022

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

ConvNeXt Base delivers 83.8% ImageNet top-1 accuracy with 89 M parameters, matching Swin-B performance while being faster to train. It is the standard reference point in the ConvNeXt family and the recommended backbone for research experiments comparing CNNs versus Transformers.

---

## Model Specifications

| Property | Value |
|----------|-------|
| **Parameters** | 89 M |
| **Input Resolution** | 224×224 |
| **ImageNet Top-1** | 83.8% |
| **ImageNet Top-5** | 96.7% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.ConvNeXtBase` |
| **Year** | 2022 |
| **Venue** | CVPR 2022 |

---

## Architecture Highlights

- 128-channel base width (double ConvNeXt Tiny's 96 channels) at stage 1
- Stage configuration [3, 3, 27, 3] with progressive channel doubling
- Layer Normalization in every block — eliminates Batch Norm's batch-size dependency
- Drop-path rate of 0.3 for regularization on ImageNet-22K pretraining
- Pretrained on ImageNet-1K; ImageNet-22K variant available for larger gains

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

## When to Use ConvNeXt Base

Choose ConvNeXt Base when you need research-grade accuracy comparable to Swin-B with faster training. Requires ~16 GB GPU for fine-tuning with batch size 32.

---

## Real-World Use Cases

- Benchmark comparison against Swin-B and ViT-B models
- Medical image analysis: MRI segmentation, pathology slide classification
- Remote sensing: multi-class land-cover classification from Sentinel-2
- Industrial inspection: defect detection on manufacturing lines
- Research ablation studies requiring a reproducible CNN baseline

---

## Folder Structure

```
ConvNeXtBase/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.ConvNeXtBase(
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

base = tf.keras.applications.ConvNeXtBase(
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

![Profile Views](https://komarev.com/ghpvc/?username=mdnuruzzamanKALLOL&label=Profile%20Views&color=FF6F00&style=flat-square)

</div>
