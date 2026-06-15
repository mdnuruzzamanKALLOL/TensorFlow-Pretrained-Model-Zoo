# DenseNet-169 — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** DenseNet-169 TensorFlow pretrained ImageNet 76.2% 14M parameters Keras transfer learning medical imaging dense connections classification

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

DenseNet-169 extends DenseNet-121 with a deeper final dense block (32 layers vs 16), achieving 76.2% ImageNet top-1 at 14 M parameters. The deeper dense block enables richer abstract feature representations while maintaining the dense connectivity that makes DenseNets exceptional feature extractors.

---

## Model Specifications

| Property | Value |
|----------|-------|
| **Parameters** | 14 M |
| **Input Resolution** | 224×224 |
| **ImageNet Top-1** | 76.2% |
| **ImageNet Top-5** | 93.2% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.DenseNet169` |
| **Year** | 2017 |
| **Venue** | CVPR 2017 |

---

## Architecture Highlights

- Four dense blocks with [6, 12, 32, 32] layers (vs DenseNet-121's [6, 12, 24, 16])
- Deeper final block captures more abstract semantic features
- Growth rate k=32 consistent across all blocks
- Compact model with 14 M parameters achieving 76.2% — very parameter-efficient
- Strong transfer baseline for pathology and retinal imaging research

---

## ImageNet Performance — DenseNet Family

| Variant | Params | Input | Top-1 | Top-5 |
|---------|:------:|:-----:|:-----:|:-----:|
| DenseNet121 | 8 M | 224² | 75.0% | 92.2% |
| DenseNet169 | 14 M | 224² | 76.2% | 93.2% |
| DenseNet201 | 20 M | 224² | 77.3% | 93.6% |

---

## When to Use DenseNet-169

Use DenseNet-169 when DenseNet-121 slightly underperforms and you have moderate GPU resources. The extra depth in the final block improves performance on fine-grained recognition without dramatically increasing memory footprint.

---

## Real-World Use Cases

- Histopathology classification requiring deep semantic feature extraction
- Multi-label chest pathology detection beyond DenseNet-121 capacity
- Ophthalmology: glaucoma detection from fundus photographs
- Plant disease classification from leaf images
- Benchmark comparisons within the DenseNet family

---

## Folder Structure

```
DenseNet169/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.DenseNet169(
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

base = tf.keras.applications.DenseNet169(
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
@inproceedings{huang2017densely,
  title={Densely Connected Convolutional Networks},
  author={Huang, Gao and Liu, Zhuang and Van Der Maaten, Laurens and Weinberger, Kilian Q},
  booktitle={CVPR},
  pages={4700--4708},
  year={2017}
}
```

**Paper:** [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
**Authors:** Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
**Venue:** CVPR 2017

---

<div align="center">
<sub>Part of the <a href="../README.md">TensorFlow Pretrained Model Zoo</a> — 38 models, 10 families, ready-to-run notebooks and scripts</sub>
</div>


---

<div align="center">

![Profile Views](https://komarev.com/ghpvc/?username=mdnuruzzamanKALLOL&label=Profile%20Views&color=FF6F00&style=flat-square)

</div>
