# NASNet Mobile — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** NASNetMobile TensorFlow pretrained 5.3M 74.4% ImageNet NAS mobile Keras neural architecture search transfer learning classification 2018

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

NASNet Mobile is a cell-based architecture discovered by Neural Architecture Search (NAS) optimized for mobile (< 600 M MAdds) constraints. With 5.3 M parameters, it achieves 74.4% ImageNet top-1 — surpassing MobileNetV2 (71.3%) at similar complexity — by using automatically designed Normal and Reduction cells rather than hand-crafted blocks.

---

## Model Specifications

<div align="center">

| Property | Value |
|----------|-------|
| **Parameters** | 5.3 M |
| **Input Resolution** | 224×224 |
| **ImageNet Top-1** | 74.4% |
| **ImageNet Top-5** | 91.9% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.NASNetMobile` |
| **Year** | 2018 |
| **Venue** | CVPR 2018 |

</div>

---

## Architecture Highlights

- Automatically designed Normal Cell and Reduction Cell via NAS controller
- Cell-based architecture with N=4 cells repeated per stage
- Skip connections and identity operations discovered by the NAS process
- 5.3 M parameters with 74.4% top-1 — stronger than hand-designed mobile models
- Transferable cells: same cell structure scales to NASNetLarge

---

## ImageNet Performance — NASNet Family

<div align="center">

| Model | Params | Input | Top-1 | Top-5 |
|-------|:------:|:-----:|:-----:|:-----:|
| NASNetMobile | 5.3 M | 224² | 74.4% | 91.9% |
| NASNetLarge | 88.9 M | 331² | 82.7% | 96.2% |

</div>

---

## When to Use NASNet Mobile

Use NASNetMobile when you need > 74% accuracy on mobile hardware and MobileNetV2 (71.3%) is insufficient. Note: training is slower than MobileNets due to cell complexity. EfficientNet-B0 (77.1%) is usually more practical for new work.

---

## Real-World Use Cases

- Mobile apps requiring higher accuracy (74%) than MobileNetV2
- NAS architecture research baseline and ablation studies
- Edge devices with 1–2 GB RAM and moderate inference requirements
- Transfer learning when discovered architecture generalization is desired

---

## Folder Structure

```
NASNetMobile/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.NASNetMobile(
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

base = tf.keras.applications.NASNetMobile(
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
@inproceedings{zoph2018learning,
  title={Learning Transferable Architectures for Scalable Image Recognition},
  author={Zoph, Barret and Vasudevan, Vijay and Shlens, Jonathon and Le, Quoc V},
  booktitle={CVPR},
  pages={8697--8710},
  year={2018}
}
```

**Paper:** [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012)
**Authors:** Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le
**Venue:** CVPR 2018

---

<div align="center">
<sub>Part of the <a href="../README.md">TensorFlow Pretrained Model Zoo</a> — 38 models, 10 families, ready-to-run notebooks and scripts</sub>
</div>

---

<div align="center">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TensorFlow-Pretrained-Model-Zoo.NASNetMobile&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
