# EfficientNet-B0 — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** EfficientNet B0 TensorFlow pretrained 5.3M 77.1% ImageNet mobile edge Keras transfer learning compound scaling NAS MBConv classification

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

EfficientNet-B0 is the baseline model discovered by Neural Architecture Search (NAS) that forms the foundation of the EfficientNet family. Using compound scaling to jointly optimize width, depth, and resolution, EfficientNet-B0 achieves 77.1% ImageNet top-1 with only 5.3 M parameters — 8.4× fewer than ResNet-50 for similar accuracy.

---

## Model Specifications

| Property | Value |
|----------|-------|
| **Parameters** | 5.3 M |
| **Input Resolution** | 224×224 |
| **ImageNet Top-1** | 77.1% |
| **ImageNet Top-5** | 93.3% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.EfficientNetB0` |
| **Year** | 2019 |
| **Venue** | ICML 2019 |

---

## Architecture Highlights

- Mobile Inverted Bottleneck (MBConv) blocks with Squeeze-and-Excitation (SE) attention
- Compound scaling coefficient φ=1.0 (baseline NAS architecture)
- Depthwise separable convolutions reduce FLOPs while retaining accuracy
- SE ratio 0.25 — learns channel-wise attention with minimal overhead
- Swish activation (x · sigmoid(x)) throughout the network

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

## When to Use EfficientNet-B0

Best for edge deployment and real-time inference. Achieves 77.1% with only 5.3 M parameters — far more efficient than MobileNetV2 (71.3%) and NASNetMobile (74.4%) at similar or lower latency on mobile hardware.

---

## Real-World Use Cases

- Mobile and edge deployment on Android / iOS via TFLite export
- Real-time inference where latency < 10 ms is required
- Embedded systems: Raspberry Pi, Coral Edge TPU, Jetson Nano
- Starting point for NAS-optimized fine-tuning on custom hardware
- Baseline comparison in efficiency-accuracy trade-off studies

---

## Folder Structure

```
EfficientNetB0/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.EfficientNetB0(
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

base = tf.keras.applications.EfficientNetB0(
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

![Profile Views](https://komarev.com/ghpvc/?username=mdnuruzzamanKALLOL&label=Profile%20Views&color=FF6F00&style=flat-square)

</div>
