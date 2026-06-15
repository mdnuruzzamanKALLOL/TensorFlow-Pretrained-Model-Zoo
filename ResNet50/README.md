# ResNet-50 — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** ResNet-50 TensorFlow pretrained 25.6M 74.9% ImageNet CVPR 2016 Keras residual learning transfer learning backbone classification baseline

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

ResNet-50 introduced identity shortcut connections that allow gradients to flow directly through the network, enabling training of very deep networks for the first time. It won ILSVRC 2015 and remains the most widely-cited CNN backbone in deep learning research, serving as the default comparison point for new architectures.

---

## Model Specifications

| Property | Value |
|----------|-------|
| **Parameters** | 25.6 M |
| **Input Resolution** | 224×224 |
| **ImageNet Top-1** | 74.9% |
| **ImageNet Top-5** | 92.1% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.ResNet50` |
| **Year** | 2016 |
| **Venue** | CVPR 2016 |

---

## Architecture Highlights

- 50-layer residual network with 3 bottleneck residual blocks per stage
- Identity shortcuts: F(x) + x skips directly when dimensions match
- Projection shortcuts (1×1 conv) when channel dimensions change
- 4 stages with [3, 4, 6, 3] bottleneck blocks and 1024-dim feature map at stage 4
- Global average pooling before final 1000-class dense layer

---

## ImageNet Performance — ResNet Family

| Variant | Params | Input | Top-1 | Top-5 |
|---------|:------:|:-----:|:-----:|:-----:|
| ResNet50 | 25.6 M | 224² | 74.9% | 92.1% |
| ResNet50V2 | 25.6 M | 224² | 75.6% | 92.8% |
| ResNet101 | 44.7 M | 224² | 76.4% | 92.8% |
| ResNet101V2 | 44.7 M | 224² | 77.2% | 93.8% |
| ResNet152 | 60.2 M | 224² | 76.6% | 93.1% |
| ResNet152V2 | 60.2 M | 224² | 78.0% | 94.2% |

---

## When to Use ResNet-50

Use ResNet-50 as the default baseline when benchmarking new methods. For best accuracy, use EfficientNetB3 or ConvNeXtTiny. ResNet-50 remains valuable for its ubiquity, reproducibility, and ecosystem support (detection, segmentation).

---

## Real-World Use Cases

- Universal transfer learning baseline across virtually all vision tasks
- Faster R-CNN and Mask R-CNN backbone for object detection and segmentation
- DeepLabV3 semantic segmentation backbone
- Image retrieval and embedding via penultimate layer features
- Knowledge distillation teacher/student experiments

---

## Folder Structure

```
ResNet50/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.ResNet50(
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

base = tf.keras.applications.ResNet50(
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
@inproceedings{he2016deep,
  title={Deep Residual Learning for Image Recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={CVPR},
  pages={770--778},
  year={2016}
}
```

**Paper:** [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
**Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
**Venue:** CVPR 2016

---

<div align="center">
<sub>Part of the <a href="../README.md">TensorFlow Pretrained Model Zoo</a> — 38 models, 10 families, ready-to-run notebooks and scripts</sub>
</div>


---

<div align="center">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TensorFlow-Pretrained-Model-Zoo.ResNet50&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
