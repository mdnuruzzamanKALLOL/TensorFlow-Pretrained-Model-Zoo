# ConvNeXt Tiny — TensorFlow / Keras Pretrained Model | ImageNet Classification

> **Keywords:** ConvNeXt Tiny TensorFlow pretrained model ImageNet classification 2022 transfer learning fine-tuning CNN backbone feature extraction Keras deep learning

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Integrated-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

ConvNeXt Tiny is a modernized pure-convolutional network from Facebook AI Research that matches Vision Transformer accuracy while retaining CNN inference efficiency. Built by re-examining the design decisions in ResNets through the lens of modern Transformers, it achieves 81.3% top-1 accuracy on ImageNet with only 28 M parameters.

---

## Model Specifications

| Property | Value |
|----------|-------|
| **Parameters** | 28 M |
| **Input Resolution** | 224×224 |
| **ImageNet Top-1** | 81.3% |
| **ImageNet Top-5** | 95.6% |
| **Framework** | TensorFlow 2.x / Keras |
| **TF Class** | `tf.keras.applications.ConvNeXtTiny` |
| **Year** | 2022 |
| **Venue** | CVPR 2022 |

---

## Architecture Highlights

- Patchify stem: 4×4, stride-4 non-overlapping convolutions replace the 7×7 ResNet stem
- Inverted bottleneck block with 7×7 depthwise convolution (large kernel like Swin attention)
- Layer Normalization instead of Batch Normalization for training stability
- GELU activation throughout; one BN/activation per block (fewer than ResNet)
- Separate downsampling layers (2×2 stride-2 convolution) between stages

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

## When to Use ConvNeXt Tiny

Choose ConvNeXt Tiny when you need a modern CNN that outperforms Swin-T (81.3 vs 81.2%) with simpler training and faster inference. Ideal for transfer learning when GPU memory is limited (< 8 GB) and ViT-scale models are impractical.

---

## Real-World Use Cases

- General-purpose image classification on custom datasets via fine-tuning
- Object detection backbone (COCO) paired with FPN or RetinaNet neck
- Semantic segmentation (ADE20K) with UperNet or DeepLabV3+ head
- Medical imaging: histopathology, chest X-ray, dermatology screening
- Agricultural AI: crop disease detection, plant phenotyping

---

## Folder Structure

```
ConvNeXtTiny/
├── NoteBook/                 # Jupyter notebook: architecture, training, evaluation
├── Python Scripts/           # Standalone .py: build, train, single-image inference
└── Using Weight File/        # feature_extraction.py, fine_tuning.py with ImageNet weights
```

---

## Quick Start

```python
import tensorflow as tf

model = tf.keras.applications.ConvNeXtTiny(
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

base = tf.keras.applications.ConvNeXtTiny(
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
