# MobileNet — Efficient CNNs for Mobile Applications (TensorFlow / Keras)

**Paper:** MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
**Authors:** Howard, Zhu, Chen, Kalenichenko, Wang, Weyand, Andreetto, Adam
**Year:** 2017 (arXiv)

---

## Overview

MobileNet replaces standard convolutions with **Depthwise Separable Convolutions**,
factoring a k×k×C_in×C_out convolution into:
1. A **depthwise** conv (k×k×C_in, one filter per channel)
2. A **pointwise** conv (1×1×C_in×C_out, mixes channels)

This reduces parameters and FLOPs by ~8-9× with only ~1% Top-1 accuracy drop,
making it ideal for mobile and embedded deployment.

---

## Depthwise Separable Conv Block

```
Input (H × W × C_in)
├── DepthwiseConv2D(3×3, stride) → BN → ReLU6       (spatial filter, per channel)
└── Conv2D(1×1, C_out)          → BN → ReLU6       (channel mixer)
Output (H/s × W/s × C_out)

Cost ratio vs standard conv: 1/C_out + 1/k²  ≈  1/9  (for 3×3, large C_out)
```

**ReLU6** = min(max(0, x), 6) — clips activations at 6 for fixed-point robustness.

---

## Architecture

```
Input (224×224×3)
│
├── Stem: Conv3×3/2 (32)    → BN → ReLU6       → 112×112×32
│
├── DW Block  1: DWSep( 64, s=1)               → 112×112×64
├── DW Block  2: DWSep(128, s=2)               →  56×56×128
├── DW Block  3: DWSep(128, s=1)               →  56×56×128
├── DW Block  4: DWSep(256, s=2)               →  28×28×256
├── DW Block  5: DWSep(256, s=1)               →  28×28×256
├── DW Block  6: DWSep(512, s=2)               →  14×14×512
├── DW Blocks 7-11: DWSep(512, s=1) × 5       →  14×14×512
├── DW Block 12: DWSep(1024, s=2)              →   7×7×1024
├── DW Block 13: DWSep(1024, s=1)              →   7×7×1024
│
└── GlobalAvgPool → Dense(num_classes, softmax)
```

---

## Width & Resolution Multipliers

| alpha | Params | Top-1 |
|-------|--------|-------|
| 1.00 | ~4.2M | ~70.6% |
| 0.75 | ~2.6M | ~68.4% |
| 0.50 | ~1.3M | ~64.0% |
| 0.25 | ~0.5M | ~50.6% |

```python
model = build_mobilenet(num_classes=10, alpha=0.75)  # smaller model
```

Resolution multiplier: change `input_shape=(192, 192, 3)` or `(160, 160, 3)`.

---

## Key Stats (alpha=1.0)

| Property | Value |
|----------|-------|
| Parameters | ~4.2M |
| Top-1 (ImageNet) | ~70.6% |
| Input size | 224×224 |
| Activation | ReLU6 |

---

## Transfer Learning

```python
from tensorflow import keras
from tensorflow.keras.applications.mobilenet import preprocess_input

base_model = keras.applications.MobileNet(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = keras.layers.GlobalAveragePooling2D()(base_model.output)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = keras.Model(base_model.input, outputs)

datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input)  # x/127.5 - 1.0
```

---

## Folder Structure

```
MobileNet/
├── README.md
├── NoteBook/
│   └── mobilenet.ipynb
├── Python Scripts/
│   ├── mobilenet.py          — build_mobilenet(alpha=1.0) from scratch
│   ├── train.py
│   ├── inference.py
│   └── How to run.txt
└── Using Weight File/
    ├── load_pretrained.py
    ├── feature_extraction.py
    ├── fine_tuning.py        — unfreeze DW blocks 11-13
    └── How to run.txt
```

---

## Citation

```bibtex
@article{howard2017mobilenets,
  title   = {MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications},
  author  = {Howard, Andrew G. and Zhu, Menglong and Chen, Bo and Kalenichenko, Dmitry
             and Wang, Weijun and Weyand, Tobias and Andreetto, Marco and Adam, Hartwig},
  journal = {arXiv:1704.04861},
  year    = {2017}
}
```
