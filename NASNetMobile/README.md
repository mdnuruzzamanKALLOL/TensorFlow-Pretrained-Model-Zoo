# NASNetMobile — Neural Architecture Search Network (TensorFlow / Keras)

**Paper:** Learning Transferable Architectures for Scalable Image Recognition
**Authors:** Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le
**Conference:** CVPR 2018

---

## Overview

NASNetMobile uses cells discovered by **Neural Architecture Search (NAS)** —
an automated process that learns the optimal building blocks rather than having
humans design them. The cells were found on CIFAR-10 and then transferred to
ImageNet scale, giving strong accuracy relative to the parameter count.

NASNetMobile is the compact, mobile-targeted variant (~5.3M parameters). It
achieves ~74.4% Top-1 accuracy — comparable to ResNet50 (~74.9%) at roughly
1/5 the parameter count.

---

## NAS Cells

NASNet uses two learned cell types stacked repeatedly:

**Normal Cell** — preserves spatial dimensions (analogous to a residual block)
```
h = relu(ip) -> conv1x1 -> BN
p = adjusted previous cell output

x1: sep5x5(h)  + identity(h)
x2: sep5x5(p)  + sep3x3(h)
x3: avg3x3(h)  + p
x4: avg3x3(p)  + avg3x3(p)
x5: max3x3(h)  + sep3x3(p)

Output: Concat([p, x1, x2, x3, x4, x5])  ->  6 × filters channels
```

**Reduction Cell** — halves spatial dimensions (analogous to a strided block)
```
x1: sep5x5_s2(h) + sep7x7_s2(p)
x2: max3x3_s2(h) + sep7x7_s2(p)
x3: avg3x3_s2(h) + sep5x5_s2(p)
x4: avg3x3_s1(x1) + x2
x5: sep3x3_s1(x1) + max3x3_s2(h)

Output: Concat([x2, x3, x4, x5])  ->  4 × filters channels
```

Each `_sep_conv_bn` is two stacked depthwise-separable convolutions,
with ReLU prefix and BN suffix on each.

---

## Architecture

```
Input (224×224×3)
│
├── Stem: Conv3×3/2 (32 filters, valid)   →  ~111×111×32
│
├── Stem Reduction Cell (filters*4=176)   →   ~56×56×704
├── Stem Reduction Cell (filters*2=88)    →   ~28×28×352
│
├── Group 1: 4 × Normal Cell (filters=44) →   ~28×28×264
├── Reduction Cell (filters*2=88)         →   ~14×14×352
│
├── Group 2: 4 × Normal Cell (filters=88) →   ~14×14×528
├── Reduction Cell (filters*4=176)        →    ~7×7×704
│
├── Group 3: 4 × Normal Cell (filters=176)→    ~7×7×1056
│
├── ReLU → GlobalAvgPool → Dense(num_classes)
```

---

## Key Stats

| Property | Value |
|----------|-------|
| Parameters | ~5.3M |
| Top-1 (ImageNet) | ~74.4% |
| Top-5 (ImageNet) | ~91.7% |
| Input size | 224×224 |
| Filters (base) | 44 |
| Normal Cells / group | 4 |

---

## NASNet Family Comparison

| Model | Params | Top-1 | Input |
|-------|--------|-------|-------|
| **NASNetMobile** | ~5.3M | ~74.4% | 224×224 |
| NASNetLarge | ~88.9M | ~82.5% | 331×331 |

---

## Transfer Learning

```python
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.nasnet import preprocess_input

base_model = keras.applications.NASNetMobile(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3),
)
x       = layers.GlobalAveragePooling2D()(base_model.output)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model   = keras.Model(base_model.input, outputs)

datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input   # x/127.5 - 1.0
)
```

---

## Folder Structure

```
NASNetMobile/
├── README.md
├── NoteBook/
│   └── nasnetmobile.ipynb          — 17-cell notebook
├── Python Scripts/
│   ├── nasnetmobile.py             — build_nasnetmobile() from scratch
│   ├── train.py                    — Adam + ReduceLROnPlateau
│   ├── inference.py                — top-K prediction
│   └── How to run.txt
└── Using Weight File/
    ├── load_pretrained.py          — load Keras NASNetMobile
    ├── feature_extraction.py       — frozen backbone
    ├── fine_tuning.py              — last 30%% layers unfrozen
    └── How to run.txt
```

---

## Citation

```bibtex
@inproceedings{zoph2018learning,
  title     = {Learning Transferable Architectures for Scalable Image Recognition},
  author    = {Zoph, Barret and Vasudevan, Vijay and Shlens, Jonathon and Le, Quoc V.},
  booktitle = {CVPR},
  year      = {2018}
}
```
