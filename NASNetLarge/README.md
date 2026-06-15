# NASNetLarge — Neural Architecture Search Network (TensorFlow / Keras)

**Paper:** Learning Transferable Architectures for Scalable Image Recognition
**Authors:** Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le
**Conference:** CVPR 2018

---

## Overview

NASNetLarge is the high-accuracy variant of NASNet. It uses the same
NAS-discovered Normal and Reduction Cells as NASNetMobile, but with
significantly more filters (168 vs 44), more blocks per group (6 vs 4),
and a larger input resolution (331×331 vs 224×224). It achieves **~82.5%
Top-1** on ImageNet — one of the highest accuracies among manually-crafted
and NAS-found architectures at the time of publication.

The key structural difference from NASNetMobile is **skip_reduction=True**:
after each explicit reduction cell, the skip connection p is NOT updated,
allowing normal cells in the next group to skip over the reduction cell
via the identity path.

---

## NASNet-A Cells (shared with NASNetMobile)

**Normal Cell** — preserves spatial dimensions
```
h = relu(ip) -> conv1x1(filters) -> BN
p = adjusted previous cell output (6*filters or 4*filters channels)

x1: sep5x5(h) + identity(h)
x2: sep5x5(p) + sep3x3(h)
x3: avg3x3(h) + p
x4: avg3x3(p) + avg3x3(p)
x5: max3x3(h) + sep3x3(p)
Output: Concat([p, x1, x2, x3, x4, x5])  ->  6 × filters channels
```

**Reduction Cell** — halves spatial dimensions
```
x1: sep5x5_s2(h) + sep7x7_s2(p)
x2: max3x3_s2(h) + sep7x7_s2(p)
x3: avg3x3_s2(h) + sep5x5_s2(p)
x4: avg3x3_s1(x1) + x2
x5: sep3x3_s1(x1) + max3x3_s2(h)
Output: Concat([x2, x3, x4, x5])  ->  4 × filters channels
```

---

## Architecture

```
Input (331×331×3)
│
├── Stem: Conv3×3/2 (96 filters, valid)      → ~165×165×96
│
├── Stem Reduction Cell (filters*4=672)      →  ~83×83×2688
├── Stem Reduction Cell (filters*2=336)      →  ~42×42×1344
│
├── Group 1: 6 × Normal Cell (filters=168)   →  ~42×42×1008
├── Reduction Cell (filters*2=336) [skip_red=True: p stays at Group 1 output]
│
├── Group 2: 6 × Normal Cell (filters=336)   →  ~21×21×2016
├── Reduction Cell (filters*4=672) [skip_red=True: p stays at Group 2 output]
│
├── Group 3: 6 × Normal Cell (filters=672)   →  ~11×11×4032
│
└── ReLU → GlobalAvgPool → Dense(num_classes)
```

---

## skip_reduction=True Explained

```
                    skip_reduction=False (NASNetMobile)
                    ────────────────────────────────────
Normal Cell N  ──>  x  ─── Reduction Cell ──>  x'  ─── Normal Cell N+1
                    │                           │
                    └───────────── p ───────────┘   (p = old x)

                    skip_reduction=True (NASNetLarge)
                    ──────────────────────────────────
Normal Cell N  ──>  x  ─── Reduction Cell ──>  x'  ─── Normal Cell N+1
             │             (p not updated)           │
             └──────────────────────── p ────────────┘  (p skips reduction)
```

---

## Key Stats

| Property | Value |
|----------|-------|
| Parameters | ~88.9M |
| Top-1 (ImageNet) | ~82.5% |
| Top-5 (ImageNet) | ~96.0% |
| Input size | **331×331** |
| Filters (base) | 168 |
| Normal Cells / group | 6 |

---

## Training Configuration (From Scratch)

| Setting | Value |
|---------|-------|
| Input size | 331×331 |
| Batch size | **8** (very memory-intensive) |
| Optimizer | Adam (lr=1e-3) |
| Scheduler | ReduceLROnPlateau (factor=0.1, patience=5) |
| Loss | categorical_crossentropy |
| Epochs | 30 |

---

## Transfer Learning

```python
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.nasnet import preprocess_input

base_model = keras.applications.NASNetLarge(
    weights='imagenet',
    include_top=False,
    input_shape=(331, 331, 3),   # must be 331x331
)
x       = layers.GlobalAveragePooling2D()(base_model.output)
x       = layers.Dropout(0.5)(x)               # higher dropout for large model
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model   = keras.Model(base_model.input, outputs)

# Same preprocessing as NASNetMobile
datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input   # x/127.5 - 1.0
)
```

---

## NASNet Family Comparison

| Model | Params | Top-1 | Input | Batch | Blocks/group |
|-------|--------|-------|-------|-------|-------------|
| NASNetMobile | ~5.3M | ~74.4% | 224×224 | 32 | 4 |
| **NASNetLarge** | ~88.9M | ~82.5% | 331×331 | 8 | 6 |

---

## Folder Structure

```
NASNetLarge/
├── README.md
├── NoteBook/
│   └── nasnetlarge.ipynb          — 17-cell notebook
├── Python Scripts/
│   ├── nasnetlarge.py             — build_nasnetlarge() from scratch
│   ├── train.py                   — Adam + ReduceLROnPlateau, batch=8
│   ├── inference.py               — top-K prediction (331x331 input)
│   └── How to run.txt
└── Using Weight File/
    ├── load_pretrained.py         — load Keras NASNetLarge
    ├── feature_extraction.py      — frozen backbone, Dropout(0.5) head
    ├── fine_tuning.py             — last 30%% layers unfrozen
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
