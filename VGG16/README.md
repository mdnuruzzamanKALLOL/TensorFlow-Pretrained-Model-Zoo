# VGG16 — Very Deep Convolutional Networks (TensorFlow / Keras)

**Paper:** Very Deep Convolutional Networks for Large-Scale Image Recognition
**Authors:** Karen Simonyan, Andrew Zisserman
**Conference:** ICLR 2015

---

## Overview

VGG16 demonstrated that network **depth** is a critical component for good performance.
The key design principle: use only 3×3 convolutions (the smallest useful size) stacked
in homogeneous blocks, separated by max-pooling. This simplicity made VGG widely adopted
as a backbone and feature extractor.

Two stacked 3×3 convolutions have the same receptive field as one 5×5 convolution, but
with fewer parameters and one extra non-linearity. Three stacked 3×3 equal one 7×7.

---

## Architecture

```
Input (224×224×3)
│
├── Block 1 : Conv(64)×2  + MaxPool(2×2)   →  64 × 112×112
├── Block 2 : Conv(128)×2 + MaxPool(2×2)   → 128 ×  56×56
├── Block 3 : Conv(256)×3 + MaxPool(2×2)   → 256 ×  28×28
├── Block 4 : Conv(512)×3 + MaxPool(2×2)   → 512 ×  14×14
├── Block 5 : Conv(512)×3 + MaxPool(2×2)   → 512 ×   7×7
│
├── Flatten                                 → 25,088
├── Dense(4096) + ReLU + Dropout(0.5)
├── Dense(4096) + ReLU + Dropout(0.5)
└── Dense(num_classes) + Softmax
```

All conv layers: 3×3 kernel, padding='same', ReLU activation, no bias when using BN.
All max-pools : 2×2 window, stride=2.

---

## Key Stats

| Property | Value |
|----------|-------|
| Parameters | ~138M |
| Top-1 (ImageNet) | ~71.3% |
| Top-5 (ImageNet) | ~90.1% |
| Input size | 224×224 |
| Framework | TensorFlow / Keras |

Note: ~102M of the 138M parameters are in the Dense(4096)×2 head (Flatten: 25,088→4096).
This is why feature extraction with a lighter head is common in practice.

---

## Training Configuration (From Scratch)

| Setting | Value |
|---------|-------|
| Input size | 224×224 |
| Batch size | 32 |
| Optimizer | Adam (lr=1e-3) |
| Scheduler | ReduceLROnPlateau (factor=0.1, patience=5) |
| Loss | categorical_crossentropy |
| Epochs | 30 |
| Augmentation | rotation, shift, flip, zoom, shear |

---

## Transfer Learning

```python
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import preprocess_input

base_model = keras.applications.VGG16(
    weights='imagenet',
    include_top=False,      # drop the Dense(4096)×2 + Dense(1000) head
    input_shape=(224, 224, 3),
)
x       = layers.GlobalAveragePooling2D()(base_model.output)
x       = layers.Dense(512, activation='relu')(x)
x       = layers.Dropout(0.5)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model   = keras.Model(inputs=base_model.input, outputs=outputs)

# IMPORTANT: use VGG16-specific preprocessing
datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input   # subtracts BGR mean, does NOT /255
)
```

### Two-Phase Fine-Tuning

```python
# Phase 1 — freeze all conv blocks, train head only
base_model.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-3), ...)
model.fit(train_gen, epochs=10, ...)

# Phase 2 — unfreeze block4 + block5 (keep blocks 1-3 frozen)
base_model.trainable = True
for layer in base_model.layers:
    if 'block1' in layer.name or 'block2' in layer.name or 'block3' in layer.name:
        layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-5), ...)
model.fit(train_gen, initial_epoch=10, epochs=30, ...)
```

---

## Folder Structure

```
VGG16/
├── README.md
├── NoteBook/
│   └── vgg16.ipynb              — 17-cell notebook (arch + train + ROC AUC)
├── Python Scripts/
│   ├── vgg16.py                 — build_vgg16() from scratch
│   ├── train.py                 — Adam + ReduceLROnPlateau training loop
│   ├── inference.py             — top-K single-image prediction
│   └── How to run.txt
└── Using Weight File/
    ├── load_pretrained.py       — load Keras Applications VGG16
    ├── feature_extraction.py    — frozen backbone, GAP + Dense head
    ├── fine_tuning.py           — two-phase fine-tuning (block4+5 unfreeze)
    └── How to run.txt
```

---

## Citation

```bibtex
@inproceedings{simonyan2015very,
  title     = {Very Deep Convolutional Networks for Large-Scale Image Recognition},
  author    = {Simonyan, Karen and Zisserman, Andrew},
  booktitle = {ICLR},
  year      = {2015}
}
```
