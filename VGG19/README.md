# VGG19 — Very Deep Convolutional Networks (TensorFlow / Keras)

**Paper:** Very Deep Convolutional Networks for Large-Scale Image Recognition
**Authors:** Karen Simonyan, Andrew Zisserman
**Conference:** ICLR 2015

---

## Overview

VGG19 is the deepest model in the VGG family, with **19 weight layers** (16 conv + 3 FC).
Compared to VGG16, blocks 3, 4, and 5 each gain one extra 3×3 convolution layer.
The added depth provides a slightly larger receptive field per stage but offers
diminishing accuracy gains over VGG16 (~71.1% vs ~71.3% Top-1 on ImageNet).

VGG19 is widely used as a **perceptual loss network** in style transfer, super-resolution,
and image generation tasks, because its deep feature representations capture both
low-level textures and high-level semantics.

---

## VGG19 vs VGG16

| Block | VGG16 convs | VGG19 convs | Channels |
|-------|-------------|-------------|----------|
| Block 1 | 2 | 2 | 64 |
| Block 2 | 2 | 2 | 128 |
| Block 3 | 3 | **4** | 256 |
| Block 4 | 3 | **4** | 512 |
| Block 5 | 3 | **4** | 512 |
| **Total** | **13 conv** | **16 conv** | — |

---

## Architecture

```
Input (224×224×3)
│
├── Block 1 : Conv(64)×2  + MaxPool(2×2)   →  64 × 112×112
├── Block 2 : Conv(128)×2 + MaxPool(2×2)   → 128 ×  56×56
├── Block 3 : Conv(256)×4 + MaxPool(2×2)   → 256 ×  28×28   ← 4 convs
├── Block 4 : Conv(512)×4 + MaxPool(2×2)   → 512 ×  14×14   ← 4 convs
├── Block 5 : Conv(512)×4 + MaxPool(2×2)   → 512 ×   7×7    ← 4 convs
│
├── Flatten                                 → 25,088
├── Dense(4096) + ReLU + Dropout(0.5)
├── Dense(4096) + ReLU + Dropout(0.5)
└── Dense(num_classes) + Softmax
```

All conv layers: 3×3 kernel, padding='same', ReLU activation.
All max-pools : 2×2 window, stride=2.

---

## Key Stats

| Property | Value |
|----------|-------|
| Parameters | ~143.7M |
| Top-1 (ImageNet) | ~71.1% |
| Top-5 (ImageNet) | ~89.8% |
| Input size | 224×224 |
| Framework | TensorFlow / Keras |

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
from tensorflow.keras.applications.vgg19 import preprocess_input

base_model = keras.applications.VGG19(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3),
)
x       = layers.GlobalAveragePooling2D()(base_model.output)
x       = layers.Dense(512, activation='relu')(x)
x       = layers.Dropout(0.5)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model   = keras.Model(inputs=base_model.input, outputs=outputs)

# IMPORTANT: VGG19 preprocessing subtracts BGR channel means (NOT /255)
datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input
)
```

### Two-Phase Fine-Tuning

```python
# Phase 1 — freeze all conv blocks
base_model.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-3), ...)
model.fit(train_gen, epochs=10, ...)

# Phase 2 — unfreeze block4 + block5 only
base_model.trainable = True
frozen = {'block1_conv1','block1_conv2','block2_conv1','block2_conv2',
          'block3_conv1','block3_conv2','block3_conv3','block3_conv4'}
for layer in base_model.layers:
    if layer.name in frozen:
        layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-5), ...)
model.fit(train_gen, initial_epoch=10, epochs=30, ...)
```

---

## Common Use: Perceptual Loss

VGG19 is the standard backbone for perceptual loss in generative tasks:

```python
vgg = keras.applications.VGG19(weights='imagenet', include_top=False)
feature_extractor = keras.Model(
    inputs=vgg.input,
    outputs=vgg.get_layer('block4_conv2').output   # deep semantic features
)
perceptual_loss = tf.reduce_mean(
    tf.abs(feature_extractor(y_true) - feature_extractor(y_pred))
)
```

---

## Folder Structure

```
VGG19/
├── README.md
├── NoteBook/
│   └── vgg19.ipynb              — 17-cell notebook (arch + train + ROC AUC)
├── Python Scripts/
│   ├── vgg19.py                 — build_vgg19() from scratch
│   ├── train.py                 — Adam + ReduceLROnPlateau training loop
│   ├── inference.py             — top-K single-image prediction
│   └── How to run.txt
└── Using Weight File/
    ├── load_pretrained.py       — load Keras Applications VGG19
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
