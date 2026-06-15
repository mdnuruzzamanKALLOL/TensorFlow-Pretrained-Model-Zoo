# InceptionResNetV2

**Framework:** TensorFlow / Keras
**Input size:** 299 × 299 × 3
**Parameters:** ~55.8 M
**ImageNet Top-1:** ~80.3%

## Architecture

InceptionResNetV2 (Szegedy et al., 2017) combines the Inception module factorisation
with residual connections and **residual scaling** (multiply residual branch by a small
constant before adding) to stabilise training of very deep networks.

| Section | Grid | Blocks | Key idea |
|---------|------|--------|----------|
| Stem | 299→35 | 7 conv/pool | Same as InceptionV3 |
| mixed_5b | 35×35 | ×1 | Inception-A type entry block → 320 ch |
| InceptionResNet-A | 35×35 | ×5 | 3-branch inception + scaled residual (s=0.17) |
| Reduction-A | 35→17 | ×1 | 320 → 1088 ch |
| InceptionResNet-B | 17×17 | ×10 | 1×7 / 7×1 factorised + scaled residual (s=0.1) |
| Reduction-B | 17→8 | ×1 | 1088 → 2080 ch |
| InceptionResNet-C | 8×8 | ×5 | 1×3 / 3×1 factorised + scaled residual (s=0.2) |
| conv_7b | 8×8 | ×1 | 1×1 projection 2080 → 1536 |
| Head | — | GAP + Dense | Softmax classifier |

**Residual scaling:** each inception branch output is multiplied by a scale factor
before the residual add. This avoids activation explosion in deep residual networks
and is the key algorithmic contribution over Inception-v4.

## Preprocessing

```python
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
# Maps [0, 255] -> [-1, 1]  (tf mode: x / 127.5 - 1.0)
```

## File Structure

```
InceptionResNetV2/
├── NoteBook/
│   └── InceptionResNetV2.ipynb    # 17-cell end-to-end notebook
├── Python Scripts/
│   ├── InceptionResNetV2.py       # Architecture from scratch
│   ├── train.py                   # Training script
│   ├── inference.py               # Single-image inference
│   └── How to run.txt
├── Using Weight File/
│   ├── load_pretrained.py         # Transfer learning head
│   ├── feature_extraction.py      # Extract 1536-d feature vectors
│   ├── fine_tuning.py             # Two-phase fine-tuning
│   └── How to run.txt
└── README.md
```

## Quick Start

```python
# From scratch
from InceptionResNetV2 import build_inception_resnet_v2
model = build_inception_resnet_v2(num_classes=10)

# Pretrained
from tensorflow.keras.applications import InceptionResNetV2
base = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(299,299,3))
```

## Fine-Tuning Strategy

| Phase | Layers | LR | Epochs |
|-------|--------|----|--------|
| 1 | Head only (base frozen) | 1e-3 | 10 |
| 2 | After mixed_7a (block8 × 5 + conv_7b) | 1e-5 | 20 |

Phase 2 targets the last Reduction-B output onwards: all 5 InceptionResNet-C blocks
plus the final 1×1 projection conv (`conv_7b`), unfreezing ~30% of the model.
