# InceptionV3

**Framework:** TensorFlow / Keras
**Input size:** 299 × 299 × 3
**Parameters:** ~23.8 M
**ImageNet Top-1:** ~77.9%

## Architecture

InceptionV3 (Szegedy et al., 2016 — *Rethinking the Inception Architecture*) introduces two key factorisation tricks:

| Section | Grid | Blocks | Key idea |
|---------|------|--------|----------|
| Stem | 299→35 | 7 conv/pool | Strided convs replace early pooling |
| Inception-A | 35×35 | ×3 | 5×5 factorised into two 3×3 |
| Reduction-A | 35→17 | ×1 | Strided conv + pool concat |
| Inception-B | 17×17 | ×4 | n×n factorised into 1×n + n×1 |
| Reduction-B | 17→8 | ×1 | Strided conv + pool concat |
| Inception-C | 8×8 | ×2 | Parallel 1×3 and 3×1 branches |
| Head | — | GAP + Dense | Softmax classifier |

Each Inception-A block outputs 256/288 channels; B blocks output 768; C blocks output 2048.

## Preprocessing

```python
from tensorflow.keras.applications.inception_v3 import preprocess_input
# Maps [0, 255] -> [-1, 1]  (tf mode: x / 127.5 - 1.0)
```

## File Structure

```
InceptionV3/
├── NoteBook/
│   └── InceptionV3.ipynb          # 17-cell end-to-end notebook
├── Python Scripts/
│   ├── InceptionV3.py             # Architecture from scratch
│   ├── train.py                   # Training script
│   ├── inference.py               # Single-image inference
│   └── How to run.txt
├── Using Weight File/
│   ├── load_pretrained.py         # Feature extraction head
│   ├── feature_extraction.py      # Extract 2048-d feature vectors
│   ├── fine_tuning.py             # Two-phase fine-tuning
│   └── How to run.txt
└── README.md
```

## Quick Start

```python
# From scratch
from InceptionV3 import build_inceptionv3
model = build_inceptionv3(num_classes=10)

# Pretrained
from tensorflow.keras.applications import InceptionV3
base = InceptionV3(weights="imagenet", include_top=False, input_shape=(299,299,3))
```

## Fine-Tuning Strategy

| Phase | Layers | LR | Epochs |
|-------|--------|----|--------|
| 1 | Head only (base frozen) | 1e-3 | 10 |
| 2 | mixed7 + mixed8 + mixed9 + mixed10 | 1e-5 | 20 |

Phase 2 unfreezes the last ~35% of the backbone starting from the `mixed7` concat layer.
