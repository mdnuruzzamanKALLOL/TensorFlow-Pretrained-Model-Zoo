# DenseNet121

**Framework:** TensorFlow / Keras  
**Task:** Image Classification  
**Input:** 224 x 224 x 3  
**Parameters:** ~8.1M  

## Architecture

DenseNet121 is a densely connected convolutional network where each layer
receives feature maps from **all preceding layers** via concatenation,
promoting feature reuse and alleviating vanishing-gradient issues.

### Key specifications

| Component       | Value |
|-----------------|-------|
| Growth rate (k) | 32 |
| Dense blocks    | 4 blocks — [6, 12, 24, 16] layers |
| Compression (θ) | 0.5 |
| Bottleneck      | 4k = 128 filters (Conv 1×1) |
| Stem            | Conv 7×7 (64 ch, stride 2) + BN + ReLU + MaxPool 3×3 |
| Head            | BN + ReLU + GAP + Dense(softmax) |

### Channel progression

| Stage | Output |
|-------|--------|
| Stem (Conv7×7 + MaxPool) | 64 ch, 56×56 |
| Dense Block 1 (6 layers) | 256 ch → Transition → 128 ch, 28×28 |
| Dense Block 2 (12 layers)| 512 ch → Transition → 256 ch, 14×14 |
| Dense Block 3 (24 layers)| 1024 ch → Transition → 512 ch, 7×7  |
| Dense Block 4 (16 layers)| 1024 ch, 7×7 |

## Files

```
DenseNet121/
├── NoteBook/
│   └── DenseNet121.ipynb          # 17-cell interactive walkthrough
├── Python Scripts/
│   ├── DenseNet121.py             # Architecture from scratch
│   ├── train.py               # Training loop (from scratch)
│   ├── inference.py           # Single-image prediction
│   └── How to run.txt
└── Using Weight File/
    ├── load_pretrained.py     # Predict with ImageNet weights
    ├── feature_extraction.py  # Phase 1: frozen base
    ├── fine_tuning.py         # Phase 1+2: progressive unfreeze
    └── How to run.txt
```

## Quick start

### Train from scratch
```bash
cd "Python Scripts"
python train.py
```

### Fine-tune pretrained model
```bash
cd "Using Weight File"
python fine_tuning.py
```

## References

- Huang et al., *Densely Connected Convolutional Networks*, CVPR 2017.  
  arXiv:1608.06993
