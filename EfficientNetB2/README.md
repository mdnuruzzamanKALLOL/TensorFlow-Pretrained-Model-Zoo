# EfficientNetB2

**Framework:** TensorFlow / Keras  
**Task:** Image Classification  
**Input:** 260 x 260 x 3  
**Parameters:** ~9.2M  

## Architecture

EfficientNetB2 is a compound-scaled EfficientNet variant. Every block uses
**MBConv** with variable depthwise kernel (3×3 or 5×5), Squeeze-Excitation
(se_ratio=0.25), and Swish activation. Width, depth, and resolution are
scaled jointly from the B0 baseline.

### Key specifications

| Component       | Value |
|-----------------|-------|
| Stem            | Conv 3×3 (32 ch, stride 2) + BN + Swish |
| Head conv       | Conv 1×1 (1408 ch) + BN + Swish |
| Head            | GAP + Dropout + Dense(softmax) |
| Dropout         | 0.3 |
| SE ratio        | 0.25 (based on pre-expansion channels) |
| Activation      | Swish |

### Stage configuration

| Stage | Filters / Expansion / Stride |
|-------|------------------------------|
| Stage 1 (k=3, n= 2) | in= 32 → out= 16, expand=1, stride=1 |
| Stage 2 (k=3, n= 3) | in= 16 → out= 24, expand=6, stride=2 |
| Stage 3 (k=5, n= 3) | in= 24 → out= 48, expand=6, stride=2 |
| Stage 4 (k=3, n= 4) | in= 48 → out= 88, expand=6, stride=2 |
| Stage 5 (k=5, n= 4) | in= 88 → out=120, expand=6, stride=1 |
| Stage 6 (k=5, n= 5) | in=120 → out=208, expand=6, stride=2 |
| Stage 7 (k=3, n= 2) | in=208 → out=352, expand=6, stride=1 |

## Files

```
EfficientNetB2/
├── NoteBook/
│   └── EfficientNetB2.ipynb          # 17-cell interactive walkthrough
├── Python Scripts/
│   ├── EfficientNetB2.py             # Architecture from scratch
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

- Tan & Le, *EfficientNet: Rethinking Model Scaling for CNNs*, ICML 2019.  
  arXiv:1905.11946
