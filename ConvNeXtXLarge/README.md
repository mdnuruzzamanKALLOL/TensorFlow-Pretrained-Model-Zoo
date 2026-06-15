# ConvNeXtXLarge

**Framework:** TensorFlow / Keras  
**Task:** Image Classification  
**Input:** 224 x 224 x 3  
**Parameters:** ~350M  

## Architecture

ConvNeXtXLarge is a pure ConvNet that adopts key design choices from Vision
Transformers: large-kernel depthwise convolution (7×7), LayerNormalization,
GELU activation, inverted bottleneck MLP, and per-channel LayerScale.

### Key specifications

| Component          | Value |
|--------------------|-------|
| Channel dims       | [256, 512, 1024, 2048] |
| Block depths       | [3, 3, 27, 3] |
| Stem               | Conv 4×4, stride 4 (patch embed) + LN |
| ConvNeXt block     | DWConv 7×7 → LN → Dense(4d) → GELU → Dense(d) → LayerScale |
| Downsampling       | LN + Conv 2×2, stride 2 (between stages) |
| Head               | GAP → LN → Dense(softmax) |
| Activation         | GELU |
| Normalisation      | LayerNorm (no BatchNorm) |

### Stage progression (224×224 input)

| Stage | Output |
|-------|--------|
| Stem (Conv 4×4, stride 4) | 256 ch, 56×56 |
| Stage 1 (3 blocks)    | 256 ch, 56×56 |
| Downsample + Stage 2 (3 blocks) | 512 ch, 28×28 |
| Downsample + Stage 3 (27 blocks) | 1024 ch, 14×14 |
| Downsample + Stage 4 (3 blocks) | 2048 ch,  7×7  |

## Files

```
ConvNeXtXLarge/
├── NoteBook/
│   └── ConvNeXtXLarge.ipynb          # 17-cell interactive walkthrough
├── Python Scripts/
│   ├── ConvNeXtXLarge.py             # Architecture from scratch
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

- Liu et al., *A ConvNet for the 2020s*, CVPR 2022.  
  arXiv:2201.03545
