<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:FF6F00,50:FF8C00,100:FFA500&height=200&section=header&text=TensorFlow%20Pretrained%20Model%20Zoo&fontSize=38&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=38%20Production-Ready%20Models%20%7C%20Notebooks%20%7C%20Scripts%20%7C%20Fine-Tuning%20Guides&descAlignY=62&descColor=ffe0b2&descSize=16"/>

<br/>

<img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
<img src="https://img.shields.io/badge/Keras-Integrated-D00000?style=for-the-badge&logo=keras&logoColor=white"/>
<img src="https://img.shields.io/badge/Models-38-4ecdc4?style=for-the-badge&logo=google&logoColor=white"/>
<img src="https://img.shields.io/badge/ImageNet-Pretrained-7b2ff7?style=for-the-badge&logo=imagenet&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-success?style=for-the-badge"/>

</div>

---

## What's Inside

Every model folder contains **three ready-to-run resources**:

<div align="center">

| Asset | Description |
|-------|-------------|
| `NoteBook/` | Interactive Jupyter notebook — architecture walkthrough, training & evaluation |
| `Python Scripts/` | Standalone `.py` files — build from scratch, train loop, single-image inference |
| `Using Weight File/` | Scripts to load ImageNet weights, feature-extract (frozen) and fine-tune (progressive unfreeze) |
| `README.md` | Architecture diagram, stage configs, key specs and references |

</div>

---

<h2 align="center">Model Table</h2>
<div align="center">

### ConvNeXt Family

| Model | Params | Input | Notebook | Script | Weights |
|:------|:------:|:-----:|:--------:|:------:|:-------:|
| [ConvNeXtTiny](ConvNeXtTiny/) | 28 M | 224² | [📓](ConvNeXtTiny/NoteBook/) | [🐍](ConvNeXtTiny/Python%20Scripts/) | [⚖️](ConvNeXtTiny/Using%20Weight%20File/) |
| [ConvNeXtSmall](ConvNeXtSmall/) | 50 M | 224² | [📓](ConvNeXtSmall/NoteBook/) | [🐍](ConvNeXtSmall/Python%20Scripts/) | [⚖️](ConvNeXtSmall/Using%20Weight%20File/) |
| [ConvNeXtBase](ConvNeXtBase/) | 89 M | 224² | [📓](ConvNeXtBase/NoteBook/) | [🐍](ConvNeXtBase/Python%20Scripts/) | [⚖️](ConvNeXtBase/Using%20Weight%20File/) |
| [ConvNeXtLarge](ConvNeXtLarge/) | 198 M | 224² | [📓](ConvNeXtLarge/NoteBook/) | [🐍](ConvNeXtLarge/Python%20Scripts/) | [⚖️](ConvNeXtLarge/Using%20Weight%20File/) |
| [ConvNeXtXLarge](ConvNeXtXLarge/) | 350 M | 224² | [📓](ConvNeXtXLarge/NoteBook/) | [🐍](ConvNeXtXLarge/Python%20Scripts/) | [⚖️](ConvNeXtXLarge/Using%20Weight%20File/) |

</div>

<div align="center">

### DenseNet Family

| Model | Params | Input | Notebook | Script | Weights |
|:------|:------:|:-----:|:--------:|:------:|:-------:|
| [DenseNet121](DenseNet121/) | 8 M | 224² | [📓](DenseNet121/NoteBook/) | [🐍](DenseNet121/Python%20Scripts/) | [⚖️](DenseNet121/Using%20Weight%20File/) |
| [DenseNet169](DenseNet169/) | 14 M | 224² | [📓](DenseNet169/NoteBook/) | [🐍](DenseNet169/Python%20Scripts/) | [⚖️](DenseNet169/Using%20Weight%20File/) |
| [DenseNet201](DenseNet201/) | 20 M | 224² | [📓](DenseNet201/NoteBook/) | [🐍](DenseNet201/Python%20Scripts/) | [⚖️](DenseNet201/Using%20Weight%20File/) |

</div>

<div align="center">

### EfficientNet Family

| Model | Params | Input | Notebook | Script | Weights |
|:------|:------:|:-----:|:--------:|:------:|:-------:|
| [EfficientNetB0](EfficientNetB0/) | 5.3 M | 224² | [📓](EfficientNetB0/NoteBook/) | [🐍](EfficientNetB0/Python%20Scripts/) | [⚖️](EfficientNetB0/Using%20Weight%20File/) |
| [EfficientNetB1](EfficientNetB1/) | 7.8 M | 240² | [📓](EfficientNetB1/NoteBook/) | [🐍](EfficientNetB1/Python%20Scripts/) | [⚖️](EfficientNetB1/Using%20Weight%20File/) |
| [EfficientNetB2](EfficientNetB2/) | 9.1 M | 260² | [📓](EfficientNetB2/NoteBook/) | [🐍](EfficientNetB2/Python%20Scripts/) | [⚖️](EfficientNetB2/Using%20Weight%20File/) |
| [EfficientNetB3](EfficientNetB3/) | 12 M | 300² | [📓](EfficientNetB3/NoteBook/) | [🐍](EfficientNetB3/Python%20Scripts/) | [⚖️](EfficientNetB3/Using%20Weight%20File/) |
| [EfficientNetB4](EfficientNetB4/) | 19 M | 380² | [📓](EfficientNetB4/NoteBook/) | [🐍](EfficientNetB4/Python%20Scripts/) | [⚖️](EfficientNetB4/Using%20Weight%20File/) |
| [EfficientNetB5](EfficientNetB5/) | 30 M | 456² | [📓](EfficientNetB5/NoteBook/) | [🐍](EfficientNetB5/Python%20Scripts/) | [⚖️](EfficientNetB5/Using%20Weight%20File/) |
| [EfficientNetB6](EfficientNetB6/) | 43 M | 528² | [📓](EfficientNetB6/NoteBook/) | [🐍](EfficientNetB6/Python%20Scripts/) | [⚖️](EfficientNetB6/Using%20Weight%20File/) |
| [EfficientNetB7](EfficientNetB7/) | 66 M | 600² | [📓](EfficientNetB7/NoteBook/) | [🐍](EfficientNetB7/Python%20Scripts/) | [⚖️](EfficientNetB7/Using%20Weight%20File/) |

</div>

<div align="center">

### EfficientNetV2 Family

| Model | Params | Input | Notebook | Script | Weights |
|:------|:------:|:-----:|:--------:|:------:|:-------:|
| [EfficientNetV2B0](EfficientNetV2B0/) | 7.1 M | 224² | [📓](EfficientNetV2B0/NoteBook/) | [🐍](EfficientNetV2B0/Python%20Scripts/) | [⚖️](EfficientNetV2B0/Using%20Weight%20File/) |
| [EfficientNetV2B1](EfficientNetV2B1/) | 8.1 M | 240² | [📓](EfficientNetV2B1/NoteBook/) | [🐍](EfficientNetV2B1/Python%20Scripts/) | [⚖️](EfficientNetV2B1/Using%20Weight%20File/) |
| [EfficientNetV2B2](EfficientNetV2B2/) | 10.1 M | 260² | [📓](EfficientNetV2B2/NoteBook/) | [🐍](EfficientNetV2B2/Python%20Scripts/) | [⚖️](EfficientNetV2B2/Using%20Weight%20File/) |
| [EfficientNetV2B3](EfficientNetV2B3/) | 14.4 M | 300² | [📓](EfficientNetV2B3/NoteBook/) | [🐍](EfficientNetV2B3/Python%20Scripts/) | [⚖️](EfficientNetV2B3/Using%20Weight%20File/) |
| [EfficientNetV2S](EfficientNetV2S/) | 21.5 M | 384² | [📓](EfficientNetV2S/NoteBook/) | [🐍](EfficientNetV2S/Python%20Scripts/) | [⚖️](EfficientNetV2S/Using%20Weight%20File/) |
| [EfficientNetV2M](EfficientNetV2M/) | 54.1 M | 480² | [📓](EfficientNetV2M/NoteBook/) | [🐍](EfficientNetV2M/Python%20Scripts/) | [⚖️](EfficientNetV2M/Using%20Weight%20File/) |
| [EfficientNetV2L](EfficientNetV2L/) | 119 M | 480² | [📓](EfficientNetV2L/NoteBook/) | [🐍](EfficientNetV2L/Python%20Scripts/) | [⚖️](EfficientNetV2L/Using%20Weight%20File/) |

</div>

<div align="center">

### Inception Family

| Model | Params | Input | Notebook | Script | Weights |
|:------|:------:|:-----:|:--------:|:------:|:-------:|
| [InceptionV3](InceptionV3/) | 23.8 M | 299² | [📓](InceptionV3/NoteBook/) | [🐍](InceptionV3/Python%20Scripts/) | [⚖️](InceptionV3/Using%20Weight%20File/) |
| [InceptionResNetV2](InceptionResNetV2/) | 55.8 M | 299² | [📓](InceptionResNetV2/NoteBook/) | [🐍](InceptionResNetV2/Python%20Scripts/) | [⚖️](InceptionResNetV2/Using%20Weight%20File/) |

</div>

<div align="center">

### MobileNet Family

| Model | Params | Input | Notebook | Script | Weights |
|:------|:------:|:-----:|:--------:|:------:|:-------:|
| [MobileNet](MobileNet/) | 4.2 M | 224² | [📓](MobileNet/NoteBook/) | [🐍](MobileNet/Python%20Scripts/) | [⚖️](MobileNet/Using%20Weight%20File/) |
| [MobileNetV2](MobileNetV2/) | 3.4 M | 224² | [📓](MobileNetV2/NoteBook/) | [🐍](MobileNetV2/Python%20Scripts/) | [⚖️](MobileNetV2/Using%20Weight%20File/) |

</div>

<div align="center">

### NASNet Family

| Model | Params | Input | Notebook | Script | Weights |
|:------|:------:|:-----:|:--------:|:------:|:-------:|
| [NASNetMobile](NASNetMobile/) | 5.3 M | 224² | [📓](NASNetMobile/NoteBook/) | [🐍](NASNetMobile/Python%20Scripts/) | [⚖️](NASNetMobile/Using%20Weight%20File/) |
| [NASNetLarge](NASNetLarge/) | 88.9 M | 331² | [📓](NASNetLarge/NoteBook/) | [🐍](NASNetLarge/Python%20Scripts/) | [⚖️](NASNetLarge/Using%20Weight%20File/) |

</div>

<div align="center">

### ResNet Family

| Model | Params | Input | Notebook | Script | Weights |
|:------|:------:|:-----:|:--------:|:------:|:-------:|
| [ResNet50](ResNet50/) | 25.6 M | 224² | [📓](ResNet50/NoteBook/) | [🐍](ResNet50/Python%20Scripts/) | [⚖️](ResNet50/Using%20Weight%20File/) |
| [ResNet50V2](ResNet50V2/) | 25.6 M | 224² | [📓](ResNet50V2/NoteBook/) | [🐍](ResNet50V2/Python%20Scripts/) | [⚖️](ResNet50V2/Using%20Weight%20File/) |
| [ResNet101](ResNet101/) | 44.7 M | 224² | [📓](ResNet101/NoteBook/) | [🐍](ResNet101/Python%20Scripts/) | [⚖️](ResNet101/Using%20Weight%20File/) |
| [ResNet101V2](ResNet101V2/) | 44.7 M | 224² | [📓](ResNet101V2/NoteBook/) | [🐍](ResNet101V2/Python%20Scripts/) | [⚖️](ResNet101V2/Using%20Weight%20File/) |
| [ResNet152](ResNet152/) | 60.2 M | 224² | [📓](ResNet152/NoteBook/) | [🐍](ResNet152/Python%20Scripts/) | [⚖️](ResNet152/Using%20Weight%20File/) |
| [ResNet152V2](ResNet152V2/) | 60.2 M | 224² | [📓](ResNet152V2/NoteBook/) | [🐍](ResNet152V2/Python%20Scripts/) | [⚖️](ResNet152V2/Using%20Weight%20File/) |

</div>

<div align="center">

### VGG Family

| Model | Params | Input | Notebook | Script | Weights |
|:------|:------:|:-----:|:--------:|:------:|:-------:|
| [VGG16](VGG16/) | 138 M | 224² | [📓](VGG16/NoteBook/) | [🐍](VGG16/Python%20Scripts/) | [⚖️](VGG16/Using%20Weight%20File/) |
| [VGG19](VGG19/) | 143.7 M | 224² | [📓](VGG19/NoteBook/) | [🐍](VGG19/Python%20Scripts/) | [⚖️](VGG19/Using%20Weight%20File/) |

</div>

<div align="center">

### Xception

| Model | Params | Input | Notebook | Script | Weights |
|:------|:------:|:-----:|:--------:|:------:|:-------:|
| [Xception](Xception/) | 22.9 M | 299² | [📓](Xception/NoteBook/) | [🐍](Xception/Python%20Scripts/) | [⚖️](Xception/Using%20Weight%20File/) |

</div>

---

## Quick Start

### 1. Train from scratch

```bash
cd "EfficientNetB0/Python Scripts"
python train.py
```

### 2. Fine-tune with pretrained ImageNet weights

```bash
cd "ResNet50/Using Weight File"
python fine_tuning.py
```

### 3. Feature extraction (frozen base)

```bash
cd "DenseNet121/Using Weight File"
python feature_extraction.py
```

---

<div align="center">

## Families at a Glance

| Family | Models | Best Use |
|--------|:------:|----------|
| **ConvNeXt** | 5 | Modern CNN, top accuracy, ViT-competitive |
| **EfficientNet** | 8 | Compound-scaled, accuracy/efficiency sweet spot |
| **EfficientNetV2** | 7 | Faster training with Fused-MBConv |
| **ResNet** | 6 | Backbone standard, easy fine-tuning |
| **DenseNet** | 3 | Dense skip connections, feature reuse |
| **Inception** | 2 | Multi-scale features, medical imaging |
| **MobileNet** | 2 | Edge deployment, low latency |
| **NASNet** | 2 | NAS-designed, high accuracy |
| **VGG** | 2 | Simple baseline, style transfer |
| **Xception** | 1 | Depthwise separable, Inception evolution |

</div>

---

<div align="center">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.TensorFlow-Pretrained-Model-Zoo&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>

<div align="center">
<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:FFA500,50:FF8C00,100:FF6F00&height=120&section=footer&text=Happy%20Training!%20%F0%9F%94%A5&fontSize=24&fontColor=ffffff&animation=fadeIn&fontAlignY=65"/>
</div>
