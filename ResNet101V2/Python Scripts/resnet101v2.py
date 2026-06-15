import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _pre_act_bottleneck(x, filters, stride=1, conv_shortcut=False, name=None):
    """
    ResNetV2 Pre-activation Bottleneck block.
    BN + ReLU applied BEFORE each convolution (Identity Mappings in Deep Residual Networks).
    expansion = 4: output channels = filters * 4.

    conv_shortcut=True  : projection shortcut uses preact (after BN+ReLU)
    conv_shortcut=False : identity shortcut; if stride>1 use MaxPool on raw x
    """
    preact = layers.BatchNormalization(name=f'{name}_preact_bn' if name else None)(x)
    preact = layers.ReLU(name=f'{name}_preact_relu' if name else None)(preact)

    if conv_shortcut:
        shortcut = layers.Conv2D(
            filters * 4, 1, strides=stride, use_bias=False,
            name=f'{name}_0_conv' if name else None)(preact)
    elif stride > 1:
        shortcut = layers.MaxPooling2D(1, strides=stride)(x)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, use_bias=False,
                      name=f'{name}_1_conv' if name else None)(preact)
    x = layers.BatchNormalization(name=f'{name}_1_bn' if name else None)(x)
    x = layers.ReLU(name=f'{name}_1_relu' if name else None)(x)

    x = layers.ZeroPadding2D(1, name=f'{name}_2_pad' if name else None)(x)
    x = layers.Conv2D(filters, 3, strides=stride, use_bias=False,
                      name=f'{name}_2_conv' if name else None)(x)
    x = layers.BatchNormalization(name=f'{name}_2_bn' if name else None)(x)
    x = layers.ReLU(name=f'{name}_2_relu' if name else None)(x)

    x = layers.Conv2D(filters * 4, 1, use_bias=False,
                      name=f'{name}_3_conv' if name else None)(x)  # NO BN/ReLU here

    x = layers.Add()([shortcut, x])
    return x


def build_resnet101v2(num_classes=1000, input_shape=(224, 224, 3)):
    """
    ResNet-101 V2 — Identity Mappings in Deep Residual Networks.
    Paper: He et al., ECCV 2016.

    Identical to ResNet-50 V2 except Stage 3 has 23 pre-activation bottleneck
    blocks instead of 6.
    Stage depths: 3-4-23-3  (ResNet-50 V2: 3-4-6-3)

    Key V2 differences vs V1:
      - BN + ReLU BEFORE each conv (pre-activation)
      - Stem: Conv7x7/2 only — NO BN/ReLU after stem
      - Post-processing: BN + ReLU added after final stage (before GAP)
      - Identity shortcuts are truly identity (no BN on shortcut path)

      Stem    : Conv7x7/2  (no BN/ReLU)  + MaxPool3x3/2  ->   64 x 56x56
      Stage 1 : 3  x PreActBottleneck(64)  stride=1       ->  256 x 56x56
      Stage 2 : 4  x PreActBottleneck(128) stride=2       ->  512 x 28x28
      Stage 3 : 23 x PreActBottleneck(256) stride=2       -> 1024 x 14x14
      Stage 4 : 3  x PreActBottleneck(512) stride=2       -> 2048 x  7x7
      Post    : BN + ReLU
      Head    : GlobalAvgPool -> Dense(num_classes)
    """
    inputs = keras.Input(shape=input_shape)

    # ── Stem (no BN/ReLU after Conv — V2 difference) ──
    x = layers.ZeroPadding2D(3, name='conv1_pad')(inputs)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1_conv')(x)  # 112x112
    x = layers.ZeroPadding2D(1, name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)                # 56x56

    # ── Stage 1: 3 blocks, 64 filters ──
    x = _pre_act_bottleneck(x, 64,  stride=1, conv_shortcut=True,  name='conv2_block1')
    x = _pre_act_bottleneck(x, 64,  stride=1, conv_shortcut=False, name='conv2_block2')
    x = _pre_act_bottleneck(x, 64,  stride=1, conv_shortcut=False, name='conv2_block3')

    # ── Stage 2: 4 blocks, 128 filters, stride=2 ──
    x = _pre_act_bottleneck(x, 128, stride=2, conv_shortcut=True,  name='conv3_block1')
    x = _pre_act_bottleneck(x, 128, stride=1, conv_shortcut=False, name='conv3_block2')
    x = _pre_act_bottleneck(x, 128, stride=1, conv_shortcut=False, name='conv3_block3')
    x = _pre_act_bottleneck(x, 128, stride=1, conv_shortcut=False, name='conv3_block4')

    # ── Stage 3: 23 blocks, 256 filters, stride=2  (key difference vs ResNet-50 V2) ──
    x = _pre_act_bottleneck(x, 256, stride=2, conv_shortcut=True,  name='conv4_block1')
    for i in range(2, 24):                                          # blocks 2-23
        x = _pre_act_bottleneck(x, 256, stride=1, conv_shortcut=False, name=f'conv4_block{i}')

    # ── Stage 4: 3 blocks, 512 filters, stride=2 ──
    x = _pre_act_bottleneck(x, 512, stride=2, conv_shortcut=True,  name='conv5_block1')
    x = _pre_act_bottleneck(x, 512, stride=1, conv_shortcut=False, name='conv5_block2')
    x = _pre_act_bottleneck(x, 512, stride=1, conv_shortcut=False, name='conv5_block3')

    # ── Post BN + ReLU (required because last block has no post-activation) ──
    x = layers.BatchNormalization(name='post_bn')(x)
    x = layers.ReLU(name='post_relu')(x)

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    return keras.Model(inputs=inputs, outputs=outputs, name='resnet101v2')
