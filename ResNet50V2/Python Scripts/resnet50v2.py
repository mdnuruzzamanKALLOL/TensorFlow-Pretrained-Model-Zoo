import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _pre_act_bottleneck(x, filters, stride=1, conv_shortcut=False, name=None):
    """
    ResNet50V2 pre-activation bottleneck block.
    BN + ReLU happen BEFORE each convolution (pre-activation / full pre-activation).

    conv_shortcut=True  : projection shortcut — Conv1x1(preact) to match channels
    conv_shortcut=False : identity shortcut (or MaxPool if stride>1)
    """
    preact = layers.BatchNormalization(name=f'{name}_preact_bn')(x)
    preact = layers.Activation('relu', name=f'{name}_preact_relu')(preact)

    if conv_shortcut:
        shortcut = layers.Conv2D(
            filters * 4, 1, strides=stride, use_bias=False,
            name=f'{name}_0_conv')(preact)
    elif stride > 1:
        shortcut = layers.MaxPooling2D(1, strides=stride)(x)
    else:
        shortcut = x

    # 1x1 compress
    x = layers.Conv2D(filters, 1, use_bias=False, name=f'{name}_1_conv')(preact)
    x = layers.BatchNormalization(name=f'{name}_1_bn')(x)
    x = layers.Activation('relu', name=f'{name}_1_relu')(x)

    # 3x3 spatial
    x = layers.ZeroPadding2D(1, name=f'{name}_2_pad')(x)
    x = layers.Conv2D(filters, 3, strides=stride, use_bias=False,
                      name=f'{name}_2_conv')(x)
    x = layers.BatchNormalization(name=f'{name}_2_bn')(x)
    x = layers.Activation('relu', name=f'{name}_2_relu')(x)

    # 1x1 expand (no BN/ReLU after — handled by next block's preact)
    x = layers.Conv2D(filters * 4, 1, use_bias=False, name=f'{name}_3_conv')(x)

    x = layers.Add(name=f'{name}_out')([shortcut, x])
    return x


def build_resnet50v2(num_classes=1000, input_shape=(224, 224, 3)):
    """
    ResNet50V2 — Identity Mappings in Deep Residual Networks.
    Paper: He et al., ECCV 2016.

    Key differences from ResNet50:
      1. Pre-activation blocks: BN + ReLU BEFORE each conv (not after)
      2. Stem conv has NO BN or ReLU (first block's preact handles it)
      3. Final post-activation BN + ReLU before GlobalAvgPool
      4. Shortcut path is a true identity when channels match

    Architecture (pre-activation Bottleneck, expansion=4):
      Stem    : Conv7x7/2 (no BN/ReLU) + MaxPool3x3/2  ->  64 x 56x56
      Stage 1 : 3 x PreActBottleneck(64)  stride=1      -> 256 x 56x56
      Stage 2 : 4 x PreActBottleneck(128) stride=2      -> 512 x 28x28
      Stage 3 : 6 x PreActBottleneck(256) stride=2      ->1024 x 14x14
      Stage 4 : 3 x PreActBottleneck(512) stride=2      ->2048 x  7x7
      Post    : BN + ReLU
      Head    : GlobalAvgPool -> Dense(num_classes)
    """
    inputs = keras.Input(shape=input_shape)

    # ── Stem — no BN/ReLU after conv (first block preact handles it) ──
    x = layers.ZeroPadding2D(3, name='conv1_pad')(inputs)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1_conv')(x)  # 112x112
    x = layers.ZeroPadding2D(1, name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)                # 56x56

    # ── Stage 1: 3 blocks, 64 filters ──
    x = _pre_act_bottleneck(x, 64, stride=1, conv_shortcut=True,  name='conv2_block1')  # 256 ch
    x = _pre_act_bottleneck(x, 64, stride=1, conv_shortcut=False, name='conv2_block2')
    x = _pre_act_bottleneck(x, 64, stride=1, conv_shortcut=False, name='conv2_block3')

    # ── Stage 2: 4 blocks, 128 filters, stride=2 ──
    x = _pre_act_bottleneck(x, 128, stride=2, conv_shortcut=True,  name='conv3_block1')  # 512 ch, 28x28
    x = _pre_act_bottleneck(x, 128, stride=1, conv_shortcut=False, name='conv3_block2')
    x = _pre_act_bottleneck(x, 128, stride=1, conv_shortcut=False, name='conv3_block3')
    x = _pre_act_bottleneck(x, 128, stride=1, conv_shortcut=False, name='conv3_block4')

    # ── Stage 3: 6 blocks, 256 filters, stride=2 ──
    x = _pre_act_bottleneck(x, 256, stride=2, conv_shortcut=True,  name='conv4_block1')  # 1024 ch, 14x14
    x = _pre_act_bottleneck(x, 256, stride=1, conv_shortcut=False, name='conv4_block2')
    x = _pre_act_bottleneck(x, 256, stride=1, conv_shortcut=False, name='conv4_block3')
    x = _pre_act_bottleneck(x, 256, stride=1, conv_shortcut=False, name='conv4_block4')
    x = _pre_act_bottleneck(x, 256, stride=1, conv_shortcut=False, name='conv4_block5')
    x = _pre_act_bottleneck(x, 256, stride=1, conv_shortcut=False, name='conv4_block6')

    # ── Stage 4: 3 blocks, 512 filters, stride=2 ──
    x = _pre_act_bottleneck(x, 512, stride=2, conv_shortcut=True,  name='conv5_block1')  # 2048 ch, 7x7
    x = _pre_act_bottleneck(x, 512, stride=1, conv_shortcut=False, name='conv5_block2')
    x = _pre_act_bottleneck(x, 512, stride=1, conv_shortcut=False, name='conv5_block3')

    # ── Post-activation (last conv block has no trailing BN/ReLU) ──
    x = layers.BatchNormalization(name='post_bn')(x)
    x = layers.Activation('relu', name='post_relu')(x)

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    return keras.Model(inputs=inputs, outputs=outputs, name='resnet50v2')
