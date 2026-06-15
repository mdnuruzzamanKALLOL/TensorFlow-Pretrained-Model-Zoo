import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _bottleneck(x, filters, stride=1, name=None):
    """
    ResNet Bottleneck block (post-activation): Conv1x1 -> Conv3x3 -> Conv1x1 + residual.
    Projection shortcut applied when stride != 1 or channels mismatch.
    expansion = 4: output channels = filters * 4.
    """
    shortcut     = x
    out_channels = filters * 4

    if stride != 1 or x.shape[-1] != out_channels:
        shortcut = layers.Conv2D(
            out_channels, 1, strides=stride, use_bias=False,
            name=f'{name}_0_conv' if name else None)(x)
        shortcut = layers.BatchNormalization(
            name=f'{name}_0_bn' if name else None)(shortcut)

    x = layers.Conv2D(filters, 1, use_bias=False,
                      name=f'{name}_1_conv' if name else None)(x)
    x = layers.BatchNormalization(name=f'{name}_1_bn' if name else None)(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False,
                      name=f'{name}_2_conv' if name else None)(x)
    x = layers.BatchNormalization(name=f'{name}_2_bn' if name else None)(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(out_channels, 1, use_bias=False,
                      name=f'{name}_3_conv' if name else None)(x)
    x = layers.BatchNormalization(name=f'{name}_3_bn' if name else None)(x)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


def build_resnet152(num_classes=1000, input_shape=(224, 224, 3)):
    """
    ResNet-152 — Deep Residual Learning for Image Recognition.
    Paper: He et al., CVPR 2016.

    Deepest standard ResNet variant. Compared to ResNet-101:
      Stage 2 expands from  4 -> 8  blocks
      Stage 3 expands from 23 -> 36 blocks

    Stage depths: 3-8-36-3

      Stem    : Conv7x7/2 + BN + ReLU + MaxPool3x3/2  ->   64 x 56x56
      Stage 1 : 3  x Bottleneck(64)   stride=1         ->  256 x 56x56
      Stage 2 : 8  x Bottleneck(128)  stride=2         ->  512 x 28x28
      Stage 3 : 36 x Bottleneck(256)  stride=2         -> 1024 x 14x14
      Stage 4 : 3  x Bottleneck(512)  stride=2         -> 2048 x  7x7
      Head    : GlobalAvgPool -> Dense(num_classes)
    """
    inputs = keras.Input(shape=input_shape)

    # ── Stem ──
    x = layers.ZeroPadding2D(3, name='conv1_pad')(inputs)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1_conv')(x)
    x = layers.BatchNormalization(name='conv1_bn')(x)
    x = layers.ReLU(name='conv1_relu')(x)
    x = layers.ZeroPadding2D(1, name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)                # 56x56

    # ── Stage 1: 3 blocks, 64 filters ──
    x = _bottleneck(x, 64,  stride=1, name='conv2_block1')
    x = _bottleneck(x, 64,  stride=1, name='conv2_block2')
    x = _bottleneck(x, 64,  stride=1, name='conv2_block3')

    # ── Stage 2: 8 blocks, 128 filters, stride=2 ──
    x = _bottleneck(x, 128, stride=2, name='conv3_block1')   # 512 ch, 28x28
    for i in range(2, 9):                                     # blocks 2-8
        x = _bottleneck(x, 128, stride=1, name=f'conv3_block{i}')

    # ── Stage 3: 36 blocks, 256 filters, stride=2 ──
    x = _bottleneck(x, 256, stride=2, name='conv4_block1')   # 1024 ch, 14x14
    for i in range(2, 37):                                    # blocks 2-36
        x = _bottleneck(x, 256, stride=1, name=f'conv4_block{i}')

    # ── Stage 4: 3 blocks, 512 filters, stride=2 ──
    x = _bottleneck(x, 512, stride=2, name='conv5_block1')   # 2048 ch, 7x7
    x = _bottleneck(x, 512, stride=1, name='conv5_block2')
    x = _bottleneck(x, 512, stride=1, name='conv5_block3')

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    return keras.Model(inputs=inputs, outputs=outputs, name='resnet152')
