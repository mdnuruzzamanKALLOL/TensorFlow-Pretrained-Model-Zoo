import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _dw_block(x, filters, stride, block_id, alpha=1.0):
    """Depthwise Separable Convolution block.
    DWConv3x3(stride) + BN + ReLU6  then  PWConv1x1 + BN + ReLU6.
    alpha: width multiplier (scales all filter counts).
    """
    filters = int(filters * alpha)
    x = layers.DepthwiseConv2D(
        3, strides=stride, padding='same', use_bias=False,
        name=f'conv_dw_{block_id}')(x)
    x = layers.BatchNormalization(name=f'conv_dw_{block_id}_bn')(x)
    x = layers.ReLU(6., name=f'conv_dw_{block_id}_relu')(x)

    x = layers.Conv2D(
        filters, 1, padding='same', use_bias=False,
        name=f'conv_pw_{block_id}')(x)
    x = layers.BatchNormalization(name=f'conv_pw_{block_id}_bn')(x)
    x = layers.ReLU(6., name=f'conv_pw_{block_id}_relu')(x)
    return x


def build_mobilenet(num_classes=1000, input_shape=(224, 224, 3), alpha=1.0):
    """
    MobileNet — Efficient Convolutional Neural Networks for Mobile Applications.
    Paper: MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
           Applications (Howard et al., 2017).

    Replaces standard convolutions with Depthwise Separable Convolutions:
      Standard Conv(k,k,c_in,c_out) -> DWConv(k,k,c_in) + PWConv(1,1,c_in,c_out)
      Parameter reduction: ~8-9x  |  FLOP reduction: ~8-9x

    alpha: width multiplier (scales all filter counts, default=1.0)
    Resolution multiplier rho: controlled via input_shape.

    Architecture:
      Stem       : Conv3x3/2  (32 ch)
      DW Block 1 : DWSep (64,  stride=1)
      DW Block 2 : DWSep (128, stride=2)
      DW Block 3 : DWSep (128, stride=1)
      DW Block 4 : DWSep (256, stride=2)
      DW Block 5 : DWSep (256, stride=1)
      DW Block 6 : DWSep (512, stride=2)
      DW Blocks 7-11: DWSep (512, stride=1) x5
      DW Block 12: DWSep (1024, stride=2)
      DW Block 13: DWSep (1024, stride=1)
      GlobalAvgPool -> Dense(num_classes, softmax)
    """
    stem_filters = int(32 * alpha)

    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(stem_filters, 3, strides=2, padding='same',
                      use_bias=False, name='conv1')(inputs)
    x = layers.BatchNormalization(name='conv1_bn')(x)
    x = layers.ReLU(6., name='conv1_relu')(x)

    # 13 Depthwise Separable blocks
    x = _dw_block(x,  64, stride=1, block_id=1,  alpha=alpha)
    x = _dw_block(x, 128, stride=2, block_id=2,  alpha=alpha)
    x = _dw_block(x, 128, stride=1, block_id=3,  alpha=alpha)
    x = _dw_block(x, 256, stride=2, block_id=4,  alpha=alpha)
    x = _dw_block(x, 256, stride=1, block_id=5,  alpha=alpha)
    x = _dw_block(x, 512, stride=2, block_id=6,  alpha=alpha)
    x = _dw_block(x, 512, stride=1, block_id=7,  alpha=alpha)
    x = _dw_block(x, 512, stride=1, block_id=8,  alpha=alpha)
    x = _dw_block(x, 512, stride=1, block_id=9,  alpha=alpha)
    x = _dw_block(x, 512, stride=1, block_id=10, alpha=alpha)
    x = _dw_block(x, 512, stride=1, block_id=11, alpha=alpha)
    x = _dw_block(x, 1024, stride=2, block_id=12, alpha=alpha)
    x = _dw_block(x, 1024, stride=1, block_id=13, alpha=alpha)

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    outputs = layers.Dense(num_classes, activation='softmax',
                           name='predictions')(x)

    return keras.Model(inputs=inputs, outputs=outputs, name='mobilenet')
