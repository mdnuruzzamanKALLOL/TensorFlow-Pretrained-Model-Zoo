"""InceptionResNetV2 from scratch — TensorFlow/Keras."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def _conv_bn(x, filters, kernel_size, strides=1, padding="same", activation="relu", name=None):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding,
                      use_bias=False, name=name)(x)
    bn_name = (name + "_bn") if name else None
    x = layers.BatchNormalization(scale=False, name=bn_name)(x)
    if activation:
        act_name = (name + "_act") if name else None
        x = layers.Activation(activation, name=act_name)(x)
    return x

def _inception_resnet_a(x, scale=0.17):
    """InceptionResNet-A: 35x35 residual inception block."""
    b0 = _conv_bn(x, 32, 1)
    b1 = _conv_bn(x, 32, 1)
    b1 = _conv_bn(b1, 32, 3, padding="same")
    b2 = _conv_bn(x, 32, 1)
    b2 = _conv_bn(b2, 48, 3, padding="same")
    b2 = _conv_bn(b2, 64, 3, padding="same")
    mixed = layers.Concatenate()([b0, b1, b2])           # 32+32+64=128
    up = _conv_bn(mixed, x.shape[-1], 1, activation=None) # project to input channels
    scaled = layers.Lambda(lambda t: t * scale)(up)
    x = layers.Add()([x, scaled])
    x = layers.Activation("relu")(x)
    return x

def _inception_resnet_b(x, scale=0.1):
    """InceptionResNet-B: 17x17 residual inception block with 1x7 / 7x1 factorisation."""
    b0 = _conv_bn(x, 192, 1)
    b1 = _conv_bn(x, 128, 1)
    b1 = _conv_bn(b1, 160, (1, 7), padding="same")
    b1 = _conv_bn(b1, 192, (7, 1), padding="same")
    mixed = layers.Concatenate()([b0, b1])                # 192+192=384
    up = _conv_bn(mixed, x.shape[-1], 1, activation=None) # project to 1088
    scaled = layers.Lambda(lambda t: t * scale)(up)
    x = layers.Add()([x, scaled])
    x = layers.Activation("relu")(x)
    return x

def _inception_resnet_c(x, scale=0.2, activation="relu"):
    """InceptionResNet-C: 8x8 residual inception block with 1x3 / 3x1 factorisation."""
    b0 = _conv_bn(x, 192, 1)
    b1 = _conv_bn(x, 192, 1)
    b1 = _conv_bn(b1, 224, (1, 3), padding="same")
    b1 = _conv_bn(b1, 256, (3, 1), padding="same")
    mixed = layers.Concatenate()([b0, b1])                # 192+256=448
    up = _conv_bn(mixed, x.shape[-1], 1, activation=None) # project to 2080
    scaled = layers.Lambda(lambda t: t * scale)(up)
    x = layers.Add()([x, scaled])
    if activation:
        x = layers.Activation(activation)(x)
    return x

def _reduction_a(x):
    """Grid Reduction-A: 35x35 -> 17x17 (320 -> 1088 channels)."""
    b0 = _conv_bn(x, 384, 3, strides=2, padding="valid")
    b1 = _conv_bn(x, 256, 1)
    b1 = _conv_bn(b1, 256, 3, padding="same")
    b1 = _conv_bn(b1, 384, 3, strides=2, padding="valid")
    b2 = layers.MaxPooling2D(3, strides=2, padding="valid")(x)
    return layers.Concatenate()([b0, b1, b2])              # 384+384+320=1088

def _reduction_b(x):
    """Grid Reduction-B: 17x17 -> 8x8 (1088 -> 2080 channels)."""
    b0 = _conv_bn(x, 256, 1)
    b0 = _conv_bn(b0, 384, 3, strides=2, padding="valid")
    b1 = _conv_bn(x, 256, 1)
    b1 = _conv_bn(b1, 288, 3, strides=2, padding="valid")
    b2 = _conv_bn(x, 256, 1)
    b2 = _conv_bn(b2, 288, 3, padding="same")
    b2 = _conv_bn(b2, 320, 3, strides=2, padding="valid")
    b3 = layers.MaxPooling2D(3, strides=2, padding="valid")(x)
    return layers.Concatenate()([b0, b1, b2, b3])          # 384+288+320+1088=2080

def build_inception_resnet_v2(num_classes=1000, input_shape=(299, 299, 3)):
    inputs = keras.Input(shape=input_shape)

    # Stem (shared with InceptionV3): 299 -> 35x35x192
    x = _conv_bn(inputs, 32, 3, strides=2, padding="valid")   # 149x149
    x = _conv_bn(x, 32, 3, padding="valid")                    # 147x147
    x = _conv_bn(x, 64, 3, padding="same")                     # 147x147
    x = layers.MaxPooling2D(3, strides=2, padding="valid")(x)  # 73x73
    x = _conv_bn(x, 80, 1, padding="valid")                    # 73x73
    x = _conv_bn(x, 192, 3, padding="valid")                   # 71x71
    x = layers.MaxPooling2D(3, strides=2, padding="valid")(x)  # 35x35x192

    # mixed_5b: Inception-A style block -> 35x35x320
    b0 = _conv_bn(x, 96, 1)
    b1 = _conv_bn(x, 48, 1)
    b1 = _conv_bn(b1, 64, 5, padding="same")
    b2 = _conv_bn(x, 64, 1)
    b2 = _conv_bn(b2, 96, 3, padding="same")
    b2 = _conv_bn(b2, 96, 3, padding="same")
    b3 = layers.AveragePooling2D(3, strides=1, padding="same")(x)
    b3 = _conv_bn(b3, 64, 1)
    x  = layers.Concatenate(name="mixed_5b")([b0, b1, b2, b3])  # 96+64+96+64=320

    # InceptionResNet-A x5: 35x35x320
    for i in range(1, 6):
        x = _inception_resnet_a(x, scale=0.17)

    # Reduction-A: 35 -> 17
    x = _reduction_a(x)                                          # 17x17x1088

    # InceptionResNet-B x10: 17x17x1088
    for i in range(1, 11):
        x = _inception_resnet_b(x, scale=0.1)

    # Reduction-B: 17 -> 8
    x = _reduction_b(x)                                          # 8x8x2080

    # InceptionResNet-C x5: 8x8x2080 (last without activation)
    for i in range(1, 5):
        x = _inception_resnet_c(x, scale=0.2, activation="relu")
    x = _inception_resnet_c(x, scale=1.0, activation=False)     # final: no scaling down

    # Final projection conv: 2080 -> 1536
    x = _conv_bn(x, 1536, 1, name="conv_7b")                    # 8x8x1536

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, x, name="InceptionResNetV2")

if __name__ == "__main__":
    model = build_inception_resnet_v2(num_classes=1000)
    model.summary()
