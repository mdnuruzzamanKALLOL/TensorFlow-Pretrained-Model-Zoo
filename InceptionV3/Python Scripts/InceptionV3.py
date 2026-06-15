"""InceptionV3 from scratch — TensorFlow/Keras."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def _conv_bn(x, filters, kernel_size, strides=1, padding="valid", activation="relu"):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(x)
    x = layers.BatchNormalization(scale=False)(x)
    if activation:
        x = layers.Activation(activation)(x)
    return x

def _inception_a(x, pool_filters):
    """35x35 Inception module (type A). Factorises 5x5 into two 3x3."""
    b0 = _conv_bn(x, 64, 1)
    b1 = _conv_bn(x, 48, 1)
    b1 = _conv_bn(b1, 64, 5, padding="same")
    b2 = _conv_bn(x, 64, 1)
    b2 = _conv_bn(b2, 96, 3, padding="same")
    b2 = _conv_bn(b2, 96, 3, padding="same")
    b3 = layers.AveragePooling2D(3, strides=1, padding="same")(x)
    b3 = _conv_bn(b3, pool_filters, 1)
    return layers.Concatenate()([b0, b1, b2, b3])

def _reduction_a(x):
    """35x35 -> 17x17, 288 -> 768 channels."""
    b0 = _conv_bn(x, 384, 3, strides=2)
    b1 = _conv_bn(x, 64, 1)
    b1 = _conv_bn(b1, 96, 3, padding="same")
    b1 = _conv_bn(b1, 96, 3, strides=2)
    b2 = layers.MaxPooling2D(3, strides=2)(x)
    return layers.Concatenate()([b0, b1, b2])

def _inception_b(x, branch_filters):
    """17x17 Inception module (type B). Factorises nxn into 1xn + nx1."""
    b0 = _conv_bn(x, 192, 1)
    b1 = _conv_bn(x, branch_filters, 1)
    b1 = _conv_bn(b1, branch_filters, (1, 7), padding="same")
    b1 = _conv_bn(b1, 192, (7, 1), padding="same")
    b2 = _conv_bn(x, branch_filters, 1)
    b2 = _conv_bn(b2, branch_filters, (7, 1), padding="same")
    b2 = _conv_bn(b2, branch_filters, (1, 7), padding="same")
    b2 = _conv_bn(b2, branch_filters, (7, 1), padding="same")
    b2 = _conv_bn(b2, 192, (1, 7), padding="same")
    b3 = layers.AveragePooling2D(3, strides=1, padding="same")(x)
    b3 = _conv_bn(b3, 192, 1)
    return layers.Concatenate()([b0, b1, b2, b3])

def _reduction_b(x):
    """17x17 -> 8x8, 768 -> 1280 channels."""
    b0 = _conv_bn(x, 192, 1)
    b0 = _conv_bn(b0, 320, 3, strides=2)
    b1 = _conv_bn(x, 192, 1)
    b1 = _conv_bn(b1, 192, (1, 7), padding="same")
    b1 = _conv_bn(b1, 192, (7, 1), padding="same")
    b1 = _conv_bn(b1, 192, 3, strides=2)
    b2 = layers.MaxPooling2D(3, strides=2)(x)
    return layers.Concatenate()([b0, b1, b2])

def _inception_c(x):
    """8x8 Inception module (type C). Parallel 1x3 and 3x1 branches."""
    b0 = _conv_bn(x, 320, 1)
    b1 = _conv_bn(x, 384, 1)
    b1a = _conv_bn(b1, 384, (1, 3), padding="same")
    b1b = _conv_bn(b1, 384, (3, 1), padding="same")
    b1 = layers.Concatenate()([b1a, b1b])
    b2 = _conv_bn(x, 448, 1)
    b2 = _conv_bn(b2, 384, 3, padding="same")
    b2a = _conv_bn(b2, 384, (1, 3), padding="same")
    b2b = _conv_bn(b2, 384, (3, 1), padding="same")
    b2 = layers.Concatenate()([b2a, b2b])
    b3 = layers.AveragePooling2D(3, strides=1, padding="same")(x)
    b3 = _conv_bn(b3, 192, 1)
    return layers.Concatenate()([b0, b1, b2, b3])

def build_inceptionv3(num_classes=1000, input_shape=(299, 299, 3)):
    inputs = keras.Input(shape=input_shape)

    # Stem: 299 -> 149 -> 147 -> 73 -> 71 -> 35
    x = _conv_bn(inputs, 32, 3, strides=2)          # 149x149
    x = _conv_bn(x, 32, 3)                           # 147x147
    x = _conv_bn(x, 64, 3, padding="same")           # 147x147
    x = layers.MaxPooling2D(3, strides=2)(x)         # 73x73
    x = _conv_bn(x, 80, 1)                           # 73x73
    x = _conv_bn(x, 192, 3)                          # 71x71
    x = layers.MaxPooling2D(3, strides=2)(x)         # 35x35

    # Inception-A x3: 35x35 (pool_filters vary: 32, 64, 64)
    x = _inception_a(x, pool_filters=32)             # 35x35x256
    x = _inception_a(x, pool_filters=64)             # 35x35x288
    x = _inception_a(x, pool_filters=64)             # 35x35x288

    # Grid Reduction-A: 35 -> 17
    x = _reduction_a(x)                              # 17x17x768

    # Inception-B x4: 17x17 (branch_filters: 128, 160, 160, 192)
    x = _inception_b(x, branch_filters=128)          # 17x17x768
    x = _inception_b(x, branch_filters=160)          # 17x17x768
    x = _inception_b(x, branch_filters=160)          # 17x17x768
    x = _inception_b(x, branch_filters=192)          # 17x17x768

    # Grid Reduction-B: 17 -> 8
    x = _reduction_b(x)                              # 8x8x1280

    # Inception-C x2: 8x8
    x = _inception_c(x)                              # 8x8x2048
    x = _inception_c(x)                              # 8x8x2048

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, x, name="InceptionV3")

if __name__ == "__main__":
    model = build_inceptionv3(num_classes=1000)
    model.summary()
