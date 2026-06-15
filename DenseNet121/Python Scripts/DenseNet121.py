import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

GROWTH_RATE = 32
COMPRESSION = 0.5


def _dense_layer(x, growth_rate=GROWTH_RATE):
    """BN-ReLU-Conv1x1(4k)-BN-ReLU-Conv3x3(k) bottleneck layer."""
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(4 * growth_rate, 1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(growth_rate, 3, padding="same", use_bias=False)(x)
    return x


def _dense_block(x, num_layers, growth_rate=GROWTH_RATE):
    """Stack dense layers; each layer receives all previous feature maps."""
    for _ in range(num_layers):
        new_feat = _dense_layer(x, growth_rate)
        x = layers.Concatenate()([x, new_feat])
    return x


def _transition_block(x, compression=COMPRESSION):
    """BN-ReLU-Conv1x1(theta*ch)-AvgPool2x2 to halve spatial and channel dims."""
    ch = int(x.shape[-1] * compression)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(ch, 1, use_bias=False)(x)
    x = layers.AveragePooling2D(2, strides=2)(x)
    return x

def build_densenet121(num_classes=1000, input_shape=(224, 224, 3)):
    """DenseNet-121: blocks=[6, 12, 24, 16], growth_rate=32, compression=0.5."""
    blocks = [6, 12, 24, 16]
    inputs = keras.Input(shape=input_shape)

    # Stem: 224->112->56
    x = layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # Dense blocks with transitions
    for i, n in enumerate(blocks):
        x = _dense_block(x, n)
        if i < len(blocks) - 1:
            x = _transition_block(x)

    # Classification head
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="DenseNet121")


if __name__ == "__main__":
    model = build_densenet121()
    model.summary()
