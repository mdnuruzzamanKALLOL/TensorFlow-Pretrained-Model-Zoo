import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class LayerScale(layers.Layer):
    """Per-channel learnable scale, initialised to a small constant (1e-6)."""

    def __init__(self, dim, init_value=1e-6, **kwargs):
        super().__init__(**kwargs)
        self._dim        = dim
        self._init_value = init_value

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape       = (self._dim,),
            initializer = tf.keras.initializers.Constant(self._init_value),
            trainable   = True,
            name        = "gamma",
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"dim": self._dim, "init_value": self._init_value})
        return cfg


def convnext_block(x, dim, layer_scale_init=1e-6):
    """ConvNeXt block: DWConv7x7 - LN - Dense(4d) - GELU - Dense(d) - LayerScale - Add."""
    shortcut = x
    # Depthwise 7x7 conv (large receptive field, channel-wise)
    x = layers.DepthwiseConv2D(7, padding="same", use_bias=True)(x)
    # Layer normalisation (applied across channels at each spatial position)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    # Inverted bottleneck MLP via Dense on the channel axis
    x = layers.Dense(4 * dim)(x)
    x = layers.Activation("gelu")(x)
    x = layers.Dense(dim)(x)
    # Per-channel learnable scale (stabilises training of deep networks)
    x = LayerScale(dim, init_value=layer_scale_init)(x)
    # Residual connection
    x = layers.Add()([shortcut, x])
    return x

def build_convnext_base(num_classes=1000, input_shape=(224, 224, 3)):
    """ConvNeXt-Base: dims=[128, 256, 512, 1024], depths=[3, 3, 27, 3]."""
    dims   = [128, 256, 512, 1024]
    depths = [3, 3, 27, 3]
    inputs = keras.Input(shape=input_shape)

    # Stem: patchify with 4x4 non-overlapping conv, stride 4 -> 56x56
    x = layers.Conv2D(dims[0], 4, strides=4, padding="same")(inputs)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Four stages; each stage except the first is preceded by downsampling
    for i in range(4):
        if i > 0:
            # Downsampling: LN + 2x2 stride-2 conv -> halve spatial dims
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            x = layers.Conv2D(dims[i], 2, strides=2, padding="same")(x)
        for _ in range(depths[i]):
            x = convnext_block(x, dims[i])

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="ConvNeXtBase")


if __name__ == "__main__":
    model = build_convnext_base()
    model.summary()
