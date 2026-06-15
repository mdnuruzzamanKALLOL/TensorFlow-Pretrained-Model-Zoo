import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _conv_bn(x, filters, kernel_size, strides=1, padding="same", activation="swish"):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding,
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if activation:
        x = layers.Activation(activation)(x)
    return x


def _se_block(x, in_ch, se_ratio):
    """Squeeze-Excitation bottleneck sized by pre-expansion in_ch."""
    se_ch = max(1, int(in_ch * se_ratio))
    ch    = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Reshape((1, 1, ch))(se)
    se = layers.Conv2D(se_ch, 1, activation="swish",   use_bias=True)(se)
    se = layers.Conv2D(ch,    1, activation="sigmoid", use_bias=True)(se)
    return layers.Multiply()([x, se])


def _mbconv(x, in_ch, out_ch, expand_ratio, stride, kernel_size, se_ratio=0.25):
    """MBConv: expand-1x1(opt) - DWConv-kxk - SE - project-1x1 - residual."""
    shortcut    = x
    expanded_ch = in_ch * expand_ratio

    # Expand (skipped when expand_ratio == 1)
    if expand_ratio != 1:
        x = _conv_bn(x, expanded_ch, 1)

    # Depthwise conv (kernel_size varies per stage: 3 or 5)
    x = layers.DepthwiseConv2D(kernel_size, strides=stride,
                               padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)

    # Squeeze-Excitation (se_ch based on pre-expansion in_ch)
    if se_ratio > 0:
        x = _se_block(x, in_ch, se_ratio)

    # Project (no activation)
    x = _conv_bn(x, out_ch, 1, activation=None)

    # Residual only when stride==1 and channels are unchanged
    if stride == 1 and in_ch == out_ch:
        x = layers.Add()([shortcut, x])

    return x


def _build_blocks(x, block_args):
    for stage in block_args:
        in_ch, out_ch = stage["in_ch"], stage["out_ch"]
        for i in range(stage["n"]):
            s  = stage["stride"] if i == 0 else 1
            ic = in_ch           if i == 0 else out_ch
            x  = _mbconv(x, ic, out_ch, stage["expand"],
                         s, stage["k"], se_ratio=stage["se"])
    return x

BLOCK_ARGS = [
    {"k":3, "n": 3, "in_ch": 56, "out_ch": 32, "expand":1, "stride":1, "se":0.25},
    {"k":3, "n": 6, "in_ch": 32, "out_ch": 40, "expand":6, "stride":2, "se":0.25},
    {"k":5, "n": 6, "in_ch": 40, "out_ch": 72, "expand":6, "stride":2, "se":0.25},
    {"k":3, "n": 8, "in_ch": 72, "out_ch":144, "expand":6, "stride":2, "se":0.25},
    {"k":5, "n": 8, "in_ch":144, "out_ch":200, "expand":6, "stride":1, "se":0.25},
    {"k":5, "n":11, "in_ch":200, "out_ch":344, "expand":6, "stride":2, "se":0.25},
    {"k":3, "n": 3, "in_ch":344, "out_ch":576, "expand":6, "stride":1, "se":0.25},
]


def build_efficientnet_b6(num_classes=1000, input_shape=(528, 528, 3)):
    """EfficientNet-B6: stem=56, head_conv=2304, input=528x528, dropout=0.5."""
    inputs = keras.Input(shape=input_shape)

    # Stem: 3x3 conv, stride 2
    x = _conv_bn(inputs, 56, 3, strides=2)

    # Seven MBConv stages (compound-scaled)
    x = _build_blocks(x, BLOCK_ARGS)

    # Head conv + GAP + Dropout + classifier
    x = _conv_bn(x, 2304, 1)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="EfficientNetB6")


if __name__ == "__main__":
    model = build_efficientnet_b6()
    model.summary()
