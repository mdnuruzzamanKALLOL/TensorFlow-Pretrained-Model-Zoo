"""EfficientNetV2S from scratch — TensorFlow/Keras."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def _conv_bn(x, filters, kernel_size, strides=1, padding="same", activation="swish"):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if activation:
        x = layers.Activation(activation)(x)
    return x

def _se_block(x, in_ch, se_ratio):
    """Squeeze-Excitation: bottleneck = in_ch * se_ratio."""
    se_ch = max(1, int(in_ch * se_ratio))
    ch    = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Reshape((1, 1, ch))(se)
    se = layers.Conv2D(se_ch, 1, activation="swish",   use_bias=True)(se)
    se = layers.Conv2D(ch,    1, activation="sigmoid", use_bias=True)(se)
    return layers.Multiply()([x, se])

def _fused_mbconv(x, in_ch, out_ch, expand, stride, se_ratio=0.0):
    """Fused-MBConv: single 3x3 conv replaces expand+DWConv (used in early stages)."""
    shortcut = x
    if expand != 1:
        x = _conv_bn(x, in_ch * expand, 3, strides=stride)  # fused expand (with act)
        x = _conv_bn(x, out_ch, 1, activation=None)          # project (no act)
    else:
        x = _conv_bn(x, out_ch, 3, strides=stride)           # single conv (with act)
    if se_ratio > 0:
        x = _se_block(x, in_ch, se_ratio)
    if stride == 1 and in_ch == out_ch:
        x = layers.Add()([shortcut, x])
    return x

def _mbconv(x, in_ch, out_ch, expand, stride, se_ratio=0.25):
    """MBConv with Squeeze-Excitation (used in later stages)."""
    shortcut = x
    x = _conv_bn(x, in_ch * expand, 1)                                                 # expand
    x = layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False)(x)   # depthwise
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    if se_ratio > 0:
        x = _se_block(x, in_ch, se_ratio)
    x = _conv_bn(x, out_ch, 1, activation=None)                                        # project
    if stride == 1 and in_ch == out_ch:
        x = layers.Add()([shortcut, x])
    return x

def _build_stages(x, block_args):
    for stage in block_args:
        in_ch, out_ch = stage["in_ch"], stage["out_ch"]
        for i in range(stage["n"]):
            s  = stage["stride"] if i == 0 else 1
            ic = in_ch           if i == 0 else out_ch
            if stage["fused"]:
                x = _fused_mbconv(x, ic, out_ch, stage["expand"], s, stage["se"])
            else:
                x = _mbconv(x, ic, out_ch, stage["expand"], s, stage["se"])
    return x

# EfficientNetV2-S block configuration
BLOCK_ARGS_S = [
    {"n": 2,  "in_ch": 24,  "out_ch": 24,  "expand": 1, "stride": 1, "se": 0.0,  "fused": True},
    {"n": 4,  "in_ch": 24,  "out_ch": 48,  "expand": 4, "stride": 2, "se": 0.0,  "fused": True},
    {"n": 4,  "in_ch": 48,  "out_ch": 64,  "expand": 4, "stride": 2, "se": 0.0,  "fused": True},
    {"n": 6,  "in_ch": 64,  "out_ch": 128, "expand": 4, "stride": 2, "se": 0.25, "fused": False},
    {"n": 9,  "in_ch": 128, "out_ch": 160, "expand": 6, "stride": 1, "se": 0.25, "fused": False},
    {"n": 15, "in_ch": 160, "out_ch": 256, "expand": 6, "stride": 2, "se": 0.25, "fused": False},
]

def build_efficientnetv2s(num_classes=1000, input_shape=(300, 300, 3)):
    inputs = keras.Input(shape=input_shape)
    x = _conv_bn(inputs, 24, 3, strides=2)          # stem: 300->150
    x = _build_stages(x, BLOCK_ARGS_S)
    x = _conv_bn(x, 1280, 1)                         # head conv
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, x, name="EfficientNetV2S")

if __name__ == "__main__":
    model = build_efficientnetv2s(num_classes=1000)
    model.summary()
