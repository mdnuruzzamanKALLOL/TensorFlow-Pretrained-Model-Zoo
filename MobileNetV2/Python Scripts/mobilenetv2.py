import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _make_divisible(v, divisor=8, min_value=None):
    """Ensure channel count is divisible by divisor (needed for width multiplier)."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_residual(x, filters, stride, expand_ratio, block_id):
    """
    MobileNetV2 Inverted Residual Block (Linear Bottleneck).

    Structure: Expand -> Depthwise -> Project (linear, no activation).
    Residual added only when stride=1 AND input channels == output channels.

    Key insight: the bottleneck is 'inverted' vs ResNet:
      ResNet   : wide -> narrow -> wide  (compress then expand)
      V2       : narrow -> wide -> narrow (expand then project)

    The final projection has NO activation (linear bottleneck) to preserve
    the information manifold before the residual addition.

    block_id=0: special case 'expanded_conv' (t=1, no expand phase, no residual)
    block_id>0: 'block_{block_id}_expand / depthwise / project'
    """
    in_channels  = x.shape[-1]
    prefix       = 'expanded_conv' if block_id == 0 else f'block_{block_id}'
    shortcut     = x

    # ── Expand ──
    if expand_ratio != 1:
        exp_ch = _make_divisible(in_channels * expand_ratio)
        x = layers.Conv2D(exp_ch, 1, padding='same', use_bias=False,
                          name=f'{prefix}_expand')(x)
        x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999,
                                      name=f'{prefix}_expand_BN')(x)
        x = layers.ReLU(6., name=f'{prefix}_expand_relu')(x)

    # ── Depthwise ──
    x = layers.DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False,
                                name=f'{prefix}_depthwise')(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999,
                                  name=f'{prefix}_depthwise_BN')(x)
    x = layers.ReLU(6., name=f'{prefix}_depthwise_relu')(x)

    # ── Project (linear — no activation) ──
    x = layers.Conv2D(filters, 1, padding='same', use_bias=False,
                      name=f'{prefix}_project')(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999,
                                  name=f'{prefix}_project_BN')(x)

    # ── Residual: only when stride=1 and channels match ──
    if stride == 1 and in_channels == filters:
        x = layers.Add(name=f'{prefix}_add')([shortcut, x])

    return x


def build_mobilenetv2(num_classes=1000, input_shape=(224, 224, 3), alpha=1.0):
    """
    MobileNetV2 — Inverted Residuals and Linear Bottlenecks.
    Paper: Sandler et al., CVPR 2018.

    Key improvements over MobileNetV1:
      1. Inverted Residuals: expand channels (6x) -> DWConv -> project back
      2. Linear Bottleneck: no activation on the projection layer
      3. Residual connections: skip connection when stride=1 and dims match
      4. Better accuracy at same or fewer parameters vs V1

    Inverted Residual Block config: (t=expand_ratio, c=channels, n=repeats, s=stride)

    t   c    n  s
    1   16   1  1   <- expanded_conv (no expand, t=1)
    6   24   2  2
    6   32   3  2
    6   64   4  2
    6   96   3  1
    6  160   3  2
    6  320   1  1
    Then: Conv1x1(1280) + BN + ReLU6  -> GAP -> Dense
    """
    # Inverted residual configs: (expand_ratio, channels, num_repeats, stride)
    configs = [
        (1,  16, 1, 1),
        (6,  24, 2, 2),
        (6,  32, 3, 2),
        (6,  64, 4, 2),
        (6,  96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(_make_divisible(32 * alpha), 3, strides=2,
                      padding='same', use_bias=False, name='Conv1')(inputs)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999,
                                  name='bn_Conv1')(x)
    x = layers.ReLU(6., name='Conv1_relu')(x)

    block_id = 0
    for t, c, n, s in configs:
        out_ch = _make_divisible(c * alpha)
        for i in range(n):
            stride = s if i == 0 else 1
            x = _inverted_residual(x, out_ch, stride, t, block_id)
            block_id += 1

    # Head conv: expand to 1280 channels before GAP
    last_ch = max(_make_divisible(1280 * alpha), 1280)
    x = layers.Conv2D(last_ch, 1, padding='same', use_bias=False,
                      name='Conv_1')(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999,
                                  name='Conv_1_bn')(x)
    x = layers.ReLU(6., name='out_relu')(x)

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    outputs = layers.Dense(num_classes, activation='softmax',
                           name='predictions')(x)

    return keras.Model(inputs=inputs, outputs=outputs, name='mobilenetv2')
