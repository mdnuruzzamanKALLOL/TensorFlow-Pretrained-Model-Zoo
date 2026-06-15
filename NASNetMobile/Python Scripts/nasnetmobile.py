import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ── Cell helpers ──────────────────────────────────────────────────────────────

def _sep_conv_bn(x, filters, kernel_size, strides=1):
    """Two stacked depthwise-separable convolutions with ReLU prefix and BN.
    Only the first SepConv uses the given stride; the second always uses stride=1.
    """
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(filters, kernel_size, strides=strides,
                                padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.9997, epsilon=1e-3)(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(filters, kernel_size, strides=1,
                                padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.9997, epsilon=1e-3)(x)
    return x


def _adjust_block(p, ip, filters):
    """Adjust p so its spatial size and channel count match what the cell expects.
    Three cases:
      p is None          -> use ip as identity
      spatial mismatch   -> 2-branch strided avg-pool + conv to halve spatial dims
      channel mismatch   -> 1x1 conv projection to 'filters' channels
    """
    if p is None:
        return ip
    if ip.shape[1] != p.shape[1]:
        # p has 2x the spatial size of ip -> downsample with two offset branches
        p = layers.Activation('relu')(p)
        p1 = layers.AveragePooling2D(1, strides=2, padding='valid')(p)
        p1 = layers.Conv2D(filters // 2, 1, padding='same', use_bias=False)(p1)
        p2 = layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(p)
        p2 = layers.Cropping2D(cropping=((1, 0), (1, 0)))(p2)
        p2 = layers.AveragePooling2D(1, strides=2, padding='valid')(p2)
        p2 = layers.Conv2D(filters // 2, 1, padding='same', use_bias=False)(p2)
        p = layers.Concatenate()([p1, p2])
        p = layers.BatchNormalization(momentum=0.9997, epsilon=1e-3)(p)
    elif p.shape[-1] != filters:
        p = layers.Activation('relu')(p)
        p = layers.Conv2D(filters, 1, padding='same', use_bias=False)(p)
        p = layers.BatchNormalization(momentum=0.9997, epsilon=1e-3)(p)
    return p


def _normal_cell(ip, p, filters):
    """NASNet-A Normal Cell (Figure 4, Zoph et al. 2018).
    Maintains spatial dimensions. Output has 6*filters channels.
    Returns: (cell_output, ip)  -- ip is passed through for the next cell's skip.

    5 parallel blocks added to adjusted-p:
      1: sep5x5(h)  + identity(h)
      2: sep5x5(p)  + sep3x3(h)
      3: avg3x3(h)  + p
      4: avg3x3(p)  + avg3x3(p)
      5: max3x3(h)  + sep3x3(p)
    Concat([p, x1, x2, x3, x4, x5])
    """
    p = _adjust_block(p, ip, filters)

    h = layers.Activation('relu')(ip)
    h = layers.Conv2D(filters, 1, padding='same', use_bias=False)(h)
    h = layers.BatchNormalization(momentum=0.9997, epsilon=1e-3)(h)

    x1 = layers.Add()([_sep_conv_bn(h, filters, 5), h])
    x2 = layers.Add()([_sep_conv_bn(p, filters, 5), _sep_conv_bn(h, filters, 3)])
    x3 = layers.Add()([layers.AveragePooling2D(3, strides=1, padding='same')(h), p])
    x4 = layers.Add()([
        layers.AveragePooling2D(3, strides=1, padding='same')(p),
        layers.AveragePooling2D(3, strides=1, padding='same')(p),
    ])
    x5 = layers.Add()([
        layers.MaxPooling2D(3, strides=1, padding='same')(h),
        _sep_conv_bn(p, filters, 3),
    ])
    return layers.Concatenate()([p, x1, x2, x3, x4, x5]), ip


def _reduction_cell(ip, p, filters):
    """NASNet-A Reduction Cell (Figure 4, Zoph et al. 2018).
    Halves spatial dimensions. Output has 4*filters channels.
    Returns: (cell_output, ip)

    5 blocks (first 3 use stride-2 ops on h/p):
      1: sep5x5_s2(h) + sep7x7_s2(p)
      2: max3x3_s2(h) + sep7x7_s2(p)
      3: avg3x3_s2(h) + sep5x5_s2(p)
      4: avg3x3_s1(x1) + x2
      5: sep3x3_s1(x1) + max3x3_s2(h)
    Concat([x2, x3, x4, x5])
    """
    p = _adjust_block(p, ip, filters)

    h = layers.Activation('relu')(ip)
    h = layers.Conv2D(filters, 1, padding='same', use_bias=False)(h)
    h = layers.BatchNormalization(momentum=0.9997, epsilon=1e-3)(h)

    x1 = layers.Add()([_sep_conv_bn(h, filters, 5, strides=2),
                        _sep_conv_bn(p, filters, 7, strides=2)])
    x2 = layers.Add()([layers.MaxPooling2D(3, strides=2, padding='same')(h),
                        _sep_conv_bn(p, filters, 7, strides=2)])
    x3 = layers.Add()([layers.AveragePooling2D(3, strides=2, padding='same')(h),
                        _sep_conv_bn(p, filters, 5, strides=2)])
    x4 = layers.Add()([layers.AveragePooling2D(3, strides=1, padding='same')(x1), x2])
    x5 = layers.Add()([_sep_conv_bn(x1, filters, 3),
                        layers.MaxPooling2D(3, strides=2, padding='same')(h)])
    return layers.Concatenate()([x2, x3, x4, x5]), ip


# ── Model ─────────────────────────────────────────────────────────────────────

def build_nasnetmobile(num_classes=1000, input_shape=(224, 224, 3)):
    """
    NASNetMobile — Neural Architecture Search Network (Mobile variant).
    Paper: Learning Transferable Architectures for Scalable Image Recognition
           Zoph et al., CVPR 2018.

    Architecture discovered by NAS on CIFAR-10 and transferred to ImageNet.
    NASNetMobile is the efficiency-optimised variant targeting mobile inference.

    Hyper-parameters:
      penultimate_filters = 1056
      filters             = 1056 // 24 = 44
      num_blocks          = 4  (Normal Cells per group)
      stem_filters        = 32
      filter_multiplier   = 2
      skip_reduction      = False

    Flow:
      Stem Conv3x3/2 (32 filters, padding=valid)
      Stem Reduction Cell  (filters*4 = 176)
      Stem Reduction Cell  (filters*2 = 88)
      Group 1: 4 Normal Cells (filters=44)       -> 6*44  = 264 ch
      Reduction Cell (filters*2=88,  skip_red=False)
      Group 2: 4 Normal Cells (filters*2=88)     -> 6*88  = 528 ch
      Reduction Cell (filters*4=176, skip_red=False)
      Group 3: 4 Normal Cells (filters*4=176)    -> 6*176 = 1056 ch
      ReLU -> GlobalAvgPool -> Dense(num_classes)
    """
    penultimate_filters = 1056
    num_blocks   = 4
    stem_filters = 32
    filt_mult    = 2
    filters      = penultimate_filters // 24   # 44

    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(stem_filters, 3, strides=2, padding='valid',
                      use_bias=False, name='stem_conv1')(inputs)
    x = layers.BatchNormalization(momentum=0.9997, epsilon=1e-3, name='stem_bn1')(x)

    p = None
    # Two stem reduction cells
    x, p = _reduction_cell(x, p, filters * filt_mult**2)   # filters*4 = 176
    x, p = _reduction_cell(x, p, filters * filt_mult)      # filters*2 = 88

    # Group 1: Normal cells (filters=44)
    for _ in range(num_blocks):
        x, p = _normal_cell(x, p, filters)

    # Reduction 1 (skip_reduction=False: p = returned ip = old x)
    x, p = _reduction_cell(x, p, filters * filt_mult)

    # Group 2: Normal cells (filters*2=88)
    for _ in range(num_blocks):
        x, p = _normal_cell(x, p, filters * filt_mult)

    # Reduction 2 (skip_reduction=False)
    x, p = _reduction_cell(x, p, filters * filt_mult**2)

    # Group 3: Normal cells (filters*4=176) -> 6*176=1056 channels
    for _ in range(num_blocks):
        x, p = _normal_cell(x, p, filters * filt_mult**2)

    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    return keras.Model(inputs=inputs, outputs=outputs, name='nasnetmobile')
