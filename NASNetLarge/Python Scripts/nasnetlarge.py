import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ── Cell helpers (identical to NASNetMobile) ──────────────────────────────────

def _sep_conv_bn(x, filters, kernel_size, strides=1):
    """Two stacked depthwise-separable convolutions with ReLU prefix and BN.
    Only the first SepConv applies the given stride; the second always uses stride=1.
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
    """NASNet-A Normal Cell. Maintains spatial dimensions. Output: 6*filters channels.
    Returns: (cell_output, ip)  -- ip is passed through for the next cell's skip.
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
    """NASNet-A Reduction Cell. Halves spatial dimensions. Output: 4*filters channels.
    Returns: (cell_output, ip)
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

def build_nasnetlarge(num_classes=1000, input_shape=(331, 331, 3)):
    """
    NASNetLarge — Neural Architecture Search Network (Large variant).
    Paper: Learning Transferable Architectures for Scalable Image Recognition
           Zoph et al., CVPR 2018.

    NASNetLarge is the high-accuracy variant. It uses the same NAS-discovered
    cell structure as NASNetMobile but with more filters and more blocks per group,
    plus skip_reduction=True (after each explicit reduction cell, p retains its
    pre-reduction value rather than being updated to the reduction output).

    Hyper-parameters:
      penultimate_filters = 4032
      filters             = 4032 // 24 = 168
      num_blocks          = 6  (Normal Cells per group)
      stem_filters        = 96
      filter_multiplier   = 2
      skip_reduction      = True

    skip_reduction=True: after each explicit reduction cell, the skip connection p
    is NOT updated to the reduction output. Normal cells in the next group see the
    reduction output as ip but the pre-reduction p as their skip input.
    This preserves richer spatial information across the skip path.

    Flow:
      Stem Conv3x3/2 (96 filters, padding=valid)
      Stem Reduction Cell  (filters*4 = 672)
      Stem Reduction Cell  (filters*2 = 336)
      Group 1: 6 Normal Cells (filters=168)      -> 6*168  = 1008 ch
      Reduction Cell (filters*2=336, skip_red=True)
      Group 2: 6 Normal Cells (filters*2=336)    -> 6*336  = 2016 ch
      Reduction Cell (filters*4=672, skip_red=True)
      Group 3: 6 Normal Cells (filters*4=672)    -> 6*672  = 4032 ch
      ReLU -> GlobalAvgPool -> Dense(num_classes)
    """
    penultimate_filters = 4032
    num_blocks   = 6
    stem_filters = 96
    filt_mult    = 2
    filters      = penultimate_filters // 24   # 168

    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(stem_filters, 3, strides=2, padding='valid',
                      use_bias=False, name='stem_conv1')(inputs)
    x = layers.BatchNormalization(momentum=0.9997, epsilon=1e-3, name='stem_bn1')(x)

    p = None
    # Two stem reduction cells (same as NASNetMobile, different filters)
    x, p = _reduction_cell(x, p, filters * filt_mult**2)   # filters*4 = 672
    x, p = _reduction_cell(x, p, filters * filt_mult)      # filters*2 = 336

    # Group 1: Normal cells (filters=168)
    for _ in range(num_blocks):
        x, p = _normal_cell(x, p, filters)

    # Reduction 1 — skip_reduction=True: keep old p (don't update to reduction's ip)
    old_p = p
    x, _  = _reduction_cell(x, p, filters * filt_mult)
    p = old_p

    # Group 2: Normal cells (filters*2=336)
    for _ in range(num_blocks):
        x, p = _normal_cell(x, p, filters * filt_mult)

    # Reduction 2 — skip_reduction=True
    old_p = p
    x, _  = _reduction_cell(x, p, filters * filt_mult**2)
    p = old_p

    # Group 3: Normal cells (filters*4=672) -> 6*672=4032 channels
    for _ in range(num_blocks):
        x, p = _normal_cell(x, p, filters * filt_mult**2)

    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    return keras.Model(inputs=inputs, outputs=outputs, name='nasnetlarge')
