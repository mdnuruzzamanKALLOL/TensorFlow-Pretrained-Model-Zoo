import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_vgg19(num_classes=1000, input_shape=(224, 224, 3)):
    """
    VGG19 — Very Deep Convolutional Networks for Large-Scale Image Recognition.
    Paper: Simonyan & Zisserman, ICLR 2015.

    VGG19 vs VGG16: blocks 3/4/5 each have 4 conv layers (not 3).
      Block 1 : Conv(64)  x2 + MaxPool  ->  64 x 112x112
      Block 2 : Conv(128) x2 + MaxPool  -> 128 x  56x56
      Block 3 : Conv(256) x4 + MaxPool  -> 256 x  28x28
      Block 4 : Conv(512) x4 + MaxPool  -> 512 x  14x14
      Block 5 : Conv(512) x4 + MaxPool  -> 512 x   7x7
      Head    : Flatten -> Dense(4096) -> Dense(4096) -> Dense(num_classes)

    All convolutions: 3x3, padding='same', ReLU.
    All max-pools   : 2x2, stride=2.
    """
    inputs = keras.Input(shape=input_shape)

    # Block 1  (2 conv)
    x = layers.Conv2D(64, 3, padding='same', activation='relu', name='block1_conv1')(inputs)
    x = layers.Conv2D(64, 3, padding='same', activation='relu', name='block1_conv2')(x)
    x = layers.MaxPooling2D(2, strides=2, name='block1_pool')(x)    # 112x112

    # Block 2  (2 conv)
    x = layers.Conv2D(128, 3, padding='same', activation='relu', name='block2_conv1')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu', name='block2_conv2')(x)
    x = layers.MaxPooling2D(2, strides=2, name='block2_pool')(x)    # 56x56

    # Block 3  (4 conv)
    x = layers.Conv2D(256, 3, padding='same', activation='relu', name='block3_conv1')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu', name='block3_conv2')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu', name='block3_conv3')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu', name='block3_conv4')(x)
    x = layers.MaxPooling2D(2, strides=2, name='block3_pool')(x)    # 28x28

    # Block 4  (4 conv)
    x = layers.Conv2D(512, 3, padding='same', activation='relu', name='block4_conv1')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu', name='block4_conv2')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu', name='block4_conv3')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu', name='block4_conv4')(x)
    x = layers.MaxPooling2D(2, strides=2, name='block4_pool')(x)    # 14x14

    # Block 5  (4 conv)
    x = layers.Conv2D(512, 3, padding='same', activation='relu', name='block5_conv1')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu', name='block5_conv2')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu', name='block5_conv3')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu', name='block5_conv4')(x)
    x = layers.MaxPooling2D(2, strides=2, name='block5_pool')(x)    # 7x7

    # Classifier head
    x = layers.Flatten(name='flatten')(x)                           # 25088
    x = layers.Dense(4096, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu', name='fc2')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    return keras.Model(inputs=inputs, outputs=outputs, name='vgg19')
