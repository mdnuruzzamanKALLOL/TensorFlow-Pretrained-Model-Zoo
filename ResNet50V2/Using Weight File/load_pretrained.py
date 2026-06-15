import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NUM_CLASSES = 10

# Load ResNet50V2 with ImageNet weights (no top)
base_model = keras.applications.ResNet50V2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3),
)

# Add custom classification head
x       = base_model.output
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model   = keras.Model(inputs=base_model.input, outputs=outputs, name='resnet50v2_transfer')

model.summary()

# Test forward pass with correct ResNet50V2 preprocessing (scales to [-1, 1])
dummy = tf.random.normal((1, 224, 224, 3)) * 127.5 + 127.5   # simulate [0,255] range
dummy = keras.applications.resnet_v2.preprocess_input(dummy)
out   = model(dummy, training=False)
print(f'Output shape : {out.shape}')
print(f'Total params : {model.count_params():,}')
