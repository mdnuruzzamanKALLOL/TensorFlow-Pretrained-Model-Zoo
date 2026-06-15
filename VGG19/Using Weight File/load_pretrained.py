import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NUM_CLASSES = 10

# Load VGG19 with ImageNet weights (convolutional base only)
base_model = keras.applications.VGG19(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3),
)

# Add custom classification head
x       = base_model.output
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dense(512, activation='relu')(x)
x       = layers.Dropout(0.5)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model   = keras.Model(inputs=base_model.input, outputs=outputs, name='vgg19_transfer')

model.summary()

# Test forward pass with correct VGG19 preprocessing
dummy = tf.random.normal((1, 224, 224, 3)) * 127.5 + 127.5   # simulate [0,255] range
dummy = keras.applications.vgg19.preprocess_input(dummy)
out   = model(dummy, training=False)
print(f'Output shape : {out.shape}')
print(f'Total params : {model.count_params():,}')
