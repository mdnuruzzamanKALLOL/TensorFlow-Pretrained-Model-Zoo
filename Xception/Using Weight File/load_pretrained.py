import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NUM_CLASSES = 10

# Load Keras Applications Xception with ImageNet weights (no top)
base_model = keras.applications.Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(299, 299, 3),
)

# Add custom classification head
x       = base_model.output
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model   = keras.Model(inputs=base_model.input, outputs=outputs, name='xception_transfer')

model.summary()

# Test forward pass with correct preprocessing
dummy = tf.random.normal((1, 299, 299, 3))
dummy = keras.applications.xception.preprocess_input(dummy)
out   = model(dummy, training=False)
print(f'Output shape : {out.shape}')
print(f'Total params : {model.count_params():,}')
