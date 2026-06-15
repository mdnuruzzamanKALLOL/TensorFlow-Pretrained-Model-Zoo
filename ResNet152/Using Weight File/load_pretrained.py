import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NUM_CLASSES = 10

base_model = keras.applications.ResNet152(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3),
)

x       = base_model.output
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model   = keras.Model(inputs=base_model.input, outputs=outputs, name='resnet152_transfer')

model.summary()

# Caffe-mode preprocessing: subtract BGR mean, do NOT divide by 255
dummy = tf.random.normal((1, 224, 224, 3)) * 127.5 + 127.5
dummy = keras.applications.resnet.preprocess_input(dummy)
out   = model(dummy, training=False)
print(f'Output shape : {out.shape}')
print(f'Total params : {model.count_params():,}')
