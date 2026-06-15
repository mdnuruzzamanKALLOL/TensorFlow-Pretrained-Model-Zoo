import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NUM_CLASSES = 10

# Load NASNetMobile with ImageNet weights (no top)
base_model = keras.applications.NASNetMobile(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3),
)

x       = base_model.output
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model   = keras.Model(inputs=base_model.input, outputs=outputs,
                      name='nasnetmobile_transfer')

model.summary()

# NASNetMobile preprocessing: x/127.5 - 1.0  -> [-1, 1]
dummy = tf.random.normal((1, 224, 224, 3)) * 127.5 + 127.5
dummy = keras.applications.nasnet.preprocess_input(dummy)
out   = model(dummy, training=False)
print(f'Output shape : {out.shape}')
print(f'Total params : {model.count_params():,}')
