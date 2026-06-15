import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import preprocess_input

DATA_DIR    = './data'
BATCH_SIZE  = 32
EPOCHS      = 15
NUM_CLASSES = 10

# ── Build model with frozen backbone ──
base_model = keras.applications.ResNet50V2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3),
)
base_model.trainable = False

x       = base_model.output
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model   = keras.Model(inputs=base_model.input, outputs=outputs, name='resnet50v2_fe')

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

trainable = int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
print(f'Trainable params (head only) : {trainable:,}')

# ── Data generators — ResNet50V2 preprocessing (scales to [-1, 1]) ──
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    f'{DATA_DIR}/train',
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
)
val_gen = val_datagen.flow_from_directory(
    f'{DATA_DIR}/val',
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        'resnet50v2_fe.keras', monitor='val_accuracy',
        save_best_only=True, verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=3,
        min_lr=1e-7, verbose=1,
    ),
]

print('Feature Extraction — backbone frozen, head only')
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1,
)
print(f"Best val_accuracy : {max(history.history['val_accuracy']):.4f}")
print('Saved: resnet50v2_fe.keras')
