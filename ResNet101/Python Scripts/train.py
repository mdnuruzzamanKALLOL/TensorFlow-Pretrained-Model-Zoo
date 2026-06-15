import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from resnet101 import build_resnet101

DATA_DIR    = './data'
BATCH_SIZE  = 16
EPOCHS      = 30
NUM_CLASSES = 10

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
)
val_datagen = ImageDataGenerator(rescale=1./255)

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

model = build_resnet101(num_classes=NUM_CLASSES)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint(
        'resnet101_best.keras', monitor='val_accuracy',
        save_best_only=True, verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=5,
        min_lr=1e-7, verbose=1,
    ),
]

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1,
)
print(f"Best val_accuracy : {max(history.history['val_accuracy']):.4f}")
print('Model saved: resnet101_best.keras')
