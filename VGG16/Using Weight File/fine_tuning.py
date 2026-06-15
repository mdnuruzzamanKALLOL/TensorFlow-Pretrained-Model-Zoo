import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

DATA_DIR      = './data'
BATCH_SIZE    = 32
NUM_CLASSES   = 10
PHASE1_EPOCHS = 10    # frozen backbone — train head
PHASE2_EPOCHS = 20    # unfreeze from block4 onward

# ── Build model ──
base_model = keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3),
)
x       = base_model.output
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dense(512, activation='relu')(x)
x       = layers.Dropout(0.5)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model   = keras.Model(inputs=base_model.input, outputs=outputs, name='vgg16_ft')

# ── Data generators ──
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    shear_range=0.1,
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

# ── Phase 1: Feature extraction (all conv blocks frozen) ──
print('Phase 1: Feature extraction — all conv blocks frozen')
base_model.trainable = False
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
history1 = model.fit(
    train_gen,
    epochs=PHASE1_EPOCHS,
    validation_data=val_gen,
    verbose=1,
)

# ── Phase 2: Fine-tune from block4_conv1 onward ──
print('\nPhase 2: Fine-tuning — block4 + block5 unfrozen (lr=1e-5)')
base_model.trainable = True
# Keep blocks 1-3 frozen (preserve low-level edge/texture features)
for layer in base_model.layers:
    if layer.name in ('block1_conv1', 'block1_conv2',
                      'block2_conv1', 'block2_conv2',
                      'block3_conv1', 'block3_conv2', 'block3_conv3'):
        layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'vgg16_ft.keras', monitor='val_accuracy',
        save_best_only=True, verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=3,
        min_lr=1e-8, verbose=1,
    ),
]
history2 = model.fit(
    train_gen,
    initial_epoch=PHASE1_EPOCHS,
    epochs=PHASE1_EPOCHS + PHASE2_EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1,
)

all_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
print(f'\nBest val_accuracy : {max(all_val_acc):.4f}')
print('Saved: vgg16_ft.keras')
