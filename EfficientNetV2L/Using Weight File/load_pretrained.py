"""ImageNet-pretrained EfficientNetV2L — head-only training."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

TRAIN_DIR = "dataset/train"
VAL_DIR   = "dataset/val"
IMG_SIZE  = (480, 480)
BATCH     = 4

train_gen = ImageDataGenerator(preprocessing_function=preprocess_input,
    rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
val_gen   = ImageDataGenerator(preprocessing_function=preprocess_input)
train_data = train_gen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode="categorical")
val_data   = val_gen.flow_from_directory(VAL_DIR,   target_size=IMG_SIZE, batch_size=BATCH, class_mode="categorical")
NUM_CLASSES = len(train_data.class_indices)

base_model = EfficientNetV2L(weights="imagenet", include_top=False, input_shape=(480, 480, 3))
base_model.trainable = False

inputs  = keras.Input(shape=(480, 480, 3))
x       = base_model(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dropout(0.4)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model   = keras.Model(inputs, outputs, name="EfficientNetV2L_pretrained")

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy", metrics=["accuracy"])
callbacks = [
    ModelCheckpoint("efficientnetv2l_pretrained.keras", monitor="val_accuracy", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, min_lr=1e-7, verbose=1),
]
model.fit(train_data, validation_data=val_data, epochs=10, callbacks=callbacks)
print("Saved: efficientnetv2l_pretrained.keras")
