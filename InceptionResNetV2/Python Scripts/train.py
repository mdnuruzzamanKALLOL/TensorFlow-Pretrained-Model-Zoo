"""Train InceptionResNetV2 from scratch."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from InceptionResNetV2 import build_inception_resnet_v2

TRAIN_DIR   = "dataset/train"
VAL_DIR     = "dataset/val"
IMG_SIZE    = (299, 299)
BATCH       = 16
EPOCHS      = 30
NUM_CLASSES = 10

train_gen = ImageDataGenerator(
    rescale=1./255, rotation_range=20, width_shift_range=0.1,
    height_shift_range=0.1, horizontal_flip=True, zoom_range=0.1,
)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode="categorical")
val_data   = val_gen.flow_from_directory(VAL_DIR,   target_size=IMG_SIZE, batch_size=BATCH, class_mode="categorical")
NUM_CLASSES = len(train_data.class_indices)

model = build_inception_resnet_v2(num_classes=NUM_CLASSES)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    ModelCheckpoint("inceptionresnetv2_best.keras", monitor="val_accuracy", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, min_lr=1e-7, verbose=1),
]

history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=callbacks)
print("Training complete. Best model saved to inceptionresnetv2_best.keras")
