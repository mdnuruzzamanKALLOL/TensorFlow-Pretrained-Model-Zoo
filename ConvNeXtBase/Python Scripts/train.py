import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from ConvNeXtBase import build_convnext_base, LayerScale

BATCH    = 32
IMG_SIZE = (224, 224)
EPOCHS   = 30

train_gen = ImageDataGenerator(
    rescale            = 1.0 / 255,
    rotation_range     = 20,
    width_shift_range  = 0.1,
    height_shift_range = 0.1,
    horizontal_flip    = True,
    zoom_range         = 0.1,
)
val_gen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_gen.flow_from_directory(
    "dataset/train",
    target_size = IMG_SIZE,
    batch_size  = 32,
    class_mode  = "categorical",
)
val_data = val_gen.flow_from_directory(
    "dataset/val",
    target_size = IMG_SIZE,
    batch_size  = 32,
    class_mode  = "categorical",
    shuffle     = False,
)

NUM_CLASSES = len(train_data.class_indices)

model = build_convnext_base(num_classes=NUM_CLASSES)
model.compile(
    optimizer = keras.optimizers.Adam(1e-3),
    loss      = "categorical_crossentropy",
    metrics   = ["accuracy"],
)

callbacks = [
    ModelCheckpoint(
        "convnext_base_best.keras",
        monitor        = "val_accuracy",
        save_best_only = True,
        verbose        = 1,
    ),
    ReduceLROnPlateau(
        monitor  = "val_loss",
        factor   = 0.1,
        patience = 5,
        min_lr   = 1e-7,
        verbose  = 1,
    ),
]

history = model.fit(
    train_data,
    validation_data = val_data,
    epochs          = EPOCHS,
    callbacks       = callbacks,
)

model.save("convnext_base_final.keras")
print("Training complete. Model saved.")
