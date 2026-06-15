"""Two-phase fine-tuning for InceptionV3."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

TRAIN_DIR    = "dataset/train"
VAL_DIR      = "dataset/val"
IMG_SIZE     = (299, 299)
BATCH        = 16
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 20
NUM_CLASSES  = 10

train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20, width_shift_range=0.1,
    height_shift_range=0.1, horizontal_flip=True,
)
val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_gen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode="categorical")
val_data   = val_gen.flow_from_directory(VAL_DIR,   target_size=IMG_SIZE, batch_size=BATCH, class_mode="categorical")
NUM_CLASSES = len(train_data.class_indices)

# ── Phase 1: train head only ──────────────────────────────────────────────────
base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(299, 299, 3))
base_model.trainable = False

inputs  = keras.Input(shape=(299, 299, 3))
x       = base_model(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model   = keras.Model(inputs, outputs, name="InceptionV3_ft")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

cb1 = [
    ModelCheckpoint("inceptionv3_ft_phase1.keras", monitor="val_accuracy", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, min_lr=1e-7, verbose=1),
]
model.fit(train_data, validation_data=val_data, epochs=PHASE1_EPOCHS, callbacks=cb1)

# ── Phase 2: unfreeze from mixed7 onwards ────────────────────────────────────
layer_names = [l.name for l in base_model.layers]
mixed7_idx  = layer_names.index("mixed7")
for i, layer in enumerate(base_model.layers):
    layer.trainable = (i >= mixed7_idx)

trainable_ct = sum(1 for l in base_model.layers if l.trainable)
print(f"Unfrozen {trainable_ct} layers (from mixed7)")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

cb2 = [
    ModelCheckpoint("inceptionv3_ft_best.keras", monitor="val_accuracy", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, min_lr=1e-8, verbose=1),
]
model.fit(train_data, validation_data=val_data,
          epochs=PHASE1_EPOCHS + PHASE2_EPOCHS,
          initial_epoch=PHASE1_EPOCHS,
          callbacks=cb2)
print("Fine-tuning complete. Best model: inceptionv3_ft_best.keras")
