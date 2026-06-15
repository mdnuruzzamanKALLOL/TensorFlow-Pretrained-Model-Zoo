"""Two-phase fine-tuning for EfficientNetV2L."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

TRAIN_DIR     = "dataset/train"
VAL_DIR       = "dataset/val"
IMG_SIZE      = (480, 480)
BATCH         = 4
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 20

train_gen = ImageDataGenerator(preprocessing_function=preprocess_input,
    rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
val_gen   = ImageDataGenerator(preprocessing_function=preprocess_input)
train_data = train_gen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode="categorical")
val_data   = val_gen.flow_from_directory(VAL_DIR,   target_size=IMG_SIZE, batch_size=BATCH, class_mode="categorical")
NUM_CLASSES = len(train_data.class_indices)

# ── Phase 1: head only ────────────────────────────────────────────────────────
base_model = EfficientNetV2L(weights="imagenet", include_top=False, input_shape=(480, 480, 3))
base_model.trainable = False

inputs  = keras.Input(shape=(480, 480, 3))
x       = base_model(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dropout(0.4)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model   = keras.Model(inputs, outputs, name="EfficientNetV2L_ft")

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy", metrics=["accuracy"])
cb1 = [ModelCheckpoint("efficientnetv2l_ft_p1.keras", monitor="val_accuracy", save_best_only=True, verbose=1),
       ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, min_lr=1e-7, verbose=1)]
model.fit(train_data, validation_data=val_data, epochs=PHASE1_EPOCHS, callbacks=cb1)

# ── Phase 2: unfreeze last 30% of backbone (stages 6 + most of stage 5) ──────
n = len(base_model.layers)
unfreeze_from = int(n * 0.70)
for i, layer in enumerate(base_model.layers):
    layer.trainable = (i >= unfreeze_from)
print(f"Unfrozen {sum(l.trainable for l in base_model.layers)}/{n} layers")

model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss="categorical_crossentropy", metrics=["accuracy"])
cb2 = [ModelCheckpoint("efficientnetv2l_ft_best.keras", monitor="val_accuracy", save_best_only=True, verbose=1),
       ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, min_lr=1e-8, verbose=1)]
model.fit(train_data, validation_data=val_data,
          epochs=PHASE1_EPOCHS + PHASE2_EPOCHS,
          initial_epoch=PHASE1_EPOCHS, callbacks=cb2)
print("Done. Best model: efficientnetv2l_ft_best.keras")
