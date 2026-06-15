"""
Two-phase fine-tuning for ConvNeXtXLarge:
  Phase 1 (10 epochs) — frozen base, train new head only.
  Phase 2 (20 epochs) — unfreeze last 40% of base layers, low LR.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.convnext import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

BATCH         = 16
IMG_SIZE      = (224, 224)
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 20

# ── data ────────────────────────────────────────────────────────────────
def make_gen(augment=False):
    kwargs = dict(preprocessing_function=preprocess_input)
    if augment:
        kwargs.update(
            rotation_range=20, width_shift_range=0.1,
            height_shift_range=0.1, horizontal_flip=True, zoom_range=0.1,
        )
    return ImageDataGenerator(**kwargs)

train_data = make_gen(augment=True).flow_from_directory(
    "dataset/train", target_size=IMG_SIZE,
    batch_size=16, class_mode="categorical",
)
val_data = make_gen().flow_from_directory(
    "dataset/val", target_size=IMG_SIZE,
    batch_size=16, class_mode="categorical", shuffle=False,
)
NUM_CLASSES = len(train_data.class_indices)

# ── build model ─────────────────────────────────────────────────────────
base = tf.keras.applications.ConvNeXtXLarge(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
base.trainable = False

inputs  = keras.Input(shape=(224, 224, 3))
x       = base(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model   = keras.Model(inputs, outputs)

# ── Phase 1: feature extraction ─────────────────────────────────────────
model.compile(
    optimizer = keras.optimizers.Adam(1e-3),
    loss      = "categorical_crossentropy",
    metrics   = ["accuracy"],
)

cbs_1 = [
    ModelCheckpoint("convnext_xlarge_phase1_best.keras",
                    monitor="val_accuracy", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                      patience=3, min_lr=1e-7, verbose=1),
]

print("=== Phase 1: Feature Extraction ===")
model.fit(
    train_data, validation_data=val_data,
    epochs=PHASE1_EPOCHS, callbacks=cbs_1,
)

# ── Phase 2: fine-tune last 40% of base layers ──────────────────────────
base.trainable = True
n             = len(base.layers)
unfreeze_from = int(n * 0.60)   # unfreeze last 40%
for layer in base.layers[:unfreeze_from]:
    layer.trainable = False

model.compile(
    optimizer = keras.optimizers.Adam(1e-5),
    loss      = "categorical_crossentropy",
    metrics   = ["accuracy"],
)

cbs_2 = [
    ModelCheckpoint("convnext_xlarge_fine_tuned_best.keras",
                    monitor="val_accuracy", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                      patience=5, min_lr=1e-8, verbose=1),
]

print("=== Phase 2: Fine-Tuning (last 40% of layers) ===")
model.fit(
    train_data, validation_data=val_data,
    epochs          = PHASE1_EPOCHS + PHASE2_EPOCHS,
    initial_epoch   = PHASE1_EPOCHS,
    callbacks       = cbs_2,
)

model.save("convnext_xlarge_fine_tuned_final.keras")
print("Fine-tuning complete.")
