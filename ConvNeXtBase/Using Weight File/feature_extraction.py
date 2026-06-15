"""
Phase 1 only (frozen base): train a new head on top of pretrained ConvNeXtBase.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.convnext import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

BATCH    = 32
IMG_SIZE = (224, 224)
EPOCHS   = 10

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
    batch_size=32, class_mode="categorical",
)
val_data = make_gen().flow_from_directory(
    "dataset/val", target_size=IMG_SIZE,
    batch_size=32, class_mode="categorical", shuffle=False,
)
NUM_CLASSES = len(train_data.class_indices)

# ── model ───────────────────────────────────────────────────────────────
base = tf.keras.applications.ConvNeXtBase(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
base.trainable = False

inputs  = keras.Input(shape=(224, 224, 3))
x       = base(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model   = keras.Model(inputs, outputs)

model.compile(
    optimizer = keras.optimizers.Adam(1e-3),
    loss      = "categorical_crossentropy",
    metrics   = ["accuracy"],
)

callbacks = [
    ModelCheckpoint("convnext_base_fe_best.keras",
                    monitor="val_accuracy", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                      patience=3, min_lr=1e-7, verbose=1),
]

history = model.fit(
    train_data, validation_data=val_data,
    epochs=EPOCHS, callbacks=callbacks,
)

model.save("convnext_base_feature_extraction.keras")
print("Feature extraction complete.")
