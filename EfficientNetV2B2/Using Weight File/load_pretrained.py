"""ImageNet-pretrained EfficientNetV2B2 — head-only training."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2B2
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

TRAIN_DIR="dataset/train"; VAL_DIR="dataset/val"; IMG_SIZE=(260,260); BATCH=32
train_gen=ImageDataGenerator(preprocessing_function=preprocess_input,rotation_range=20,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True)
val_gen=ImageDataGenerator(preprocessing_function=preprocess_input)
train_data=train_gen.flow_from_directory(TRAIN_DIR,target_size=IMG_SIZE,batch_size=BATCH,class_mode="categorical")
val_data=val_gen.flow_from_directory(VAL_DIR,target_size=IMG_SIZE,batch_size=BATCH,class_mode="categorical")
NUM_CLASSES=len(train_data.class_indices)
base_model=EfficientNetV2B2(weights="imagenet",include_top=False,input_shape=(260,260,3))
base_model.trainable=False
inputs=keras.Input(shape=(260,260,3)); x=base_model(inputs,training=False)
x=layers.GlobalAveragePooling2D()(x); x=layers.Dropout(0.3)(x)
outputs=layers.Dense(NUM_CLASSES,activation="softmax")(x)
model=keras.Model(inputs,outputs,name="EfficientNetV2B2_pretrained")
model.compile(optimizer=keras.optimizers.Adam(1e-3),loss="categorical_crossentropy",metrics=["accuracy"])
cbs=[ModelCheckpoint("efficientnetv2b2_pretrained.keras",monitor="val_accuracy",save_best_only=True,verbose=1),
     ReduceLROnPlateau(monitor="val_loss",factor=0.1,patience=5,min_lr=1e-7,verbose=1)]
model.fit(train_data,validation_data=val_data,epochs=10,callbacks=cbs)
print("Saved: efficientnetv2b2_pretrained.keras")
