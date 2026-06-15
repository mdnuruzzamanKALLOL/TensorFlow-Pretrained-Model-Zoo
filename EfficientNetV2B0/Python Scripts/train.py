"""Train EfficientNetV2B0 from scratch."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from EfficientNetV2B0 import build_efficientnetv2b0

TRAIN_DIR="dataset/train"; VAL_DIR="dataset/val"; IMG_SIZE=(224,224); BATCH=32; EPOCHS=30
train_gen=ImageDataGenerator(rescale=1./255,rotation_range=20,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True,zoom_range=0.1)
val_gen=ImageDataGenerator(rescale=1./255)
train_data=train_gen.flow_from_directory(TRAIN_DIR,target_size=IMG_SIZE,batch_size=BATCH,class_mode="categorical")
val_data=val_gen.flow_from_directory(VAL_DIR,target_size=IMG_SIZE,batch_size=BATCH,class_mode="categorical")
NUM_CLASSES=len(train_data.class_indices)
model=build_efficientnetv2b0(num_classes=NUM_CLASSES)
model.compile(optimizer=keras.optimizers.Adam(1e-3),loss="categorical_crossentropy",metrics=["accuracy"])
cbs=[ModelCheckpoint("efficientnetv2b0_best.keras",monitor="val_accuracy",save_best_only=True,verbose=1),
     ReduceLROnPlateau(monitor="val_loss",factor=0.1,patience=5,min_lr=1e-7,verbose=1)]
model.fit(train_data,validation_data=val_data,epochs=EPOCHS,callbacks=cbs)
print("Done. Saved: efficientnetv2b0_best.keras")
