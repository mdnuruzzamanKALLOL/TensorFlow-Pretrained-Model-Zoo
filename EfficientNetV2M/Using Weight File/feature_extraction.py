"""Extract 1280-d features from EfficientNetV2M."""
import numpy as np, tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2M
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

extractor = EfficientNetV2M(weights="imagenet", include_top=False,
                            pooling="avg", input_shape=(384, 384, 3))
extractor.trainable = False
print("Feature dim:", extractor.output_shape[-1])  # 1280

def extract(img_path):
    img = load_img(img_path, target_size=(384, 384))
    x   = preprocess_input(np.expand_dims(img_to_array(img), 0))
    return extractor.predict(x, verbose=0)[0]

if __name__ == "__main__":
    v = extract("test.jpg")
    print("Shape:", v.shape, " L2:", np.linalg.norm(v))
