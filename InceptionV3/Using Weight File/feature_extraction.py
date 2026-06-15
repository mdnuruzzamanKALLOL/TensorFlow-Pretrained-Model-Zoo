"""Extract deep features from InceptionV3 (no top layer)."""
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

extractor = InceptionV3(weights="imagenet", include_top=False, pooling="avg",
                        input_shape=(299, 299, 3))
extractor.trainable = False
print("Feature vector size:", extractor.output_shape[-1])  # 2048

def extract(img_path):
    img = load_img(img_path, target_size=(299, 299))
    x   = img_to_array(img)
    x   = preprocess_input(np.expand_dims(x, 0))
    return extractor.predict(x, verbose=0)[0]

if __name__ == "__main__":
    vec = extract("test.jpg")
    print("Feature vector shape:", vec.shape)
    print("L2-norm:", np.linalg.norm(vec))
