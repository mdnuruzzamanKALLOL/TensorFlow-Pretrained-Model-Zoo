import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

MODEL_PATH  = "efficientnet_b6_best.keras"
IMG_SIZE    = (528, 528)
CLASS_NAMES = []   # fill with your class labels

model = tf.keras.models.load_model(MODEL_PATH)


def predict_image(img_path, class_names=CLASS_NAMES):
    img   = load_img(img_path, target_size=IMG_SIZE)
    x     = img_to_array(img) / 255.0
    x     = np.expand_dims(x, axis=0)
    probs = model.predict(x, verbose=0)[0]
    idx   = np.argmax(probs)
    label = class_names[idx] if class_names else str(idx)
    print(f"Prediction : {label}")
    print(f"Confidence : {probs[idx]*100:.1f}%")
    return label, probs


if __name__ == "__main__":
    predict_image(sys.argv[1])
