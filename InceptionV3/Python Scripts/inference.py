"""Run single-image inference with a saved InceptionV3 model."""
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

MODEL_PATH  = "inceptionv3_best.keras"
IMG_SIZE    = (299, 299)
CLASS_NAMES = ["class_0", "class_1"]   # replace with your class names

model = tf.keras.models.load_model(MODEL_PATH)

def predict(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    x   = img_to_array(img) / 255.0
    x   = np.expand_dims(x, 0)
    probs = model.predict(x, verbose=0)[0]
    idx   = int(np.argmax(probs))
    print(f"Image : {img_path}")
    print(f"Predicted: {CLASS_NAMES[idx]}  (confidence: {probs[idx]*100:.1f}%)")
    return CLASS_NAMES[idx], probs

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    predict(path)
