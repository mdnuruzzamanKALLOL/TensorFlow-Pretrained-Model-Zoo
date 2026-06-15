import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.convnext import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load Keras pretrained ConvNeXtTiny (ImageNet weights, TF 2.11+)
base = tf.keras.applications.ConvNeXtTiny(
    include_top  = False,
    weights      = "imagenet",
    input_shape  = (224, 224, 3),
    pooling      = "avg",
)

model = keras.Sequential([
    base,
    layers.Dense(1000, activation="softmax"),
])
model.summary()


def predict(img_path):
    img  = load_img(img_path, target_size=(224, 224))
    x    = img_to_array(img)
    x    = np.expand_dims(x, axis=0)
    x    = preprocess_input(x)   # ConvNeXt: ImageNet mean/std normalisation
    pred = model.predict(x, verbose=0)[0]
    top5 = pred.argsort()[-5:][::-1]
    print("Top-5 class indices:", top5)
    print("Top-5 probabilities:", pred[top5].round(4))
    return pred


if __name__ == "__main__":
    predict(sys.argv[1])
