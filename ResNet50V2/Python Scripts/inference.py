import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from resnet50v2 import build_resnet50v2

MODEL_PATH  = 'resnet50v2_best.keras'
NUM_CLASSES = 10
CLASS_NAMES = [f'class_{i}' for i in range(NUM_CLASSES)]

model = build_resnet50v2(num_classes=NUM_CLASSES)
model.load_weights(MODEL_PATH)
model.trainable = False


def predict(image_path, top_k=5):
    img    = keras.utils.load_img(image_path, target_size=(224, 224))
    arr    = keras.utils.img_to_array(img) / 255.0
    tensor = tf.expand_dims(arr, 0)
    probs  = model(tensor, training=False)[0].numpy()
    top_idx = probs.argsort()[::-1][:top_k]
    print(f'\nTop-{top_k} predictions for: {image_path}')
    for rank, idx in enumerate(top_idx, 1):
        print(f'  {rank}. {CLASS_NAMES[idx]:<20} {probs[idx]*100:.2f}%')


if __name__ == '__main__':
    predict(sys.argv[1] if len(sys.argv) > 1 else 'test.jpg')
