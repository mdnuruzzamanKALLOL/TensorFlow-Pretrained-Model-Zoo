"""Extract 1280-d features from EfficientNetV2B2."""
import numpy as np,tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B2
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img,img_to_array
extractor=EfficientNetV2B2(weights="imagenet",include_top=False,pooling="avg",input_shape=(260,260,3))
extractor.trainable=False
print("Feature dim:",extractor.output_shape[-1])  # 1280
def extract(img_path):
    x=preprocess_input(np.expand_dims(img_to_array(load_img(img_path,target_size=(260,260))),0))
    return extractor.predict(x,verbose=0)[0]
if __name__=="__main__":
    v=extract("test.jpg"); print("Shape:",v.shape," L2:",np.linalg.norm(v))
