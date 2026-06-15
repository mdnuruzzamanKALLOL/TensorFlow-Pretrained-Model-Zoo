"""Single-image inference for EfficientNetV2B1."""
import sys,numpy as np,tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img,img_to_array
MODEL_PATH="efficientnetv2b1_best.keras"; IMG_SIZE=(240,240); CLASS_NAMES=["class_0","class_1"]
model=tf.keras.models.load_model(MODEL_PATH)
def predict(img_path):
    x=np.expand_dims(img_to_array(load_img(img_path,target_size=IMG_SIZE))/255.,0)
    probs=model.predict(x,verbose=0)[0]; idx=int(np.argmax(probs))
    print(f"{img_path}  ->  {CLASS_NAMES[idx]}  ({probs[idx]*100:.1f}%)"); return CLASS_NAMES[idx],probs
if __name__=="__main__": predict(sys.argv[1] if len(sys.argv)>1 else "test.jpg")
