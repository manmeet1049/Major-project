import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = tf.keras.models.load_model("models/weapon_model.h5")

def encoding(path):
    face = plt.imread(path)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)
    return face

def prediction(image):
    encode_image = encoding(image)
    (no_knife,knife) = model.predict(encode_image)[0]
    if knife>no_knife:
        return "weapon"
    else:
        return "no weapon"



