
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import json
from tensorflow import keras

class imagenet_utils():

    def load_test_img():
        img_path='dog.jpg'
        img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        array = keras.preprocessing.image.img_to_array(img)
        array = np.expand_dims(array, axis=0)
        preprocess_input =keras.applications.vgg16.preprocess_input
        # decode_predictions = keras.applications.vgg16.decode_predictions
        preprocessed_input=preprocess_input(array)
        return img,preprocessed_input

        
