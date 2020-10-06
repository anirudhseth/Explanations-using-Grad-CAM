
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

class imagenet_utils():

    def load_test_img():
        orig_img = image.load_img('data/dog.jpg',target_size=(224,224))
        orig_img = np.asarray(orig_img)
        img = np.expand_dims(orig_img, axis=0)
        return orig_img,img

        
