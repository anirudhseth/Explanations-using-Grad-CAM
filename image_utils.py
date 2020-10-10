
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import json
class imagenet_utils():

    def load_test_img():

        # !wget https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
        # f = open("imagenet_class_index.json", "r")
        # class_name_dict = json.load(f)
        # print(class_name_dict['202'])
        # !wget -O dog.jpg https://i.insider.com/536aa78069bedddb13c60c3a?width=600&format=jpeg&auto=webp
        dog_img='dog.jpg'
        orig_img = image.load_img(dog_img,target_size=(224,224))
        # orig_img = image.load_img('data/dog.jpg',target_size=(224,224))
        orig_img = np.asarray(orig_img)
        img = np.expand_dims(orig_img, axis=0)
        return orig_img,img

        
