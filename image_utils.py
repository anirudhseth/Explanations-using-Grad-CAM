
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import json
from tensorflow import keras

class imagenet_utils():

    def load_test_img(img_path=None):
        if img_path==None:
            img_path='dog.jpg'
        img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        array = keras.preprocessing.image.img_to_array(img)
        array = np.expand_dims(array, axis=0)
        preprocess_input =keras.applications.vgg16.preprocess_input
        # decode_predictions = keras.applications.vgg16.decode_predictions
        preprocessed_input=preprocess_input(array)
        return img,preprocessed_input

    def overlay_heatmap(img,heatmap_lower_dim, cmap='COLORMAP_JET'):
        '''
        returns the heatmap with the applied colormap and the overlayed image 
        '''

        heatmap = imagenet_utils.resize_heatmap(img, heatmap_lower_dim)

        if cmap != None:
            heatmap = imagenet_utils.apply_cmap(img, heatmap, cmap)

        fused_img = heatmap * 0.4 + img
        fused_img = np.clip(fused_img, 0, 255).astype("uint8")
        # fused_img = cv2.cvtColor(fused_img, cv2.COLOR_BGR2RGB)
        return heatmap,fused_img

    def combine_guided_bp(img, heatmap_lower_dim, guided_bp, cmap='COLORMAP_JET'):
        '''
        returns the heatmap and the combined guided backprop images
        '''

        heatmap = imagenet_utils.resize_heatmap(img, heatmap_lower_dim)
        combined_img = (-1 * heatmap + 1) * guided_bp
        if cmap != None:
            heatmap = imagenet_utils.apply_cmap(img, heatmap, cmap)

        return heatmap, combined_img

    def apply_cmap(img, heatmap, cmap):
        cmap = getattr(cv2, cmap)
        heatmap = (heatmap * 255).astype("uint8")
        heatmap = cv2.applyColorMap(heatmap, cmap)

        return heatmap

    def resize_heatmap(img, heatmap_lower_dim):
        '''
        resizes the heatmap to the same size as the original image
        '''

        img = keras.preprocessing.image.img_to_array(img)
        heatmap = np.squeeze(heatmap_lower_dim)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = -1*heatmap +1
        heatmap = np.clip(heatmap, 0., 1.)

        return heatmap

        
