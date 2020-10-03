import tensorflow as tf
from tensorflow.keras import Model
class load_model():

    def __init__(self,name):
        if name=='VGG16':
            self.model_name='VGG16'
            from tensorflow.keras.applications import VGG16
            model_vgg16 = VGG16
            self.model=model_vgg16(weights="imagenet")
        else:
            print('To do')


    
