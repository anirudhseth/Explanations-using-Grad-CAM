import tensorflow as tf
from tensorflow.keras import Model

class load_model():

    def __init__(self,name):
        '''
            class to load pretrained models.
        '''
        if name=='VGG16':
            self.model_name='VGG16'
            from tensorflow.keras.applications import VGG16
            model_vgg16 = VGG16
            self.model=model_vgg16(weights="imagenet")
            
        elif name=='resnet':
            self.model_name='resnet'
            from tensorflow.keras.applications import ResNet50
            model_resnet = ResNet50
            self.model=model_resnet(weights="imagenet")



    
