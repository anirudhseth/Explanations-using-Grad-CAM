import tensorflow as tf
from tensorflow.keras import Model
import numpy as np

class load_model():

    def __init__(self,name):
        '''
            class to load pretrained models.
        '''
        if name=='VGG16':
            self.model_name='VGG16'
            self.input_dim=(224,224)
            from tensorflow.keras.applications import VGG16
            model_vgg16 = VGG16
            self.model=model_vgg16(weights="imagenet")

        elif name=='resnet':
            self.model_name='resnet'
            self.input_dim=(224,224)
            from tensorflow.keras.applications import ResNet50
            model_resnet = ResNet50
            self.model=model_resnet(weights="imagenet")

        elif name=='inceptionv3':
            self.model_name='InceptionV3'
            self.input_dim=(299,299)
            from tensorflow.keras.applications import InceptionV3
            model_InceptionV3 = InceptionV3
            self.model=model_InceptionV3(weights="imagenet")

        elif name=='mobilenetv2':
            self.model_name='MobileNet_v2'
            self.input_dim=(224,224)
            from tensorflow.keras.applications import MobileNetV2
            model_mobilenetv2 = MobileNetV2
            self.model=model_mobilenetv2(weights="imagenet")
        
        elif name=='densenet':
            self.model_name='DenseNet121'
            self.input_dim=(224,224)
            from tensorflow.keras.applications import DenseNet121
            model_densenet = DenseNet121
            self.model=model_densenet(weights="imagenet")
        
        # TODO: StyleGAN  (compatibility issues) 


    def get_predictions(self,img):
        '''
        returns probabilities for each class
        '''
        return self.model.predict(img)

    def class_index(self,img):
        '''
        returns the class index 
        '''
        from tensorflow.keras.applications.imagenet_utils import decode_predictions
        index=np.argmax(self.get_predictions(img)[0])
        decode_predictions(self.get_predictions(img))
        return index


    
