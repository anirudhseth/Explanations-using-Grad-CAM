import tensorflow as tf 


class gradcam():
    '''
    Implimentation of GradCAM : Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
    url: https://arxiv.org/abs/1610.02391
    '''

    def __init__(self,model,layer_name):
        self.gradcamModel= tf.keras.Model(
        inputs = [model.inputs],
        outputs = [model.get_layer(layer_name).output, model.output])
    
    def computeHeatmap(self,img,index):
        

class guided_backprop():

    def __init__(self):
        print('test')

class gradcam_plusplus():

    def __init__(self):
        print('test')