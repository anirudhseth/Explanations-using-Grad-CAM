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
    
    def get_heatmap(self,img,index):
        with tf.GradientTape() as tape:
            inputs = tf.cast(img, tf.float32)
            (layer_output, class_prediction) = self.gradcamModel(inputs) # score of class c , y^c after softmax
            # TODO : compare score before and after softmax ref : https://eli5.readthedocs.io/en/latest/tutorials/keras-image-classifiers.html#choosing-the-target-class-target-prediction
            loss = class_prediction[:,index]  # 

        gradients = tape.gradient(loss, layer_output) # gradient of y^c wrt to a^k of the covolution layer given as input
        # TODO : global average pool to get ack
        

class guided_backprop():

    def __init__(self):
        print('test')

class gradcam_plusplus():

    def __init__(self):
        print('test')