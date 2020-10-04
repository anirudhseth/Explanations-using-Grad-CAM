import tensorflow as tf 
import numpy as np
import cv2

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
        # TODO : image with batch dimension or without, better to add when getting predictions 
        with tf.GradientTape() as tape:
            inputs = tf.cast(img, tf.float32)  # use keras preprocessing instead before the function
            (layer_output, class_prediction) = self.gradcamModel(inputs) # score of class c , y^c after softmax  1, 14, 14, 512 and 1, 1000 , have batch dimension
            # TODO : compare score before and after softmax ref : https://eli5.readthedocs.io/en/latest/tutorials/keras-image-classifiers.html#choosing-the-target-class-target-prediction
            loss = class_prediction[:,index]  # 

        


        gradients = tape.gradient(loss, layer_output) # gradient of y^c wrt to a^k of the covolution layer given as input
        # [1, 14, 14, 512]) same dimensions as the output of the last layer
        layer_output = layer_output[0]
        gradients=gradients[0] # remove batch dimension [14, 14, 512]

        normalized_gradients = tf.divide(gradients, tf.sqrt(tf.reduce_mean(tf.square(gradients))) + tf.keras.backend.epsilon())
        # check directly normalizing https://donghwa-kim.github.io/backpropa_CNN.html
        
        neuron_importance_weights=tf.reduce_mean(normalized_gradients,axis=(0,1)) # ack in the paper  512

        gradcam = tf.reduce_sum(tf.multiply(neuron_importance_weights, layer_output), axis=-1)  #14 x 14

        gradcam = tf.keras.activations.relu(gradcam)
        gradcam = gradcam/np.max(gradcam)   # check without this 
        gradcam = cv2.resize(np.float32(gradcam), (224,224),interpolation=cv2.INTER_LINEAR)  #upscaling cv2 is buggy here
        return gradcam

    def overlay_heatmap(self,img,heatmap):
        '''
        returns the heatmap with the applied colormap and the overlayed image 
        '''
        heatmap3d = np.expand_dims(heatmap, axis=2)
        heatmap3d = np.tile(heatmap3d, [1,1,3])
        heatmap3d_colormap= cv2.applyColorMap(np.uint8(255*heatmap3d),cv2.COLORMAP_HOT)
        fused_img = cv2.addWeighted(img,0.7,heatmap3d_colormap,0.3,0)
        return heatmap3d_colormap,fused_img
class guided_backprop():

    def __init__(self):
        print('test')

class gradcam_plusplus():

    def __init__(self):
        print('test')