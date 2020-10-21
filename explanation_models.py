import tensorflow as tf 
import numpy as np
import cv2
from skimage.color import rgb2gray
from tensorflow import keras


class gradcam():
    '''
    Implimentation of GradCAM : Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
    url: https://arxiv.org/abs/1610.02391
    '''

    def __init__(self,model,layer_name,input_dim):
        self.gradcamModel= tf.keras.Model(
        inputs = [model.inputs],
        outputs = [model.get_layer(layer_name).output, model.output])
        self.input_dim=input_dim
    
    def get_heatmap(self,img,index=None,CounterfactualExp=False):
 
        with tf.GradientTape() as tape:
            (layer_output, class_prediction) = self.gradcamModel(img) 
            if(index==None):
                index=tf.argmax(class_prediction[0])  # use the index with best score if none provided
            tape.watch(layer_output)
            loss = class_prediction[:,index]  # 
        gradients = tape.gradient(loss, layer_output) # gradient of y^c wrt to a^k of the covolution layer given as input
        # [1, 14, 14, 512]) same dimensions as the output of the last layer
        layer_output = layer_output.numpy()[0]
        if(CounterfactualExp):
            neuron_importance_weights=tf.reduce_mean(gradients, axis=(0, 1, 2)).numpy()
        else:
            neuron_importance_weights=tf.reduce_mean(gradients, axis=(0, 1, 2)).numpy()
        for i in range(neuron_importance_weights.shape[-1]):
          layer_output[:, :, i] *= neuron_importance_weights[i]
        gradcam = np.mean(layer_output, axis=-1)
        gradcam = tf.keras.activations.relu(gradcam)
        gradcam = gradcam/np.max(gradcam)    
        return gradcam

class guided_backprop():
    '''
    Implimentation of Guided Backprop 
    url: https://www.cs.toronto.edu/~guerzhoy/321/lec/W07/HowConvNetsSee.pdf
    '''

    # https://www.tensorflow.org/api_docs/python/tf/custom_gradient

    @tf.custom_gradient
    def guided_Relu(self,x):
        def grad(dy):
            return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
        return tf.nn.relu(x), grad

    def __init__(self,model,layer_name,input_dim):
        self.guided_backpropModel= tf.keras.Model(
        inputs = [model.inputs],
        outputs = [model.get_layer(layer_name).output])
        self.input_dim=input_dim
        for layer in self.guided_backpropModel.layers[1:]:
            if hasattr(layer,"activation") and layer.activation == tf.keras.activations.relu :
                layer.activation = self.guided_Relu
    
    def get_heatmap(self,img):

        with tf.GradientTape() as tape:
            img = tf.cast(img, tf.float32)
            tape.watch(img)
            layer_output = self.guided_backpropModel(img)

        grads = tape.gradient(layer_output, img)[0]

        guided_backprop_map = cv2.resize(np.asarray(grads), self.input_dim)
        guided_backprop_map_gray = rgb2gray(guided_backprop_map)
        return guided_backprop_map_gray

class gradcam_robust():

    def __init__(self,model,layers,input_dim,gradcam_variant):
        self.gradcam_variant=gradcam_variant
        self.layers=layers
        self.model=model
        self.input_dim=input_dim

    def get_heatmap(self,img,index=None):
        robust_heatmap=[]
        for layer in self.layers:
            if self.gradcam_variant=='GradCAM':
                g= gradcam(self.model,layer,self.input_dim)
            elif self.gradcam_variant=='GradCAMpp':
                g= gradcam_plusplus(self.model,layer,self.input_dim)
            heatmap=g.get_heatmap(img)
            robust_heatmap.append(heatmap)
            del g
        return robust_heatmap

class gradcam_plusplus():
    '''
    Implimentation of Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks
    url: https://arxiv.org/abs/1710.11063
    '''

    def __init__(self,model,layer_name,input_dim):
        self.gradcamModel= tf.keras.Model(
        inputs = [model.inputs],
        outputs = [model.get_layer(layer_name).output, model.output])
        self.input_dim=input_dim
    
    def get_heatmap(self,img,index=None):
 
        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                with tf.GradientTape() as tape3:
                    (layer_output, class_prediction) = self.gradcamModel(img) 
                    if(index==None):
                        index=tf.argmax(class_prediction[0])
                    loss = class_prediction[:,index]  
                    gradients1 = tape3.gradient(loss, layer_output) 
                gradients2 = tape2.gradient(gradients1, layer_output)
            gradients3 = tape1.gradient(gradients2, layer_output)

        global_sum = np.sum(layer_output, axis=(0, 1, 2))

        neuron_importance_weights_num = gradients2[0]
        neuron_importance_weights_denom = 2*gradients2[0] + gradients3[0]*global_sum
        neuron_importance_weights_denom = np.where(neuron_importance_weights_denom != 0.0, neuron_importance_weights_denom, 1e-10)
        
        neuron_importance_weights = neuron_importance_weights_num/neuron_importance_weights_denom
        Z = np.sum(neuron_importance_weights, axis=(0,1))
        neuron_importance_weights /= Z

        weights = tf.keras.activations.relu(gradients1[0]) 
        weights = np.sum(weights*neuron_importance_weights, axis=(0,1))
        gradcampp = np.sum(weights*layer_output[0], axis=2)

        gradcampp = tf.keras.activations.relu(gradcampp)
        gradcampp=gradcampp/np.max(gradcampp) 

        return gradcampp
  