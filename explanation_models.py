import tensorflow as tf 
import numpy as np
import cv2
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
    
    def get_heatmap(self,img,index=None):
 
        with tf.GradientTape() as tape:
           
           
            (layer_output, class_prediction) = self.gradcamModel(img) # score of class c , y^c after softmax  1, 14, 14, 512 and 1, 1000 , have batch dimension
            # TODO : compare score before and after softmax ref : https://eli5.readthedocs.io/en/latest/tutorials/keras-image-classifiers.html#choosing-the-target-class-target-prediction
            if(index==None):
                index=tf.argmax(class_prediction[0])
            tape.watch(layer_output)
            loss = class_prediction[:,index]  # 
        gradients = tape.gradient(loss, layer_output) # gradient of y^c wrt to a^k of the covolution layer given as input
        # [1, 14, 14, 512]) same dimensions as the output of the last layer
        layer_output = layer_output.numpy()[0]
        # gradients=gradients[0] # remove batch dimension [14, 14, 512]

        # normalized_gradients = tf.divide(gradients, tf.sqrt(tf.reduce_mean(tf.square(gradients))) + tf.keras.backend.epsilon())
        # check directly normalizing https://donghwa-kim.github.io/backpropa_CNN.html
        
        # neuron_importance_weights=tf.reduce_mean(normalized_gradients,axis=(0,1)) # ack in the paper  512
        neuron_importance_weights=tf.reduce_mean(gradients, axis=(0, 1, 2)).numpy()
        # gradcam_test = tf.reduce_sum(tf.multiply(neuron_importance_weights, layer_output), axis=-1)  #14 x 14
        
        for i in range(neuron_importance_weights.shape[-1]):
          layer_output[:, :, i] *= neuron_importance_weights[i]
        gradcam = np.mean(layer_output, axis=-1)
        gradcam = tf.keras.activations.relu(gradcam)
        gradcam = gradcam/np.max(gradcam)   # check without this 
        # gradcam = cv2.resize(np.float32(gradcam), self.input_dim,interpolation=cv2.INTER_LINEAR)  #upscaling cv2 is buggy here
        return gradcam

    def overlay_heatmap(self,img,heatmap_lower_dim):
        '''
        returns the heatmap with the applied colormap and the overlayed image 
        '''
        img=keras.preprocessing.image.img_to_array(img)
        heatmap=np.squeeze(heatmap_lower_dim)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = (heatmap*255).astype("uint8")
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        fused_img = heatmap * 0.4 + img
        fused_img = np.clip(fused_img,0,255).astype("uint8")
        # fused_img = cv2.cvtColor(fused_img, cv2.COLOR_BGR2RGB)

        return heatmap,fused_img  

class guided_backprop():

    def __init__(self):
        print('test')

class gradcam_robust():

    def __init__(self,model,layers,input_dim):
        self.layers=layers
        self.model=model
        self.input_dim=input_dim

    def get_heatmap(self,img,index=None):
        robust_heatmap=[]
        for layer in self.layers:
            g= gradcam(self.model,layer,self.input_dim)
            heatmap=g.get_heatmap(img)
            robust_heatmap.append(heatmap)
            del g
        return robust_heatmap
        # g= gradcam(model,layer_name,input_dim)
        # heatmap=g.get_heatmap(img)

    def overlay_heatmap(self,img,heatmap_lower_dim):
        '''
        returns the heatmap with the applied colormap and the overlayed image 
        '''
        img=keras.preprocessing.image.img_to_array(img)
        heatmap=np.squeeze(heatmap_lower_dim)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = (heatmap*255).astype("uint8")
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        fused_img = heatmap * 0.4 + img
        fused_img = np.clip(fused_img,0,255).astype("uint8")
        # fused_img = cv2.cvtColor(fused_img, cv2.COLOR_BGR2RGB)

        return heatmap,fused_img 