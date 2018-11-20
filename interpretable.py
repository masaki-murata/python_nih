# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 16:04:04 2018

@author: murata
"""
from keras.models import load_model
import numpy as np

def grad_cam(data, label, 
             layer_name,
             path_to_model="",
             ):
    model = load_model(path_to_model)
    
    predictions = model.predict(data)
    class_idx = np.argmax(predictions[0])
    class_output = model.output[:, class_idx]

    conv_output = model.get_layer(layer_name).output