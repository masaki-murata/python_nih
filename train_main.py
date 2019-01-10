# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 16:28:22 2019

@author: murata
"""

import numpy as np
import pandas as pd
import os, csv, shutil, random, sys, datetime, re
from keras.utils import to_categorical
from PIL import Image
from keras.optimizers import Adam, SGD
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
#from keras.utils.training_utils import multi_gpu_model
from keras.utils import multi_gpu_model
from keras.models import Model
from keras.layers import Dense, Flatten, Input, Conv2D, BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import DenseNet169
from keras.applications.densenet import DenseNet201
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.models import Sequential
from keras import metrics
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from data_process import make_dataset, class_balance


class CNN():
    def __init__(self, 
                 input_shape,
                 ratio=[0.7,0.1,0.2],
                 ):
        self.ratio = ratio
        self.input_shape = input_shape
                
    def load_dataset(self,
                     path_to_data_label,
                     ):
        data, labels = {}, {}
        for group in ["train", "validation", "test"]:
            data[group] = np.load(path_to_data_label % ("validation", "data"))
            labels[group] = np.load(path_to_data_label % ("validation", "labels"))
        data["validation"], labels["validation"] = class_balance(data["validation"], labels["validation"])
        
        
    def make_model(self, hp_value):
        input_img = Input(shape=self.input_shape)
        x = Conv2D(filters=3, kernel_size=3, padding="same", activation="relu")(input_img)
        if hp_value["network"]=="VGG16":
            transfer = VGG16(include_top=False, weights=None, input_tensor=x)
        if hp_value["network"]=="VGG19":
            transfer = VGG19(include_top=False, weights=None, input_tensor=x)
        if hp_value["network"]=="DenseNet121":
            transfer = DenseNet121(include_top=False, weights=None, input_tensor=x)
        if hp_value["network"]=="DenseNet169":
            transfer = DenseNet169(include_top=False, weights=None, input_tensor=x)
        if hp_value["network"]=="DenseNet201":
            transfer = DenseNet201(include_top=False, weights=None, input_tensor=x)
        if hp_value["network"]=="InceptionV3":
            transfer = InceptionV3(include_top=False, weights=None, input_tensor=x)
        if hp_value["network"]=="ResNet50":
            transfer = ResNet50(include_top=False, weights=None, input_tensor=x)
        if hp_value["network"]=="Xception":
            transfer = Xception(include_top=False, weights=None, input_tensor=x)
    
        top_model = Sequential()
        top_model.add(Flatten(input_shape=transfer.output_shape[1:]))
        top_model.add(Dense(hp_value["dense_units1"], activation='relu'))
    #    top_model.add(Dropout(0.5))
        if hp_value["dense_layer_num"] > 1:
            top_model.add(Dense(2, activation='softmax'))
    
        model = Model(input=input_img, output=top_model(transfer.output))
    
    #    model = Model(input_img, output)
        if optimizer=="Adam":
            opt_generator = Adam(lr=hp_value["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        elif optimizer=="SGD":
            opt_generator = SGD(lr=hp_value["learning_rate"], momentum=hp_value["momentum"], decay=0.0, nesterov=False)
        model.compile(loss='binary_crossentropy', optimizer=opt_generator, metrics=['acc'])
    
        model.summary()
        
        return model



    