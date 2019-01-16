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
from random_search import chose_hyperparam

base_dir = os.getcwd()+"/"
if not re.search("nih_python", base_dir):
    base_dir = base_dir + "/xray/nih_python/"

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
if_DLB=False
if os.name=='posix' and if_DLB:
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list="2", # specify GPU number
            allow_growth=True
        )
    )
    
    set_session(tf.Session(config=config))


class CNN():
    def __init__(self, 
                 pathology,
                 input_shape,
                 if_single_pathology,
                 ratio=[0.7,0.1,0.2],
                 ):
        self.ratio = ratio
        self.input_shape = input_shape
        self.pathology = pathology
        self.size = input_shape[0]
        self.if_single_pathology = if_single_pathology
                
    def load_dataset(self,
#                     path_to_data_label, #  "%s_size%d_%s_%s.npy" % (group, size, pathology, data/labels)
                     ):
        path_to_data_label = base_dir+"../nih_data/ratio_t%.2fv%.2ft%.2f/" % tuple(self.ratio)
        if not self.if_single_pathology:
            path_to_data_label = path_to_data_label[:-1] + "_multipathology/"
        path_to_data_label = path_to_data_label + "%s_size%d_%s_%s.npy"
        self.data, self.labels = {}, {}
        for group in ["train", "validation", "test"]:
            self.data[group] = np.load(path_to_data_label % (group, self.size, self.pathology, "data"))
            self.labels[group] = np.load(path_to_data_label % (group, self.size, self.pathology, "labels"))
        self.data["validation"], self.labels["validation"] = class_balance(self.data["validation"], self.labels["validation"])
        
        
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
        if hp_value["optimizer"]=="Adam":
            opt_generator = Adam(lr=hp_value["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        elif hp_value["optimizer"]=="SGD":
            opt_generator = SGD(lr=hp_value["learning_rate"], momentum=hp_value["momentum"], decay=0.0, nesterov=False)
        model.compile(loss='binary_crossentropy', optimizer=opt_generator, metrics=['acc'])
    
        model.summary()
        
        return model


    def train(self, hp_value, nb_gpus,
              ):
        # load dataset
        self.load_dataset()
        # make model
        model = self.make_model(hp_value)
        if int(nb_gpus) > 1:
            model_multiple_gpu = multi_gpu_model(model, gpus=nb_gpus)
        else:
            model_multiple_gpu = model


def main():
    hp_value = chose_hyperparam()
    nb_gpus=1
    if_single_pathology=False
    
    cnn = CNN(pathology="Effusion", input_shape=(128,128,3), if_single_pathology=if_single_pathology)
    cnn.train(hp_value,nb_gpus)

if __name__ == '__main__':
    main()        

    