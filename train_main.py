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
from nih import loss_ambiguous, auc
from hyperparameters import chose_hyperparam

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
    # initialize
    def __init__(self, 
                 pathology,
                 input_shape,
                 if_single_pathology,
                 if_loss_ambiguous,
                 nb_gpus=0,
                 ratio=[0.7,0.1,0.2],
                 ):
        self.ratio = ratio
        self.input_shape = input_shape
        self.pathology = pathology
        self.size = input_shape[0]
        self.if_single_pathology = if_single_pathology
        self.nb_gpus = nb_gpus
        
    
    # load train, validation, test dataset
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
        
    
    # make cnn model
    def make_model(self, hp_value):
        # set architecture
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
        # set optimizer
        if hp_value["optimizer"]=="Adam":
            opt_generator = Adam(lr=hp_value["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        elif hp_value["optimizer"]=="SGD":
            opt_generator = SGD(lr=hp_value["learning_rate"], momentum=hp_value["momentum"], decay=0.0, nesterov=False)
        
        # set loss function
        if hp_value["if_loss_ambiguous"]:
            def loss_amb(y_true, y_pred):
                return loss_ambiguous(y_true, y_pred, eps=0.1)
            loss=loss_amb
        else:
            loss="binary_crossentropy"
        
        # compile the model
        model.compile(loss=loss, optimizer=opt_generator, metrics=['acc'])
    
        model.summary()
        
        return model

    
    # train a cnn model
    def train(self, hp_value, path_to_model_save,
              ):
        # load dataset
        self.load_dataset()
        # make model
        model = self.make_model(hp_value)
        if int(nb_gpus) > 1:
            model_multiple_gpu = multi_gpu_model(model, gpus=nb_gpus)
        else:
            model_multiple_gpu = model
            
        datagen = ImageDataGenerator(rotation_range=hp_value["rotation_range"],
                                     width_shift_range=hp_value["width_shift_range"],
                                     height_shift_range=hp_value["height_shift_range"],
                                     zoom_range=hp_value["zoom_range"],
                                     )   
        
        val_auc_0, count_patience = 0, 0
        for ep in hp_value["epochs"]:
            # set training data class balanced
            train_data_epoch, train_labels_epoch = class_balance(self.data["train"], self.labels["train"])
            
            # train for one epoch
            model_multiple_gpu.fit_generator(datagen.flow(train_data_epoch, train_labels_epoch, batch_size=hp_value["batch_size"]),
                                             steps_per_epoch=int(len(train_data_epoch) / hp_value["batch_size"]), epochs=1)
            
            # predict for validation set
            val_pred = model_multiple_gpu.predict(self.data["validation"], batch_size=hp_value["batch_size"])
            val_auc = auc(self.labels["validation"], val_pred)
            
            # save history in csv
            df_history = pd.DataFrame(columns=["epoch", "validation_auc"])
            df_history.loc[ep] = [ep+1, val_auc]
            df_history.to_csv(path_to_model_save[:-3]+"_history.csv", index=False)

            if val_auc > val_auc_0:
                count_patience=0
                val_auc_0 = val_auc
#                test_pred = model_multiple_gpu.predict(self.data["test"], batch_size=self.batch_size)
#                test_auc = auc(self.labels["test"], test_pred)
#                print("test_auc = ", test_auc)
                
                model.save(path_to_model_save)
            else:
                count_patience+=1
                # early stopping
                if count_patience>hp_value["patience"]:
                    break
                
        return val_auc
        
    def random_search(self,
                      iteration_num,
                      ):
        # setting directory to save models
        now = datetime.datetime.now()
        path_to_model_save = base_dir+"../nih_data/models/random_search_mm%02ddd%02d" % (now.month, now.day) + "_%03d/" # % (count)
        if not os.path.exists(path_to_model_save):
            os.makedirs(path_to_model_save)
        for count in range(1000):
            if not os.path.exists(path_to_model_save % count):
                path_to_model_save = path_to_model_save % count
                os.makedirs(path_to_model_save)
                break
            
        for iteration in iteration_num:
            hp_value = chose_hyperparam()
            self.train(hp_value, path_to_model_save)
        
            

def main():
    hp_value = chose_hyperparam()
    nb_gpus=1
    if_single_pathology=False
    
    cnn = CNN(pathology="Effusion", input_shape=(128,128,3), if_single_pathology=if_single_pathology)
    cnn.train(hp_value,nb_gpus)

if __name__ == '__main__':
    main()        

    