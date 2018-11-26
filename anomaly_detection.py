# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 19:43:36 2018

@author: murata
"""

from keras import backend as K
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Dense, Flatten, Input, Conv2D, BatchNormalization
from keras.optimizers import Adam, SGD
import os

# import original module
import nih

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
if os.name=='posix':
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list="2", # specify GPU number
            allow_growth=True
        )
    )
    
    set_session(tf.Session(config=config))


def ad_loss(y_true, y_pred, eps):
    # y_true.shape=[batch_num, 1]
    return -K.sum( y_true*((1-eps)*K.log(y_pred)+eps*K.log(1-y_pred)) + (1-y_true)*((1-eps)*K.log(1-y_pred)+eps*K.log(y_pred)) )

def make_cnn_ad(input_shape=(64,64,1),
                ):
    input_img = Input(shape=input_shape)
    x = Conv2D(filters=4, kernel_size=3, padding="same", activation="relu")(input_img)
    x = Conv2D(filters=4, kernel_size=3, strides=2, padding="valid", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=16, kernel_size=3, padding="same", activation="relu")(x)
    x = Conv2D(filters=16, kernel_size=3, strides=2, padding="valid", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(x)
    x = Conv2D(filters=64, kernel_size=3, strides=2, padding="valid", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(x)
    x = Conv2D(filters=128, kernel_size=3, strides=2, padding="valid", activation="relu")(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)
    
    model = Model(input=input_img, output=output)
    
    return model

def anomaly_detection(path_to_csv="",
                      input_shape=(64,64,1),
                      eps=0.01,
                      batch_size=32,
                      ):
    # load data and labels
    df = pd.read_csv(path_to_csv)
    data = nih.load_images(df, input_shape=input_shape)
    assert len(df)==len(data)
    labels = np.ones((len(data),1))
    
    model = make_cnn_ad(input_shape=input_shape)
    model.summary()
    opt_generator = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='binary_crossentropy', optimizer=opt_generator, metrics=['acc'])
    
    model.fit(data, labels,
              batch_size=batch_size,
              epochs=32)

anomaly_detection(path_to_csv="../nih_data/Data_Entry_2017.csv")


    