# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 19:43:36 2018

@author: murata
"""

from keras import backend as K
def loss(y_true, y_pred, eps):
    # y_true.shape=[batch_num, 1]
    -K.sum( y_true*((1-eps)*K.log(y_pred)+eps*K.log(1-y_pred)) + (1-y_true)*((1-eps)*K.log(1-y_pred)+eps*K.log(y_pred)) )

def anomaly_detection(path_to_csv="",
                      ):
    # load data and labels
    df = pd.read_csv(path_to_csv)