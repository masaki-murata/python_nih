# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 16:28:22 2019

@author: murata
"""

import numpy as np

from data_process import make_dataset, class_balance


class CNN():
    def __init__(self, 
                 ratio=[0.7,0.1,0.2],
                 ):
        self.ratio = ratio
                
    def make_dataset(path_to_data_label,
                     ):
        data, labels = {}, {}
        for group in ["train", "validation", "test"]:
            data[group] = np.load(path_to_data_label % ("validation", "data"))
            labels[group] = np.load(path_to_data_label % ("validation", "labels"))
        data["validation"], labels["validation"] = class_balance(data["validation"], labels["validation"])
        
        



    