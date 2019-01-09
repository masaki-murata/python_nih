# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 13:34:20 2019

@author: murata
"""
import numpy as np
import math

# ハイパーパラメータの探索範囲を指定
def dict_hyperparam():
    hp = {}
    hp["learning_rate"] = list(range(1,7))
    hp["momentum"] = [0, 0.99]
    hp["optimizer"] = ["SGD", "Adam"]
    hp["batch_size"] = [2**x for x in range(3, 7)] #[2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**11] #[2**x for x in range(6)]
    hp["epoch_num"] = [32]
    
    hp['pool_stride'] = ['pool', 'stride']
#    hp["downsample_num"] = list( range(1,log_width) )
#    for x in range(1, log_width):
#        hp["fn%d" % x] = [2**y for y in range(3,7)] #list(range(8,65))
#        hp["conv_width%d" % x] = list(range(2, max(3,width//2**x)))
#        hp["conv_num%d" % x] = [1,2,3,4]
    
    hp["dense_layer_num"] = [1,2]
    hp["dense_units1"] = [2**x for x in range(1,9)]#list(range(2,256))
    hp["dense_units2"] = [2]
    
    return hp

        
# ハイパーパラメータをランダムに選択　or 乱数から選択
def chose_hyperparam(ransuu={}):
    hp = dict_hyperparam()
    hp_value = {}
    for hyperparam in hp.keys():
        if hyperparam in ransuu.keys():
            rp = ransuu[hyperparam]
        else:
            rp = np.random.rand()
        index = int( math.floor( rp*len(hp[hyperparam]) ) )
        hp_value[hyperparam] = hp[hyperparam][index]
#        if hyperparam == "conv_layer_num" and hp_value[hyperparam]==2:            
#                hp["dense_units1"] = list(range(2,(voi_width//2//2)))
            
    
    hp_value["learning_rate"] = 10**(-hp_value["learning_rate"] )
    
    if hp_value["dense_layer_num"] == 1:
        hp_value["dense_units1"] = 2
     
    
    return hp_value
