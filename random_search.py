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
    hp['network'] = ["VGG16", "VGG19", "DenseNet121", "DenseNet169", "DenseNet201", "InceptionV3", "ResNet50", "Xception"]
    
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


def random_search(pathology,
                  iteration,
                  ratio,
                  ):
    
    train(input_shape,#=(128,128,1),
              path_to_image_dir,#="",
              batch_size,#=32,
              val_num,#=128,
              epochs,#=100,
              ratio,#=[0.7,0.15,0.15],
              pathology,#="Effusion",
              patience,#=8,
              path_to_model_save,#="",
              eps,#=0.1,
              network="",
              if_transfer=True,
              if_rgb=False,
              if_batch_from_df=False,
              if_normalize=True,
              if_duplicate=True,
              if_augment=False,
              if_train=True,
              if_datagen_self=True,
              if_loss_ambiguous=False,
              if_single_pathology=True,
              nb_gpus=1,
              
              


