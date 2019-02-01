# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 16:04:04 2018

@author: murata
"""
from keras.models import load_model
import numpy as np
from PIL import Image
from keras import backend as K
from keras.utils import multi_gpu_model
from keras import initializers, layers
from keras.models import Model
import pandas as pd
import os, re, sys

# import original module
import nih
import data_process

base_dir = os.getcwd()+"/"
if not re.search("nih_python", base_dir):
    base_dir = base_dir + "xray/nih_python/"
print(base_dir)
   
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


class noise_layer(layers.Layer):
    def __init__(self, noiselevel, **kwargs):
        self.noiselevel = noiselevel
        super(noise_layer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        mean = K.mean(inputs)
        stddev = K.std(inputs)
        return inputs + K.random_normal(shape=K.shape(inputs),
                                        mean=mean,
                                        stddev=self.noiselevel*stddev)

class CAM:
    def __init__(self, 
                 layer_names, 
                 path_to_model, 
                 path_to_image_dir,
                 pathology, 
                 input_shape,
                 batch_size,
                 if_load_npy,
                 if_save_npy,
                 cam_methods,
                 samplesize,
                 noiselevel,
                 noiselayer_place, #  initial, intermediate
                 nb_gpus,
                 ratio=[0.7,0.1,0.2],
                 if_single_pathology=False,
                 if_duplicate=True,
                 if_murata_select=True,
                 ):
        if type(input_shape)==int:
            input_shape=(input_shape,input_shape,1)

        self.layer_names=layer_names
        self.pathology=pathology
        self.path_to_model=path_to_model % (pathology)
        self.path_to_image_dir=path_to_image_dir
        self.ratio=ratio
        self.if_duplicate=if_duplicate
        self.input_shape=input_shape
        self.batch_size=batch_size
        self.samplesize=samplesize
        self.noiselevel=noiselevel
        self.noiselayer_place=noiselayer_place
        self.if_load_npy=if_load_npy
        self.if_save_npy=if_save_npy
        self.cam_methods=cam_methods
        self.if_murata_select=if_murata_select
        self.if_single_pathology=if_single_pathology
        self.nb_gpus=nb_gpus
    
    # テストデータをロードする関数
    def load_test(self):
#        path_to_model=self.path_to_model % self.pathology
        path_to_csv_dir = base_dir+"../nih_data/ratio_t%.2fv%.2ft%.2f/" % tuple(self.ratio) 
        if not self.if_single_pathology:
            path_to_csv_dir = path_to_csv_dir[:-1] + "_multipathology/"
        path_to_group_csv = path_to_csv_dir + "%s.csv" 
        if self.if_duplicate:
            path_to_group_csv = path_to_group_csv[:-4]+"_duplicate.csv"
        self.df_test = pd.read_csv(path_to_group_csv % "test")#[:50]
        print("path_to_group_csv =" + path_to_group_csv % "test")
        print("len(self.df_test) = ", len(self.df_test))
        if self.if_murata_select:
            murata_select = os.listdir(base_dir+"../nih_data/bb_images/%s/murata_select/" % self.pathology)
            print(murata_select)
            self.df_test = self.df_test[self.df_test["Image Index"].isin(murata_select)]
            print(self.df_test.info())
            self.if_load_npy=False
            self.if_save_npy=False
        print("len(self.df_test) = ", len(self.df_test))
        print("input_shape = ",self.input_shape)
        self.test_data, self.test_label, self.df_test = data_process.make_dataset(df=self.df_test,
                                                                         group="test",
                                                                         ratio=self.ratio,
                                                                         path_to_image_dir=self.path_to_image_dir,
                                                                         input_shape=self.input_shape,
                                                                         data_num=len(self.df_test),
                                                                         pathology=self.pathology,
                                                                         path_to_group_csv=path_to_group_csv,
                                                                         if_rgb=False,
#                                                                         if_normalize=True,
                                                                         if_load_npy=self.if_load_npy,
                                                                         if_save_npy=self.if_save_npy,
                                                                         if_return_df=True,
                                                                         if_load_df=False,
                                                                         if_single_pathology=False,
                                                                         )
    
    """ 将来的には nn の学習も入れたい """
    # nn の出力を出す
    def predict(self):
        self.load_test()
        self.model = load_model(self.path_to_model)
        if int(self.nb_gpus) > 1:
            self.model_multiple_gpu = multi_gpu_model(self.model, gpus=self.nb_gpus)
        else:
            self.model_multiple_gpu = self.model
        self.model.summary()
#        print(self.layer_name)
#        print("aho")
        self.predictions = self.model_multiple_gpu.predict(self.test_data, batch_size=self.batch_size)
    
    def add_noise_layer(self, layer_name):
        previous_layer_name="maeno layer no namae"
        count = 0
        for i, layer in enumerate(self.model.layers):
            if i==0:
                input_layer = layer.input
                x = input_layer
            else:
                if previous_layer_name==layer_name:
                    count = 1
#                    layer.activation = activations.linear
                    x = noise_layer(self.noiselevel)(x)
                    x = layer(x)
#                    x = Activation("relu")(x)
                else:
                    x = layer(x)
            previous_layer_name = layer.name
        assert count==1, print("count = ", count)
    
        _model = Model(input_layer, x)
        if int(self.nb_gpus) > 1:
            self.model_multiple_gpu = multi_gpu_model(_model, gpus=self.nb_gpus)
        else:
            self.model_multiple_gpu = _model
        
        
#        return model, predictions
    def save_cam(self, layer_name, cam_method, cams, start_index):
        path_to_save_cam = self.path_to_model[:-3]+"/cams/"
        path_to_save_cam = path_to_save_cam + cam_method + "_" + layer_name
        if self.samplesize > 0:
            path_to_save_cam = path_to_save_cam + "_samplesize%d_noiselevel%.2f_noiselayer=%s" % (self.samplesize, self.noiselevel, self.noiselayer_place)
        path_to_save_cam = path_to_save_cam + "/%s/" # % (TPFP)
        if start_index==0:
            if not os.path.exists(path_to_save_cam % "TP"):
                os.makedirs(path_to_save_cam % "TP")
            if not os.path.exists(path_to_save_cam % "FP"):
                os.makedirs(path_to_save_cam % "FP")
        path_to_save_cam = path_to_save_cam + "%s"
        count=start_index
        for cam in cams:
            if self.test_label[count, 1]==1:
                TPFP = "TP"
            elif self.test_label[count, 1]==0:
                TPFP = "FP"
            cam = np.maximum(cam, 0) 
            if np.max(cam) > 0:
                cam = 255*cam / np.max(cam)
#            cam = np.uint8(255*cam / np.max(cam.max))
            cam.astype(np.uint8)
#            print(cam.shape)
            cam = Image.fromarray(cam).resize((1024,1024)).convert('L')
#            print(count)
            cam.save(path_to_save_cam % (TPFP, self.df_test["Image Index"].values[count]))
            count+=1
    
    
    def compute_cams(self, input_data, gradient_function, cam_method):
        output, grads_val = gradient_function([input_data])
#            print("output.shape =", output.shape)
#             重みを平均化して、レイヤーのアウトプットに乗じる
#            weights = np.mean(grads_val, axis=(0, 1))
        if cam_method=="grad_cam":
            weights = np.mean(grads_val, axis=(1, 2)) # global average pooling
            weights = weights.reshape((weights.shape[0],1,1,weights.shape[-1]))
        elif cam_method=="grad_cam+":
            weights = np.mean(grads_val, axis=(1, 2)) # global average pooling
            weights = np.maximum(weights, 0)
            weights = weights.reshape((weights.shape[0],1,1,weights.shape[-1]))
        elif cam_method=="grad_cam+2":
            weights = np.mean(np.maximum(grads_val,0), axis=(1, 2)) # global average pooling
            weights = weights.reshape((weights.shape[0],1,1,weights.shape[-1]))
        elif cam_method=="grad_cam_murata":
            weights = np.maximum(grads_val,0)
#            print("weights.shape = ", weights.shape)
        return np.sum(output*weights, axis=3)
        
    
    # 一組の（レイヤー、方法）に対して saliency map を評価
    def grad_cam_single(self, layer_name, cam_method):
#        self.predict()
        mask_predictions = self.predictions[:,1] > 0.5
#        print(mask_predictions.shape)
        print("model_multiple_gpu.output = ", self.model_multiple_gpu.output)
        class_output = self.model_multiple_gpu.output[:, 1]
        conv_output = self.model_multiple_gpu.get_layer(layer_name).output  # layer_nameのレイヤーのアウトプット
        grads = K.gradients(class_output, conv_output)[0]  # gradients(loss, variables) で、variablesのlossに関しての勾配を返す
        gradient_function = K.function([self.model_multiple_gpu.input], [conv_output, grads])  # model.inputを入力すると、conv_outputとgradsを出力する関数
        start_index=0
        end_index=min(self.batch_size, len(mask_predictions))
        print("start_index = ", start_index)
        while start_index < end_index:
            if self.samplesize==1:
                cams = self.compute_cams(self.test_data[start_index:end_index], gradient_function, cam_method)
            elif self.samplesize>1:
                cams = 0
                original_data = self.test_data[start_index:end_index]
                for sample in range(self.samplesize):
                    input_data = original_data + (self.noiselayer_place=="initial")*self.noiselevel*np.random.normal(size=original_data.shape)  
                    cams += self.compute_cams(input_data, gradient_function, cam_method)
                cams = cams / float(self.samplesize)
            self.save_cam(layer_name, cam_method, cams=cams, start_index=start_index)
            start_index=start_index+self.batch_size
            end_index=min(start_index+self.batch_size, len(mask_predictions))
        print(end_index, len(mask_predictions))
        
    def grad_cam(self):
        self.predict()
        print(self.predictions)
        print(self.test_data.shape)
        for layer_name in self.layer_names:
            for cam_method in self.cam_methods:
                if self.noiselayer_place == "intermediate":
                    self.add_noise_layer(layer_name)
                    self.model_multiple_gpu.summary()
                print( "layer_name = {0}, cam_method = {1}".format(layer_name,cam_method) )
                self.grad_cam_single(layer_name, cam_method)
       



def main():
    print("start interpretable")
    arg_nih={}
    arg_nih['pathologies']=[]#['Effusion']#['Edema', 'Effusion', 'Consolidation', 'Atelectasis', 'Hernia', 'Cardiomegaly', 'Infiltration', 'Fibrosis']
    arg_nih['layer_names']=[]#["block5_conv4"]#["block4_conv4", "block5_conv4", "block5_pool"]
    arg_nih['cam_methods']=["grad_cam_murata"]#["grad_cam+2", "grad_cam_murata"]
    arg_nih['path_to_model'] = "../nih_data/models/mm11dd26_size256/%s.h5"
    arg_nih['path_to_image_dir'] = "../nih_data/images/"
    arg_nih['ratio_train']=0.7
    arg_nih['ratio_validation']=0.1
    arg_nih['noiselayer_place']="initial"
    arg_nih['noiselevel']=0.1    
    arg_nih['input_shape']=256
    arg_nih['batch_size']=64
    arg_nih['samplesize']=100
    arg_nih['nb_gpus']=1
    arg_nih['if_duplicate']=True
    arg_nih['if_murata_select']=True
    arg_nih['if_single_pathology']=False
    arg_nih['if_load_npy']=True
    arg_nih['if_save_npy']=False


    int_args = ['batch_size', 'input_shape', 'nb_gpus', 'samplesize']
    float_args = ['ratio_train', 'ratio_validation', 'noiselevel']
    str_args = ["path_to_model", "path_to_image_dir", "noiselayer_place"]
    list_args = ['pathologies', 'layer_names', 'cam_methods']
    bool_args = ['if_duplicate', 'if_murata_select', 'if_single_pathology', 'if_load_npy', 'if_save_npy']
    total_args=int_args+str_args+bool_args+list_args+float_args

    argvs=sys.argv[1:]
    argc=len(argvs)
    if argc > 0:
        for arg_index in range(argc):
            comandline = argvs[arg_index]
            for arg_name in total_args:
                nih.read_comandline(arg_nih, 
                                str_args, int_args, bool_args, list_args, float_args,
#                                total_args,
                                comandline)

    arg_nih['ratio']=[arg_nih['ratio_train'], arg_nih['ratio_validation'], 1-arg_nih['ratio_train']-arg_nih['ratio_validation']]

#    layer_names = ["block4_conv4", "block5_conv4", "block5_pool"]
#    cam_methods = ["grad_cam+2"]
#    pathology="Effusion"
    
    print("arg_nih['noiselevel']  = ", arg_nih['noiselevel'] )
#    print("arg_nih['if_murata_select'] = ", arg_nih['if_murata_select'])
    for pathology in arg_nih['pathologies']:
        print(pathology)
        interpretable = CAM(layer_names=arg_nih['layer_names'],
                             ratio=arg_nih['ratio'],
                             input_shape=arg_nih['input_shape'],
                             batch_size=arg_nih['batch_size'],
                             samplesize=arg_nih['samplesize'],
                             noiselayer_place=arg_nih['noiselayer_place'],
                             noiselevel=arg_nih['noiselevel'],
                             pathology=pathology,
                             path_to_model=base_dir+arg_nih['path_to_model'],
                             path_to_image_dir=arg_nih['path_to_image_dir'],
                             if_duplicate=arg_nih['if_duplicate'],
                             if_single_pathology=arg_nih['if_single_pathology'],
                             if_load_npy=arg_nih['if_load_npy'],
                             if_save_npy=arg_nih['if_save_npy'],
                             cam_methods=arg_nih['cam_methods'],
                             if_murata_select=arg_nih['if_murata_select'],
                             nb_gpus=arg_nih['nb_gpus'],
                             )
        interpretable.grad_cam()
        path_to_cams=base_dir+arg_nih['path_to_model'][:-3]+"/cams/"
#        base_dir + "../nih_data/models/mm11dd26_size256/%s/cams/" # % pathology
        data_process.move_cam_pngs(pathology, path_to_cams=path_to_cams)
        data_process.glue_cams(pathology, 1024, path_to_cams)
#    grad_cam(input_shape=(256,256,1),layer_name="block4_conv4")

if __name__ == '__main__':
    main()
    