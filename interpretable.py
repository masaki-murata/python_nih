# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 16:04:04 2018

@author: murata
"""
from keras.models import load_model
import numpy as np
from PIL import Image
from keras import backend as K
import pandas as pd
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

def grad_cam(layer_name="block3_conv4",
             ratio=[0.7,0.1,0.2],
             input_shape=(128,128,1),
             pathology="Effusion",
             path_to_model="../nih_data/models/mm11dd26_size256/%s.h5",
             if_duplicate=True,
             ):
    path_to_model=path_to_model % pathology
    path_to_csv_dir = "../nih_data/ratio_t%.2fv%.2ft%.2f/" % tuple(ratio) 
    path_to_group_csv = path_to_csv_dir+ "%s.csv" 
    if if_duplicate:
        path_to_group_csv = path_to_group_csv[:-4]+"_duplicate.csv"
    df_test = pd.read_csv(path_to_group_csv % "test")[:50]
    test_data, test_label, df_test = nih.make_dataset(df_test,
                                         group="test",
                                         ratio=ratio,
                                         input_shape=input_shape,
                                         data_num=len(df_test),
                                         pathology=pathology,
                                         path_to_group_csv=path_to_group_csv,
                                         if_rgb=False,
                                         if_normalize=True,
                                         if_load_npy=False,
                                         if_save_npy=False,
                                         if_return_df=True,
                                         )
    print(test_data.shape, test_label.shape, len(df_test))
#    print(test_label)
#    path_to_save_cam = path_to_model[:-3]+"/cams/%s/%s" # % (TPFP, image_index)
    path_to_save_cam = path_to_model[:-3]+"/cams/%s/" # % (TPFP)
    if not os.path.exists(path_to_save_cam % "TP"):
        os.makedirs(path_to_save_cam % "TP")
    if not os.path.exists(path_to_save_cam % "FP"):
        os.makedirs(path_to_save_cam % "FP")
    path_to_save_cam = path_to_save_cam + "%s"
    model = load_model(path_to_model)
    model.summary()
    
    for count in range(len(test_label)):
        data = test_data[count]
        data = data.reshape((1,)+data.shape)
        predictions = model.predict(data)
        class_idx = np.argmax(predictions[0])
        if class_idx == 0:
            continue
        elif class_idx==1:
            if test_label[count, class_idx]==1:
                TPFP = "TP"
            elif test_label[count, class_idx]==0:
                TPFP = "FP"
        class_output = model.output[:, class_idx]
    
        conv_output = model.get_layer(layer_name).output  # layer_nameのレイヤーのアウトプット
        grads = K.gradients(class_output, conv_output)  # gradients(loss, variables) で、variablesのlossに関しての勾配を返す
#        print(grads)
        grads = grads[0]
        gradient_function = K.function([model.input], [conv_output, grads])  # model.inputを入力すると、conv_outputとgradsを出力する関数
        
        output, grads_val = gradient_function([data])
        print(output.shape)
        output, grads_val = output[0], grads_val[0]
    
        # 重みを平均化して、レイヤーのアウトプットに乗じる
        weights = np.mean(grads_val, axis=(0, 1)) # global average pooling
#        print("output.shape={0}, weights.shape={1}".format(output.shape, weights.shape))
        cam = np.sum(output*weights.reshape((1,1)+weights.shape), axis=2)
#        cam = np.dot(output, weights)
        
        cam = np.maximum(cam, 0) 
        cam = np.uint8(255*cam / cam.max())
        cam = Image.fromarray(cam).resize((512,512))
        
        cam.save(path_to_save_cam % (TPFP, df_test["Image Index"].values[count]))
        """
        # 画像化してヒートマップにして合成
    
        cam = cv2.resize(cam, (200, 200), cv2.INTER_LINEAR) # 画像サイズは200で処理したので
        cam = np.maximum(cam, 0) 
        cam = cam / cam.max()

    jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # モノクロ画像に疑似的に色をつける
    jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)  # 色をRGBに変換
    jetcam = (np.float32(jetcam) + x / 2)   # もとの画像に合成
    """
class CAM:
    def __init__(self, 
                 layer_name, 
                 path_to_model, 
                 pathology, 
                 input_shape,
                 batch_size,
                 if_load_npy,
                 if_save_npy,
                 ratio=[0.7,0.1,0.2],
                 if_duplicate=True,
                 ):
        self.layer_name=layer_name
        self.pathology=pathology
        self.path_to_model=path_to_model % (pathology)
        self.ratio=ratio
        self.if_duplicate=if_duplicate
        self.input_shape=input_shape
        self.batch_size=batch_size
        self.if_load_npy=if_load_npy
        self.if_save_npy=if_save_npy
    
    # テストデータをロードする関数
    def load_test(self):
#        path_to_model=self.path_to_model % self.pathology
        path_to_csv_dir = "../nih_data/ratio_t%.2fv%.2ft%.2f/" % tuple(self.ratio) 
        path_to_group_csv = path_to_csv_dir+ "%s.csv" 
        if self.if_duplicate:
            path_to_group_csv = path_to_group_csv[:-4]+"_duplicate.csv"
        df_test = pd.read_csv(path_to_group_csv % "test")#[:50]
        test_data, test_label, df_test = nih.make_dataset(df_test,
                                             group="test",
                                             ratio=self.ratio,
                                             input_shape=self.input_shape,
                                             data_num=len(df_test),
                                             pathology=self.pathology,
                                             path_to_group_csv=path_to_group_csv,
                                             if_rgb=False,
                                             if_normalize=True,
                                             if_load_npy=self.if_load_npy,
                                             if_save_npy=self.if_save_npy,
                                             if_return_df=True,
                                             )
        return test_data, test_label, df_test 
#    
#    def make_dir(self):
#        path_to_save_cam = self.path_to_model[:-3]+"/cams/%s/" # % (TPFP)
#        if not os.path.exists(path_to_save_cam % "TP"):
#            os.makedirs(path_to_save_cam % "TP")
#        if not os.path.exists(path_to_save_cam % "FP"):
#            os.makedirs(path_to_save_cam % "FP")
#        path_to_save_cam = path_to_save_cam + "%s"
    
    """ 将来的には nn の学習も入れたい """
    # nn の出力を出す
    def predict(self):
        self.test_data, self.test_label, self.df_test = self.load_test()
        self.model = load_model(self.path_to_model)
        self.model.summary()
        print(self.layer_name)
#        print("aho")
        self.predictions = self.model.predict(self.test_data, batch_size=self.batch_size)
        
        
#        return model, predictions
    def save_cam(self, method, cams, start_index):
        path_to_save_cam = self.path_to_model[:-3]+"/cams/" + method + "_" + self.layer_name+"/%s/" # % (TPFP)
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
            cam = Image.fromarray(cam).resize((512,512)).convert('L')
#            print(count)
            cam.save(path_to_save_cam % (TPFP, self.df_test["Image Index"].values[count]))
            count+=1
        
    
    def grad_cam(self):
        self.predict()
        mask_predictions = self.predictions[:,1] > 0.5
#        print(mask_predictions.shape)
        class_output = self.model.output[:, 1]
        conv_output = self.model.get_layer(self.layer_name).output  # layer_nameのレイヤーのアウトプット
        grads = K.gradients(class_output, conv_output)[0]  # gradients(loss, variables) で、variablesのlossに関しての勾配を返す
        gradient_function = K.function([self.model.input], [conv_output, grads])  # model.inputを入力すると、conv_outputとgradsを出力する関数
        start_index=0
        end_index=min(self.batch_size, len(mask_predictions))
        print("start_index = ", start_index)
        while start_index < end_index:
#            print(self.test_data.shape)
            output, grads_val = gradient_function([self.test_data[start_index:end_index]])
#            print("output.shape =", output.shape)
#             重みを平均化して、レイヤーのアウトプットに乗じる
#            weights = np.mean(grads_val, axis=(0, 1))
            weights = np.mean(grads_val, axis=(1, 2)) # global average pooling
#            print("weights.shape = ", weights.shape)
    #        print("output.shape={0}, weights.shape={1}".format(output.shape, weights.shape))
            cams = np.sum(output*weights.reshape((weights.shape[0],1,1,weights.shape[-1])), axis=3)
            self.save_cam(method="grad_cam", cams=cams, start_index=start_index)
#            print(cams.shape)
            start_index=start_index+self.batch_size
            end_index=min(start_index+self.batch_size, len(mask_predictions))
        print(end_index, len(mask_predictions))

            
    def grad_cam_murata(self):
        self.predict()
        mask_predictions = self.predictions[:,1] > 0.5
        class_output = self.model.output[:, 1]
        conv_output = self.model.get_layer(self.layer_name).output  # layer_nameのレイヤーのアウトプット
        grads = K.gradients(class_output, conv_output)[0]  # gradients(loss, variables) で、variablesのlossに関しての勾配を返す
        gradient_function = K.function([self.model.input], [conv_output, grads])  # model.inputを入力すると、conv_outputとgradsを出力する関数
        start_index=0
        end_index=min(self.batch_size, len(mask_predictions))
        while start_index < end_index:
#            print(self.test_data.shape)
            output, grads_val = gradient_function([self.test_data[start_index:end_index]])
#            print("output.shape =", output.shape)
            # 重みを平均化して、レイヤーのアウトプットに乗じる
    #        weights = np.mean(grads_val, axis=(0, 1))
#            weights = np.mean(grads_val, axis=(1, 2)) # global average pooling
#            print("weights.shape = ", grads_val.shape)
    #        print("output.shape={0}, weights.shape={1}".format(output.shape, weights.shape))
            grads_val = np.maximum(grads_val,0)
            cams = np.sum(output*grads_val, axis=3)
            self.save_cam(method="grad_cam_murata",cams=cams, start_index=start_index)
#            print(cams.shape)
            start_index=start_index+self.batch_size
            end_index=min(start_index+self.batch_size, len(mask_predictions))
        print(end_index, len(mask_predictions))

def main():
    
    interpretable = CAM(layer_name="block4_conv4",
                         ratio=[0.7,0.1,0.2],
                         input_shape=(256,256,1),
                         batch_size=32,
                         pathology="Effusion",
                         path_to_model="../nih_data/models/mm11dd26_size256/%s.h5",
                         if_load_npy=False,
                         if_save_npy=True,
                         )
    interpretable.grad_cam_murata()
#    grad_cam(input_shape=(256,256,1),layer_name="block4_conv4")

if __name__ == '__main__':
    main()
    