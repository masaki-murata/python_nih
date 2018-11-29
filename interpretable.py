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
             path_to_model="../nih_data/models/mm11dd26_size128/%s.h5",
             if_duplicate=True,
             ):
    path_to_model=path_to_model % pathology
    path_to_csv_dir = "../nih_data/ratio_t%.2fv%.2ft%.2f/" % tuple(ratio) 
    path_to_group_csv = path_to_csv_dir+ "%s.csv" 
    if if_duplicate:
        path_to_group_csv = path_to_group_csv[:-4]+"_duplicate.csv"
    df_test = pd.read_csv(path_to_group_csv % "test")[:10]
    test_data, test_label, df_test = nih.make_dataset(df_test,
                                         group="test",
                                         ratio=ratio,
                                         input_shape=input_shape,
                                         data_num=len(df_test),
                                         pathology=pathology,
                                         path_to_group_csv=path_to_group_csv,
                                         if_rgb=False,
                                         if_normalize=True,
                                         if_load_npy=True,
                                         if_save_npy=True,
                                         if_return_df=True,
                                         )

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
        grads = K.gradients(class_output, conv_output)[0]  # gradients(loss, variables) で、variablesのlossに関しての勾配を返す
        gradient_function = K.function([model.input], [conv_output, grads])  # model.inputを入力すると、conv_outputとgradsを出力する関数
        
        output, grads_val = gradient_function([data])
        output, grads_val = output[0], grads_val[0]
    
        # 重みを平均化して、レイヤーのアウトプットに乗じる
        weights = np.mean(grads_val, axis=(0, 1)) # global average pooling
#        print("output.shape={0}, weights.shape={1}".format(output.shape, weights.shape))
        cam = np.sum(output*weights.reshape((1,1)+weights.shape), axis=2)
#        cam = np.dot(output, weights)
        
        cam = np.maximum(cam, 0) 
        cam = np.uint8(255*cam / cam.max())
        cam = Image.fromarray(cam).resize((512,512))
        
        cam.save(path_to_save_cam % (TPFP, df_test["Image Index"][count]))
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
                 ratio=[0.7,0.1,0.2],
                 if_duplicate=True,
                 ):
        self.layer_name=layer_name
        self.path_to_model=path_to_model
        self.pathology=pathology
        self.ratio=ratio
        self.if_duplicate=if_duplicate
        self.input_shape=input_shape
        self.batch_size=batch_size
    
    # テストデータをロードする関数
    def load_test(self):
#        path_to_model=self.path_to_model % self.pathology
        path_to_csv_dir = "../nih_data/ratio_t%.2fv%.2ft%.2f/" % tuple(self.ratio) 
        path_to_group_csv = path_to_csv_dir+ "%s.csv" 
        if self.if_duplicate:
            path_to_group_csv = path_to_group_csv[:-4]+"_duplicate.csv"
        df_test = pd.read_csv(path_to_group_csv % "test")
        test_data, test_label, df_test = nih.make_dataset(df_test,
                                             group="test",
                                             ratio=self.ratio,
                                             input_shape=self.input_shape,
                                             data_num=len(df_test),
                                             pathology=self.pathology,
                                             path_to_group_csv=path_to_group_csv,
                                             if_rgb=False,
                                             if_normalize=True,
                                             if_load_npy=True,
                                             if_save_npy=True,
                                             if_return_df=True,
                                             )
        return test_data, test_label, df_test 
    
    def make_dir(self):
        path_to_save_cam = self.path_to_model[:-3]+"/cams/%s/" # % (TPFP)
        if not os.path.exists(path_to_save_cam % "TP"):
            os.makedirs(path_to_save_cam % "TP")
        if not os.path.exists(path_to_save_cam % "FP"):
            os.makedirs(path_to_save_cam % "FP")
        path_to_save_cam = path_to_save_cam + "%s"
    
    # nn の出力を出す
    def predict(self):
        self.test_data, self.test_label, self.df_test = self.load_test(self)
        self.model = load_model(self.path_to_model)
        self.predictions = self.model.predict(self.test_data, batch_size=self.batch_size)
        
#        return model, predictions
    
    def grad_cam(self):
        self.predict(self)
        mask_predictions = self.predictions[:,1] > 0.5
        class_output = self.model.output[:, 1]
        conv_output = self.model.get_layer(self.layer_name).output  # layer_nameのレイヤーのアウトプット
        grads = K.gradients(class_output, conv_output)  # gradients(loss, variables) で、variablesのlossに関しての勾配を返す
        gradient_function = K.function([self.model.input], [conv_output, grads])  # model.inputを入力すると、conv_outputとgradsを出力する関数
        
        output, grads_val = gradient_function([self.test_data])
        # 重みを平均化して、レイヤーのアウトプットに乗じる
        weights = np.mean(grads_val, axis=(0, 1)) # global average pooling
#        print("output.shape={0}, weights.shape={1}".format(output.shape, weights.shape))
        cam = np.sum(output*weights.reshape((1,1)+weights.shape), axis=2)


def main():

    grad_cam()

if __name__ == '__main__':
    main()
    