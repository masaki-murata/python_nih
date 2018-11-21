# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 16:04:04 2018

@author: murata
"""
from keras.models import load_model
import numpy as np
from PIL import Image
from keras import backend as K
import nih
import pandas as pd

def grad_cam(layer_name,
             ratio,
             input_shape=(128,128,1),
             pathology="Effusion",
             path_to_model="",
             ):
    path_to_csv_dir = "../nih_data/ratio_t%.2fv%.2ft%.2f/" % tuple(ratio) 
    path_to_group_csv = path_to_csv_dir+ "%s.csv" 
    df_test = pd.read_csv(path_to_group_csv % "test")
    test_data, test_label, df_test = nih.make_dataset(df_test,
                                         group="test",
                                         ratio=ratio,
                                         input_shape=input_shape,
                                         data_num=len(df_test),
                                         pathology=pathology,
                                         if_rgb=False,
                                         if_normalize=True,
                                         if_load_npy=True,
                                         if_save_npy=True,
                                         if_return_df=True,
                                         )

    path_to_save_cam = path_to_model[:-3]+"/cams/%s/%s" # % (TPFP, image_index)
    model = load_model(path_to_model)
    
    for count in range(len(test_label)):
        data = test_data[count]
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
        cam = np.dot(output, weights)
        
        cam = np.maximum(cam, 0) 
        cam = np.uint8(255*cam / cam.max())
        cam = Image.fromarray(cam)
        
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

#    return jetcam

    
    