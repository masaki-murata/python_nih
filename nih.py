# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 16:57:22 2018

@author: murata
"""

import numpy as np
import pandas as pd
import os, csv, shutil, random
from keras.utils import to_categorical
from PIL import Image
from keras.optimizers import Adam, SGD
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.utils.training_utils import multi_gpu_model
from keras.models import Model
from keras.layers import Dense, Flatten, Input, Conv2D, BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.models import Sequential

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
if os.name=='posix':
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list="1", # specify GPU number
            allow_growth=True
        )
    )
    
    set_session(tf.Session(config=config))

    
"""
def set_label(path_to_nih_data_csv = "../nih_data/nih_data_000.csv",
              path_to_png_dir = "../nih_data/pngs/",
              ):
#    path_to_nih_data_csv = "../nih_data/nih_data_000.csv"
#    path_to_png_dir = "../nih_data/pngs/"
#    name_png = "%08d_000.png"
    num_pngs = len(pd.read_csv(path_to_nih_data_csv))
    nih_csv = open(path_to_nih_data_csv, 'r', encoding="utf-8")
    reader = csv.reader(nih_csv)
    header = next(reader)
#    no_findings = []
    gts = np.ones(num_pngs, dtype=np.int)
    count = 0
    for row in reader:
#        print(path_to_png_dir+row[0])
        img = np.asarray(Image.open(path_to_png_dir+row[0]).convert('L'))
#        print(row[0], img.shape)
        if row[1] == "No Finding":
#            print(row[1])
#            no_findings.append(row[0])
            gts[count] = 0
        count += 1
    if args[if_save_gts]:
        np.save(args[path_to_gts],gts)
#    print(len(no_findings))
    return gts
"""

# ground truth を作る関数
def set_gts(path_to_nih_data_csv = "../nih_data/Data_Entry_2017_murarta.csv",
            path_to_png_dir = "../nih_data/pngs/",
            path_to_gts = "../nih_data/gts.npy",
            if_save = False,
            ):
    
    df = pd.read_csv(path_to_nih_data_csv)
    gts = np.array(df['gt'].values, dtype=np.int)
    if if_save:
        np.save(path_to_gts, gts)
    
    return gts
    
#def load_images(path_to_nih_data_csv = "../nih_data/Data_Entry_2017_murata.csv",
#                path_to_png_dir = "../nih_data/images/",
#                path_to_images = "../nih_data/images.npy",
#                image_num=128,
#                if_save = False,
#                if_shuffle = False,
#                ):
#    df = pd.read_csv(path_to_nih_data_csv)
#    if if_shuffle:
#        df = df.sample(frac=1)
#    df = df[:image_num]
#    images = np.zeros((len(df),1024,1024))
#    count = 0
#    for image_index in df['Image Index'].values:
#        images[count]  = np.asarray(Image.open(path_to_png_dir+image_index).convert('L'))
#        count += 1
#    
#    if if_save:
#        np.save(path_to_images, images)
#
#    return images
    
def load_images(df,
                input_shape=(128, 128, 1),
                path_to_image_dir = "../nih_data/images/",
                if_rgb=False,
                if_normalize=True,
                ):
    images = np.zeros((len(df),)+input_shape)
        
    count = 0
    for image_index in df['Image Index'].values:
        if if_rgb:
            for rgb in range(3):
                images[count,:,:,rgb]  = np.asarray(Image.open(path_to_image_dir+image_index).convert('L'))
        else:
            image = np.asarray( Image.open(path_to_image_dir+image_index).convert('L').resize(input_shape[:-1]) )
            if if_normalize:
                image = (image-image.mean()) / image.std()
            images[count] = image.reshape(input_shape)
        count += 1
#    images = images.reshape(images.shape+(1,))
    
    return images

# ground truth を dataframe からロードする関する
def load_gts(df,
             ):
    gts = np.array(df['gt'].values, dtype=np.int)
    gts = gts.reshape(gts.shape+(1,))
    
    return gts    
    
# Follow up が 0 のデータを抽出    
def move_images(path_to_original_dir="/mnt/nas-public/nih-cxp-dataset/images/",
                path_to_moved_dir="../nih_data/images/",
                path_to_nih_data_csv = "../nih_data/Data_Entry_2017_murata.csv",
                ):
    df = pd.read_csv(path_to_nih_data_csv)
    for image_index in df['Image Index'].values:
        shutil.copyfile(path_to_original_dir+image_index, path_to_moved_dir+image_index)
    
        
def grouping(path_to_nih_data_csv = "../nih_data/Data_Entry_2017_murata.csv",
             path_to_train_csv = "../nih_data/Data_Entry_2017_train.csv",
             path_to_validation_csv = "../nih_data/Data_Entry_2017_validation.csv",
             path_to_test_csv = "../nih_data/Data_Entry_2017_test.csv",
             ratio = [0.7, 0.15, 0.15],
#             if_save = False,
             ):
    df = pd.read_csv(path_to_nih_data_csv)
    train_num, validation_num = int(ratio[0]*len(df)), int(ratio[1]*len(df))
#    test_num = len(df) - (train_num + validation_num)
    df_shuffle = df.sample(frac=1)
    df_train, df_validation, df_test = df_shuffle[:train_num], df_shuffle[train_num:train_num+validation_num], df_shuffle[train_num+validation_num:]
    # save to csv
    df_train.to_csv(path_to_train_csv)
    df_validation.to_csv(path_to_validation_csv)
    df_test.to_csv(path_to_test_csv)
#    image_ids = list( df['Image Index'].values )
#    image_ids = random.sample(image_ids, len(image_ids))
#    train_ids, validation_ids, test_ids = image_ids[:train_num], image_ids[train_num:train_num+validation_num], image_ids[train_num+validation_num:]
    
#    return train_ids, validation_ids, test_ids

def make_validation_dataset(df,
                            input_shape=(128, 128, 1),
                            val_num=128,
                            if_rgb=False,
                            if_normalize=True,
                            ):
    df_shuffle = df.sample(frac=1)
    data = load_images(df_shuffle[:val_num], input_shape=input_shape, if_rgb=if_rgb, if_normalize=if_normalize)
    labels = load_gts(df_shuffle[:val_num])
    return data, labels

    
def batch_iter_np(df,
                  input_shape=(128,128,1),
                  batch_size=32,
                  if_rgb=False,
                  if_normalize=True,
                  ):
    data_num = len(df)
    steps_per_epoch = int( (data_num - 1) / batch_size ) + 1
    def data_generator():
        while True:
            for batch_num in range(steps_per_epoch):
                if batch_num==0:
                    df_shuffle = df.sample(frac=1)
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_num)
                df_epoch = df_shuffle[start_index:end_index]
                data = load_images(df_epoch, input_shape=input_shape, if_rgb=if_rgb, if_normalize=if_normalize)
                labels = load_gts(df_epoch)
                
                yield data, labels
    
    return data_generator(), steps_per_epoch

def make_model(input_shape=(128, 128, 1)):
    input_img = Input(shape=input_shape)
    x = Conv2D(filters=3, kernel_size=3, padding="same", activation="relu")(input_img)
    vgg16 = VGG16(include_top=False, weights=None, input_tensor=x)
#    x = Conv2D(filters=8, kernel_size=3, padding="same", activation="relu")(input_img)
#    x = Conv2D(filters=8, kernel_size=3, strides=2, padding="valid", activation="relu")(x)
#    x = BatchNormalization()(x)
#    x = Conv2D(filters=32, kernel_size=3, padding="same", activation="relu")(x)
#    x = Conv2D(filters=32, kernel_size=3, strides=2, padding="valid", activation="relu")(x)
#    x = BatchNormalization()(x)
#    x = Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(x)
#    x = Conv2D(filters=64, kernel_size=3, strides=2, padding="valid", activation="relu")(x)
#    x = BatchNormalization()(x)
#    x = Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(x)
#    x = Conv2D(filters=128, kernel_size=3, strides=2, padding="valid", activation="relu")(x)
#    x = BatchNormalization()(x)

#    x = Flatten()(x)
#    x = Dense(256, activation="relu")(x)
#    output = Dense(2, activation="softmax")(x)
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
#    top_model.add(Dropout(0.5))
    top_model.add(Dense(2, activation='softmax'))

    model = Model(input=input_img, output=top_model(vgg16.output))

#    model = Model(input_img, output)
    opt_generator = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='binary_crossentropy', optimizer=opt_generator, metrics=['acc'])

    model.summary()
    
    return model
    

    # 転移学習用のモデルを作る関数
def make_model_transfer(input_shape=(128, 128, 1)):
    input_tensor = Input(shape=(1024, 1024, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
    for layer in vgg16.layers:
        layer.trainable = False
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
#    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    model = Model(input=vgg16.input, output=top_model(vgg16.output))
    
    model.summary()
    
    return model


def auc(y_true, y_pred):
    # y_true, y_pred は numpy 形式
    sick_true = y_true[:,1]
    sick_pred = y_pred[:,1]
    pred_sorted = np.sort(sick_pred)[::-1] # 大きい順から、つまり最小感度から
    positive_num = float( len(sick_true[sick_true==1]) )
    negative_num = float( len(sick_true[sick_true==0]) )
    data_num = len(sick_true)
    assert positive_num+negative_num == float( data_num )
#    sensitivities, specificities = np.zeros(data_num), np.zeros(data_num)
#    count = 0
    sensitivity, specificity = 0, 0
    auc = 0
    for threshold in pred_sorted:
        tp_num = len( sick_true[(sick_pred>=threshold)&(sick_true==1)] )
        fn_num = len( sick_true[(sick_pred>=threshold)&(sick_true==0)] )
        sensitivity_next = tp_num / positive_num
        specificity_next = fn_num / negative_num
        auc_part = 0.5*(sensitivity+sensitivity_next)*(specificity_next-specificity)
        assert auc_part >= 0
        auc += auc_part
#        sensitivities[count] = tp_num / positive_num
#        specificities[count] = tn_num / positive_num
#        count += 1
        
#auc += 0.5*(fp_per_case[fp_id]-fp_per_case[fp_id+1])*(tpr[fp_id]+tpr[fp_id+1])    
        
            
def train(input_shape=(128,128,1),
          batch_size=32,
          val_num=128,
          epochs=100,
          if_transfer=True,
          if_rgb=False,
          if_batch_from_df=False,
          if_normalize=True,
          nb_gpus=1,
          ):
    path_to_train_csv = "../nih_data/Data_Entry_2017_train.csv"
    path_to_validation_csv = "../nih_data/Data_Entry_2017_validation.csv"
    
    # set validation data
    print("---  start make_validation_dataset  ---")
    df_validation = pd.read_csv(path_to_validation_csv)
    val_data, val_label = make_validation_dataset(df_validation,
                                                  input_shape=input_shape,
                                                  val_num=val_num,
                                                  if_rgb=if_rgb,
                                                  if_normalize=if_normalize,
                                                  )
    print(np.sum(val_label==0), np.sum(val_label==1))
    val_label = to_categorical(val_label)
    
    # set generator for training data
    df_train = pd.read_csv(path_to_train_csv)
    if if_batch_from_df:
        train_gen , steps_per_epoch= batch_iter_np(df_train,
                                                   batch_size=batch_size
                                                   )
    else:
        train_data, train_label = make_validation_dataset(df_train,
                                                          input_shape=input_shape,
                                                          val_num=len(df_train),
                                                          if_rgb=if_rgb,
                                                          if_normalize=if_normalize,
                                                          )
        train_label = to_categorical(train_label)
    
    # setting model
    print("---  start make_model  ---")
    model = make_model(input_shape=input_shape)
    if int(nb_gpus) > 1:
        model_multiple_gpu = multi_gpu_model(model, gpus=nb_gpus)
    else:
        model_multiple_gpu = model
#    else:
#        model = model

    opt_generator = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#    model_multi_gpu.compile(loss='binary_crossentropy', optimizer=opt_generator)
    model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["acc"])
    
    # start training
    for epoch in range(1,epochs+1):
        if if_batch_from_df:
            model_multiple_gpu.fit_generator(train_gen,
                                             steps_per_epoch=steps_per_epoch,
                                             epochs=1,
                                             validation_data=(val_data,val_label),
                                             )
        else:
            model_multiple_gpu.fit(train_data, train_label,
    #                               steps_per_epoch=steps_per_epoch,
                                   epochs=1,
                                   validation_data=(val_data,val_label),
                                   )
            val_pred = model_multiple_gpu.predict(val_data, batch_size=batch_size)
            val_auc = auc(val_label, val_pred)
            print("val_auc = ", val_auc)
    
    
train(batch_size=32,
      input_shape=(256,256,1),
      epochs=100,
      val_num=2048,
      if_batch_from_df=False,
      if_normalize=True,
      nb_gpus=1,
      )
