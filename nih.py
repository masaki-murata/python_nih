# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 16:57:22 2018

@author: murata
"""

import numpy as np
import pandas as pd
import os, csv, shutil, random
from PIL import Image
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.utils.training_utils import multi_gpu_model
from keras.models import Model
from keras.layers import Dense, Flatten, Input
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
                ):
    images = np.zeros((len(df),)+input_shape)
        
    count = 0
    for image_index in df['Image Index'].values:
        if if_rgb:
            for rgb in range(3):
                images[count,:,:,rgb]  = np.asarray(Image.open(path_to_image_dir+image_index).convert('L'))
        else:
            image = np.asarray( Image.open(path_to_image_dir+image_index).convert('L').resize(input_shape[-1:]) )
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
                            val_num=128,
                            ):
    df_shuffle = df.sample(frac=1)
    data = load_images(df_shuffle[:val_num])
    labels = load_gts(df_shuffle[:val_num])
    
    return data, labels

    
def batch_iter(df,
               batch_size=32,
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
                data = load_images(df_epoch)
                labels = load_gts(df_epoch)
                
                yield data, labels
    
    return data_generator(), steps_per_epoch

def make_model(input_shape=(128, 128, 1)):
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
            
def train(if_transfer=True,
          batch_size=32,
          val_num=128,
          nb_gpus=1,
          ):
    path_to_train_csv = "../nih_data/Data_Entry_2017_train.csv"
    path_to_validation_csv = "../nih_data/Data_Entry_2017_validation.csv"
    
    # set validation data
    print("---  start make_validation_dataset  ---")
    df_validation = pd.read_csv(path_to_validation_csv)
    val_data, val_label = make_validation_dataset(df_validation,
                                                  val_num=val_num
                                                  )
    
    # set generator for training data
    df_train = pd.read_csv(path_to_train_csv)
    train_gen , steps_per_epoch= batch_iter(df_train,
                                            batch_size=batch_size
                                            )
    
    # setting model
    print("---  start make_model  ---")
    model = make_model()
    if int(nb_gpus) > 1:
        model_multiple_gpu = multi_gpu_model(model, gpus=nb_gpus)
    else:
        model_multiple_gpu = model
#    else:
#        model = model

    opt_generator = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#    model_multi_gpu.compile(loss='binary_crossentropy', optimizer=opt_generator)
    model.compile(loss="binary_crossentropy", optimizer=opt_generator, metrics=["acc"])
    
    # start training
#    for epoch in range(1,epochs+1):
    model_multiple_gpu.fit_generator(train_gen,
                                     steps_per_epoch=steps_per_epoch,
                                     epochs=10,
                                     validation_data=(val_data,val_label),
                                     )
    
    
    
train(batch_size=4,
      val_num=1,
      nb_gpus=1,
      )
