# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 16:57:22 2018

@author: murata
"""

import numpy as np
import pandas as pd
import os, csv, shutil, random, sys, datetime
from keras.utils import to_categorical
from PIL import Image
from keras.optimizers import Adam, SGD
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
#from keras.utils.training_utils import multi_gpu_model
from keras.utils import multi_gpu_model
from keras.models import Model
from keras.layers import Dense, Flatten, Input, Conv2D, BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.models import Sequential
from keras import metrics


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

    

#def set_label(path_to_nih_data_csv = "../nih_data/nih_data_000.csv",
#              path_to_png_dir = "../nih_data/pngs/",
#              ):
##    path_to_nih_data_csv = "../nih_data/nih_data_000.csv"
##    path_to_png_dir = "../nih_data/pngs/"
##    name_png = "%08d_000.png"
#    num_pngs = len(pd.read_csv(path_to_nih_data_csv))
#    nih_csv = open(path_to_nih_data_csv, 'r', encoding="utf-8")
#    reader = csv.reader(nih_csv)
#    header = next(reader)
##    no_findings = []
#    gts = np.ones(num_pngs, dtype=np.int)
#    count = 0
#    for row in reader:
##        print(path_to_png_dir+row[0])
#        img = np.asarray(Image.open(path_to_png_dir+row[0]).convert('L'))
##        print(row[0], img.shape)
#        if row[1] == "No Finding":
##            print(row[1])
##            no_findings.append(row[0])
#            gts[count] = 0
#        count += 1
#    if args[if_save_gts]:
#        np.save(args[path_to_gts],gts)
##    print(len(no_findings))
#    return gts


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
                if_duplicate=True,
                ):
    if if_duplicate:
        path_to_nih_data_csv = "../nih_data/Data_Entry_2017_murata.csv"
    df = pd.read_csv(path_to_nih_data_csv)
    for image_index in df['Image Index'].values:
        shutil.copyfile(path_to_original_dir+image_index, path_to_moved_dir+image_index)
    
        
def grouping(path_to_nih_data_csv = "../nih_data/Data_Entry_2017_murata.csv",
             path_to_bb = "../nih_data/BBox_List_2017.csv",
#             path_to_save_dir = "../nih_data/ratio_t%.2fv%.2ft%.2f/",
#             path_to_train_csv = "../nih_data/Data_Entry_2017_train.csv",
#             path_to_validation_csv = "../nih_data/Data_Entry_2017_validation.csv",
#             path_to_test_csv = "../nih_data/Data_Entry_2017_test.csv",
             if_duplicate=True,
             ratio = [0.8, 0.1, 0.1],
#             if_save = False,
             ):
    path_to_save_dir = "../nih_data/ratio_t%.2fv%.2ft%.2f/" % tuple(ratio) 
    os.makedirs(path_to_save_dir)
    path_to_group_csv = path_to_save_dir+ "%s.csv"
    
    df = pd.read_csv(path_to_nih_data_csv)
    train_num, validation_num = int(ratio[0]*len(df)), int(ratio[1]*len(df))
#    test_num = len(df) - (train_num + validation_num)
    df_shuffle = df.sample(frac=1)
    df_train, df_validation, df_test = df_shuffle[:train_num], df_shuffle[train_num:train_num+validation_num], df_shuffle[train_num+validation_num:]
    
    if if_duplicate:
        # 重複を含んだリストを読み込む
        df_duplicate = pd.read_csv("../nih_data/Data_Entry_2017.csv")
        # 患者リストを作成
        train_ids = list(df_train["Patient ID"].values)
        validation_ids = list(df_validation["Patient ID"].values)
        test_ids = list(df_test["Patient ID"].values)
        # 重複を許して患者を取り出す
        df_train = df_duplicate[df_duplicate["Patient ID"].isin(train_ids)]
        df_validation = df_duplicate[df_duplicate["Patient ID"].isin(validation_ids)]
        df_test = df_duplicate[df_duplicate["Patient ID"].isin(test_ids)]
        # 保存先を変更
        path_to_group_csv = path_to_group_csv[:-4]+"_duplicate.csv"
        print(len(df_train))
#        path_to_train_csv = path_to_train_csv[:-4]+"_duplicate.csv"
#        path_to_validation_csv = path_to_validation_csv[:-4]+"_duplicate.csv"
#        path_to_test_csv = path_to_test_csv[:-4]+"_duplicate.csv"
    
    # save to csv
    df_train.to_csv(path_to_group_csv % "train")
    df_validation.to_csv(path_to_group_csv % "validation")
    df_test.to_csv(path_to_group_csv % "test")
#    df_train.to_csv(path_to_train_csv)
#    df_validation.to_csv(path_to_validation_csv)
#    df_test.to_csv(path_to_test_csv)
#    image_ids = list( df['Image Index'].values )
#    image_ids = random.sample(image_ids, len(image_ids))
#    train_ids, validation_ids, test_ids = image_ids[:train_num], image_ids[train_num:train_num+validation_num], image_ids[train_num+validation_num:]
    
#    return train_ids, validation_ids, test_ids

def make_dataset(df,
                 group="train",
                 ratio=[0.7,0.15,0.15],
                 input_shape=(128, 128, 1),
                 data_num=128,
                 pathology="Effusion",
                 path_to_group_csv="",
                 if_rgb=False,
                 if_normalize=True,
                 if_load_npy=False,
                 if_save_npy=False,
                 if_return_df=False,
                 ):
    size = input_shape[0]
    path_to_data = "../nih_data/ratio_t%.2fv%.2ft%.2f/" % tuple(ratio) + "%s_size%d_%s_data.npy" % (group, size, pathology)
    path_to_labels = "../nih_data/ratio_t%.2fv%.2ft%.2f/" % tuple(ratio) + "%s_size%d_%s_labels.npy" % (group, size, pathology)
    if if_load_npy and os.path.exists(path_to_data):
        data = np.load(path_to_data)
        labels = np.load(path_to_labels)
        df_shuffle = pd.read_csv(path_to_group_csv[:-4] % group + "_" + pathology + ".csv")
#    df_deplicate = pd.read_csv()
    else:
        df = df[(df["Finding Labels"]=="No Finding") | (df["Finding Labels"]==pathology)]
#        df = df[(df["Finding Labels"]=="No Finding") | (df["Finding Labels"].str.contains(pathology))]
        data_num = min(data_num, len(df))
        df_shuffle = df.sample(frac=1)
        data = load_images(df_shuffle[:data_num], input_shape=input_shape, if_rgb=if_rgb, if_normalize=if_normalize)
        labels = np.array(df_shuffle["Finding Labels"].str.contains(pathology)*1.0)
        labels = to_categorical(labels[:data_num])
    
    if if_save_npy and (not os.path.exists(path_to_data)):
        np.save(path_to_data, data)
        np.save(path_to_labels, labels)
        df_shuffle[:data_num].to_csv(path_to_group_csv[:-4] % group + "_" + pathology + ".csv")
    
    if if_return_df:
        return data, labels, df_shuffle[:data_num]
    else:
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
    transfer = VGG19(include_top=False, weights=None, input_tensor=x)
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
    top_model.add(Flatten(input_shape=transfer.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
#    top_model.add(Dropout(0.5))
    top_model.add(Dense(2, activation='softmax'))

    model = Model(input=input_img, output=top_model(transfer.output))

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
    print("pred_sorted[0] = ", pred_sorted[0])
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
        sensitivity = sensitivity_next+0
        specificity = specificity_next+0
    
    return auc
#        sensitivities[count] = tp_num / positive_num
#        specificities[count] = tn_num / positive_num
#        count += 1
        
#auc += 0.5*(fp_per_case[fp_id]-fp_per_case[fp_id+1])*(tpr[fp_id]+tpr[fp_id+1])    

#def weighted_crossentropy(y_true, y_pred):
#    y_pred[y_true]

def class_balance(data, labels):
    data_norm = data[labels[:,1]==0]
    data_sick = data[labels[:,1]==1]
    norm_num, sick_num = len(data_norm), len(data_sick)
    
    norm_indices = np.random.randint(norm_num, size=sick_num)
    data_norm_select = data_norm[norm_indices]
    data_epoch = np.vstack((data_norm_select, data_sick))
    labels_epoch = np.vstack(( np.tile(np.array([1,0]), (sick_num,1)), \
                                               np.tile(np.array([0,1]), (sick_num,1)) ))
    shuffle_indices = np.random.permutation(np.arange(len(labels_epoch)))
    data_epoch = data_epoch[shuffle_indices]
    labels_epoch = labels_epoch[shuffle_indices]
    
    return data_epoch, labels_epoch

def train(input_shape=(128,128,1),
          batch_size=32,
          val_num=128,
          epochs=100,
          ratio=[0.7,0.15,0.15],
          pathology="Effusion",
          patience=8,
          if_transfer=True,
          if_rgb=False,
          if_batch_from_df=False,
          if_normalize=True,
          if_duplicate=True,
          nb_gpus=1,
          ):
    print("train for ", pathology)
    path_to_csv_dir = "../nih_data/ratio_t%.2fv%.2ft%.2f/" % tuple(ratio) 
    now = datetime.datetime.now()
    path_to_model_save = "../nih_data/models/mm%02ddd%02d/" % (now.month, now.day)
    if not os.path.exists(path_to_model_save):
        os.makedirs(path_to_model_save)
    path_to_model_save = path_to_model_save+"%s.h5" % (pathology)
    if not os.path.exists(path_to_csv_dir):
        grouping(if_duplicate=if_duplicate, ratio=ratio)
    path_to_group_csv = path_to_csv_dir+ "%s.csv" 
#    path_to_train_csv = "../nih_data/Data_Entry_2017_train.csv"
#    path_to_validation_csv = "../nih_data/Data_Entry_2017_validation.csv"
    if if_duplicate:
        path_to_group_csv = path_to_group_csv[:-4]+"_duplicate.csv"
#        path_to_train_csv = path_to_train_csv[:-4]+"_duplicate.csv"
#        path_to_validation_csv = path_to_validation_csv[:-4]+"_duplicate.csv"
#        path_to_test_csv = path_to_test_csv[:-4]+"_duplicate.csv"
            
    # set validation data
    print("---  start make_validation_test_dataset  ---")
    df_validation = pd.read_csv(path_to_group_csv % "validation")
    val_data, val_label = make_dataset(df_validation,
                                       group="validation",
                                       ratio=ratio,
                                       input_shape=input_shape,
                                       data_num=len(df_validation),
                                       pathology=pathology,
                                       path_to_group_csv=path_to_group_csv,
                                       if_rgb=if_rgb,
                                       if_normalize=if_normalize,
                                       if_load_npy=True,
                                       if_save_npy=True,
                                       )
    val_data, val_label = class_balance(val_data, val_label)
    print(np.sum(val_label[:,1]==0), np.sum(val_label[:,1]==1))
    df_test = pd.read_csv(path_to_group_csv % "test")
    test_data, test_label = make_dataset(df_test,
                                         group="test",
                                         ratio=ratio,
                                         input_shape=input_shape,
                                         data_num=len(df_test),
                                         pathology=pathology,
                                         path_to_group_csv=path_to_group_csv,
                                         if_rgb=if_rgb,
                                         if_normalize=if_normalize,
                                         if_load_npy=True,
                                         if_save_npy=True,
                                         )
#    test_data, test_label = class_balance(test_data, test_label)
    print(np.sum(test_label[:,1]==0), np.sum(test_label[:,1]==1))
    
    # set generator for training data
    df_train = pd.read_csv(path_to_group_csv % "train")
    if if_batch_from_df:
        train_gen , steps_per_epoch= batch_iter_np(df_train,
                                                   batch_size=batch_size
                                                   )
    else:
        train_data, train_label = make_dataset(df_train,
                                               group="train",
                                               ratio=ratio,
                                               input_shape=input_shape,
                                               data_num=len(df_train),
                                               pathology=pathology,
                                               path_to_group_csv=path_to_group_csv,
                                               if_rgb=if_rgb,
                                               if_normalize=if_normalize,
                                               if_load_npy=True,
                                               if_save_npy=True,
                                               )
#        train_label = to_categorical(train_label)
    print(len(df_train), len(train_label))
    
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
#    class_weight = {0:np.sum(train_label[:,1])/float(len(train_label)), 1:np.sum(train_label[:,0])/float(len(train_label))}
#    print("class_weight = ", class_weight)
#    model_multi_gpu.compile(loss='binary_crossentropy', optimizer=opt_generator)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=[metrics.categorical_accuracy])
    
    # start training
    train_data_sick = train_data[train_label[:,1]==1]
    train_data_norm = train_data[train_label[:,1]==0]
    train_sick_num = len(train_data_sick)
    train_norm_num = len(train_data_norm)
    assert train_norm_num > train_sick_num
    val_auc_0, count_patience = 0, 0
    for epoch in range(1,epochs+1):
        print("epoch = ", epoch)
        if if_batch_from_df:
            model_multiple_gpu.fit_generator(train_gen,
                                             steps_per_epoch=steps_per_epoch,
                                             epochs=1,
                                             validation_data=(val_data,val_label),
                                             )
        else:
            train_data_epoch, train_labels_epoch = class_balance(train_data, train_label)
#            norm_indices = np.random.randint(train_norm_num, train_sick_num)
#            train_data_norm_select = train_data_norm[norm_indices]
#            train_data_epoch = np.vstack((train_data_norm_select, train_data_sick))
#            train_label_epoch = np.vstack(( np.tile(np.array([1,0]), (train_sick_num,1)), \
#                                               np.tile(np.array([0,1]), (train_sick_num,1)) ))
#            shuffle_indices = np.random.permutation(np.arange(len(train_label_epoch)))
#            train_data_epoch = train_data_epoch[shuffle_indices]
#            train_label_epoch = train_label_epoch[shuffle_indices]
            model_multiple_gpu.fit(train_data_epoch, train_labels_epoch,
    #                               steps_per_epoch=steps_per_epoch,
                                   epochs=1,
#                                   class_weight=class_weight,
                                   validation_data=(val_data,val_label),
                                   )
            val_pred = model_multiple_gpu.predict(val_data, batch_size=batch_size)
            print(val_pred.shape)
            val_auc = auc(val_label, val_pred)
            print("val_auc = ", val_auc)
            if val_auc > val_auc_0:
                count_patience=0
                val_auc_0 = val_auc
                test_pred = model_multiple_gpu.predict(test_data, batch_size=batch_size)
                test_auc = auc(test_label, test_pred)
                print("test_auc = ", test_auc)
                
                model.save(path_to_model_save)
            else:
                count_patience+=1
                if count_patience>patience:
                    sys.exit()
    return test_auc


def main():
    pathologies = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
                   'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']          
    
    test_aucs={}
    for pathology in pathologies:
        test_aucs[pathology] = train(batch_size=32,
                                     input_shape=(128,128,1),
                                     epochs=32,
                                     val_num=2048,
                                     ratio=[0.7,0.1,0.2],
                                     pathology=pathology,
                                     patience=4,
                                     if_batch_from_df=False,
                                     if_duplicate=True,
                                     if_normalize=True,
                                     nb_gpus=1,
                                     )
        print(test_aucs)

        
if __name__ == '__main__':
    main()        
