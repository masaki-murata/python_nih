# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 16:57:22 2018

@author: murata
"""

import numpy as np
import pandas as pd
import os, csv, shutil, random, sys, datetime, re
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
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import DenseNet169
from keras.applications.densenet import DenseNet201
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.models import Sequential
from keras import metrics
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K


from data_process import make_dataset, class_balance, grouping, load_images


base_dir = os.getcwd()
if not re.search("nih_python", base_dir):
    base_dir = base_dir + "/xray/nih_python/"

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

    

# ground truth を作る関数
def set_gts(path_to_nih_data_csv = base_dir+"../nih_data/Data_Entry_2017_murarta.csv",
            path_to_png_dir = base_dir+"../nih_data/pngs/",
            path_to_gts = base_dir+"../nih_data/gts.npy",
            if_save = False,
            ):
    
    df = pd.read_csv(path_to_nih_data_csv)
    gts = np.array(df['gt'].values, dtype=np.int)
    if if_save:
        np.save(path_to_gts, gts)
    
    return gts
    
    
# ground truth を dataframe からロードする関する
def load_gts(df,
             ):
    gts = np.array(df['gt'].values, dtype=np.int)
    gts = gts.reshape(gts.shape+(1,))
    
    return gts    
    
# Follow up が 0 のデータを抽出    
def move_images(path_to_original_dir="/mnt/nas-public/nih-cxp-dataset/images/",
                path_to_moved_dir=base_dir+"../nih_data/images/",
                path_to_nih_data_csv = base_dir+"../nih_data/Data_Entry_2017_murata.csv",
                if_duplicate=True,
                ):
    if if_duplicate:
        path_to_nih_data_csv = base_dir+"../nih_data/Data_Entry_2017_murata.csv"
    df = pd.read_csv(path_to_nih_data_csv)
    for image_index in df['Image Index'].values:
        shutil.copyfile(path_to_original_dir+image_index, path_to_moved_dir+image_index)
    
        
#    df_train.to_csv(path_to_train_csv)
#    df_validation.to_csv(path_to_validation_csv)
#    df_test.to_csv(path_to_test_csv)
#    image_ids = list( df['Image Index'].values )
#    image_ids = random.sample(image_ids, len(image_ids))
#    train_ids, validation_ids, test_ids = image_ids[:train_num], image_ids[train_num:train_num+validation_num], image_ids[train_num+validation_num:]
    
#    return train_ids, validation_ids, test_ids


    
def batch_iter_df(df,
                  path_to_image_dir="",
                  input_shape=(128,128,1),
                  batch_size=32,
                  if_rgb=False,
#                  if_normalize=True,
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
                data = load_images(df_epoch, path_to_image_dir=path_to_image_dir, input_shape=input_shape, if_rgb=if_rgb)#, if_normalize=if_normalize)
                labels = load_gts(df_epoch)
                
                yield data, labels
    
    return data_generator(), steps_per_epoch

def batch_iter(data, labels, if_normalize, if_augment,
               batch_size=32,
               ):
    norm_indices = np.where(labels[:,1]==0)[0]
    sick_indices = np.where(labels[:,1]==1)[0]
    sick_num = len(sick_indices)
#    norm_num, sick_num = len(norm_indices), len(sick_indices)
    norm_indices = np.random.choice(norm_indices, sick_num, replace=False)
    indices = np.hstack((norm_indices, sick_indices))
    np.random.shuffle(indices)
    data_num = len(indices)
    steps_per_epoch = int( (data_num - 1) / batch_size ) + 1
    def data_generator():
        while True:
            for batch_num in range(steps_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_num)
                batch_indices = indices[start_index:end_index]
                batch_data = np.asarray(data[batch_indices], dtype=np.float32)
                batch_labels = labels[batch_indices]
                if if_normalize:
                    batch_data = (batch_data - np.mean(batch_data, axis=(1,2,3), keepdims=True) ) / np.std(batch_data, axis=(1,2,3), keepdims=True)
                    
#                df_epoch = df_shuffle[start_index:end_index]
#                data = load_images(df_epoch, path_to_image_dir=path_to_image_dir, input_shape=input_shape, if_rgb=if_rgb, if_normalize=if_normalize)
#                labels = load_gts(df_epoch)
                
                yield batch_data, batch_labels
    
    return data_generator(), steps_per_epoch


# ground_truth に間違いがあるときの誤差関数
def loss_ambiguous(y_true, y_pred, eps):
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    return -K.mean( y_true*((1-eps)*K.log(y_pred)+eps*K.log(1-y_pred)), -1)
#    y_pred=y_pred[:,1]
#    y_true=y_true[:,1]
#    return -K.sum( y_true*((1-eps)*K.log(y_pred)+eps*K.log(1-y_pred)) + (1-y_true)*((1-eps)*K.log(1-y_pred)+eps*K.log(y_pred)) )


def make_model(network, input_shape=(128, 128, 1)):
    input_img = Input(shape=input_shape)
    x = Conv2D(filters=3, kernel_size=3, padding="same", activation="relu")(input_img)
    if network=="VGG16":
        transfer = VGG16(include_top=False, weights=None, input_tensor=x)
    if network=="VGG19":
        transfer = VGG19(include_top=False, weights=None, input_tensor=x)
    if network=="DenseNet121":
        transfer = DenseNet121(include_top=False, weights=None, input_tensor=x)
    if network=="DenseNet169":
        transfer = DenseNet169(include_top=False, weights=None, input_tensor=x)
    if network=="DenseNet201":
        transfer = DenseNet201(include_top=False, weights=None, input_tensor=x)
    if network=="InceptionV3":
        transfer = InceptionV3(include_top=False, weights=None, input_tensor=x)
    if network=="ResNet50":
        transfer = ResNet50(include_top=False, weights=None, input_tensor=x)
    if network=="Xception":
        transfer = Xception(include_top=False, weights=None, input_tensor=x)

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
#def make_model_transfer(input_shape=(128, 128, 1)):
#    input_tensor = Input(shape=(1024, 1024, 3))
#    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
#    for layer in vgg16.layers:
#        layer.trainable = False
#    top_model = Sequential()
#    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
#    top_model.add(Dense(256, activation='relu'))
##    top_model.add(Dropout(0.5))
#    top_model.add(Dense(1, activation='sigmoid'))
#
#    model = Model(input=vgg16.input, output=top_model(vgg16.output))
#    
#    model.summary()
#    
#    return model


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
        fp_num = len( sick_true[(sick_pred>=threshold)&(sick_true==0)] )
        sensitivity_next = tp_num / float(positive_num)
        specificity_next = fp_num / float(negative_num)
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


def train(input_shape,#=(128,128,1),
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
          ):
    if type(input_shape)==int:
        input_shape=(input_shape,input_shape,1)
    print("train for ", pathology)
    path_to_csv_dir = base_dir+"../nih_data/ratio_t%.2fv%.2ft%.2f/" % tuple(ratio) 
    if not if_single_pathology:
        path_to_csv_dir = path_to_csv_dir[:-1] + "_multipathology/"
    now = datetime.datetime.now()
    if len(path_to_model_save)==0:
        path_to_model_save = base_dir+"../nih_data/models/mm%02ddd%02d_%s/" % (now.month, now.day, network)
    if not os.path.exists(path_to_model_save):
        os.makedirs(path_to_model_save)
    path_to_model_save = path_to_model_save+"%s.h5" % (pathology)
    if not os.path.exists(path_to_csv_dir+"train_duplicate.csv"):
        grouping(path_to_save_dir=path_to_csv_dir, if_duplicate=if_duplicate, ratio=ratio)
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
    df_validation = pd.read_csv(path_to_group_csv % ("validation") )
    val_data, val_label = make_dataset(df_validation,
                                       group="validation",
                                       path_to_image_dir=path_to_image_dir,
                                       ratio=ratio,
                                       input_shape=input_shape,
                                       data_num=len(df_validation),
                                       pathology=pathology,
                                       path_to_group_csv=path_to_group_csv,
                                       if_rgb=if_rgb,
#                                       if_normalize=if_normalize,
                                       if_load_npy=True,
                                       if_save_npy=True,
                                       if_return_df=False,
                                       if_load_df=True,
                                       if_single_pathology=if_single_pathology,
                                       )
    val_data, val_label = class_balance(val_data, val_label)
    print(np.sum(val_label[:,1]==0), np.sum(val_label[:,1]==1))
    df_test = pd.read_csv(path_to_group_csv % ("test") )
    test_data, test_label = make_dataset(df_test,
                                         group="test",
                                         path_to_image_dir=path_to_image_dir,
                                         ratio=ratio,
                                         input_shape=input_shape,
                                         data_num=len(df_test),
                                         pathology=pathology,
                                         path_to_group_csv=path_to_group_csv,
                                         if_rgb=if_rgb,
#                                         if_normalize=if_normalize,
                                         if_load_npy=True,
                                         if_save_npy=True,
                                         if_return_df=False,
                                         if_load_df=True,
                                         if_single_pathology=if_single_pathology,
                                         )
#    test_data, test_label = class_balance(test_data, test_label)
    print(np.sum(test_label[:,1]==0), np.sum(test_label[:,1]==1))
    
    # set generator or dataset for training
    df_train = pd.read_csv(path_to_group_csv % ("train") )
    if if_batch_from_df:
        train_gen , steps_per_epoch= batch_iter_df(df_train,
                                                   batch_size=batch_size
                                                   )        
    else:
        train_data, train_label = make_dataset(df_train,
                                               group="train",
                                               path_to_image_dir=path_to_image_dir,
                                               ratio=ratio,
                                               input_shape=input_shape,
                                               data_num=len(df_train),
                                               pathology=pathology,
                                               path_to_group_csv=path_to_group_csv,
                                               if_rgb=if_rgb,
#                                               if_normalize=if_normalize,
                                               if_load_npy=True,
                                               if_save_npy=True,
                                               if_return_df=False,
                                               if_load_df=True,
                                               if_single_pathology=if_single_pathology,
                                               )
        assert train_data.itemsize==1, print("train_data.itemsize = ", train_data.itemsize)
        if if_datagen_self:
            train_gen , steps_per_epoch= batch_iter(train_data, train_label, if_normalize, if_augment, batch_size=batch_size)

#        train_label = to_categorical(train_label)
    print(len(df_train), len(train_label))
    datagen = ImageDataGenerator(rotation_range=5,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 zoom_range=0.1,
                                 )    
    if not if_train:
        return 0
    # setting model
    print("---  start make_model  ---")
    model = make_model(network, input_shape=input_shape)
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
    if if_loss_ambiguous:
        def loss_amb(y_true, y_pred):
            return loss_ambiguous(y_true, y_pred, eps)
        loss=loss_amb
    else:
        loss="categorical_crossentropy"
       
  
    model_multiple_gpu.compile(loss=loss, optimizer=sgd, metrics=[metrics.categorical_accuracy])
    
    # start training
#    train_data_sick = train_data[train_label[:,1]==1]
#    train_data_norm = train_data[train_label[:,1]==0]
#    train_sick_num = len(train_data_sick)
#    train_norm_num = len(train_data_norm)
#    assert train_norm_num > train_sick_num
    val_auc_0, count_patience = 0, 0
    for epoch in range(1,epochs+1):
        print("epoch = ", epoch)
        if if_batch_from_df:
            model_multiple_gpu.fit_generator(train_gen,
                                             steps_per_epoch=steps_per_epoch,
                                             epochs=1,
                                             validation_data=(val_data,val_label),
                                             )
        elif if_datagen_self:
            model_multiple_gpu.fit_generator(train_gen,
                                             steps_per_epoch=steps_per_epoch,
                                             epochs=1,
                                             validation_data=(val_data,val_label),
                                             )
            
        else:
            print("train_data.shape = ", train_data.shape)            
            train_data_epoch, train_labels_epoch = class_balance(train_data, train_label)
            print("train_data_epoch.shape = ", train_data_epoch.shape)
#            norm_indices = np.random.randint(train_norm_num, train_sick_num)
#            train_data_norm_select = train_data_norm[norm_indices]
#            train_data_epoch = np.vstack((train_data_norm_select, train_data_sick))
#            train_label_epoch = np.vstack(( np.tile(np.array([1,0]), (train_sick_num,1)), \
#                                               np.tile(np.array([0,1]), (train_sick_num,1)) ))
#            shuffle_indices = np.random.permutation(np.arange(len(train_label_epoch)))
#            train_data_epoch = train_data_epoch[shuffle_indices]
#            train_label_epoch = train_label_epoch[shuffle_indices]
            if if_augment:
#                batches = 0
#                for x_batch, y_batch in datagen.flow(train_data_epoch, train_labels_epoch, batch_size=batch_size):
#                    model_multiple_gpu.fit(x_batch, y_batch)
#                    batches += 1
#                    if batches >= len(train_data) / 32:
#                        # we need to break the loop by hand because
#                        # the generator loops indefinitely
#                        break                
                model_multiple_gpu.fit_generator(datagen.flow(train_data_epoch, train_labels_epoch, batch_size=batch_size),
                                                 steps_per_epoch=int(len(train_data_epoch) / batch_size), epochs=1)
            else:
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
                break
    return test_auc


def train_pathologies(pathologies,#=[],
                      path_to_image_dir,#="",
                      input_shape,#=(128,128,1),
                      batch_size,#=32,
                      val_num,#=128,
                      epochs,#=100,
                      ratio,#=[0.7,0.15,0.15],
                      patience,#=8,
                      eps,
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
                      ):
    if type(input_shape)==int:
        input_shape=(input_shape,input_shape,1)
    now = datetime.datetime.now()
    path_to_model_save = base_dir+"../nih_data/models/mm%02ddd%02d_size%d_%s/" % (now.month, now.day, input_shape[0], network)
    if not if_single_pathology:
        path_to_model_save = path_to_model_save[:-1] + "_multipathology/"
#    print(os.getcwd())
#    os.mkdir("../nih_data/models")
    if not os.path.exists(path_to_model_save):
        os.makedirs(path_to_model_save)
    shutil.copyfile(base_dir+"./nih.py", path_to_model_save+"nih.py")
#    shutil.copyfile("./data_process.py", path_to_model_save+"data_process.py")
    
    
    df = pd.DataFrame(columns=["pathology", "test_auc"])
    count=0
    for pathology in pathologies:
        test_auc = train(batch_size=batch_size,
                         network=network,
                         path_to_image_dir=path_to_image_dir,
                         input_shape=input_shape,
                         epochs=epochs,
                         val_num=val_num,
                         ratio=ratio,
                         pathology=pathology,
                         patience=patience,
                         eps=eps,
                         path_to_model_save=path_to_model_save,
                         if_batch_from_df=if_batch_from_df,
                         if_duplicate=if_duplicate,
                         if_normalize=if_normalize,
                         if_augment=if_augment,
                         if_train=if_train,
                         if_datagen_self=if_datagen_self,
                         if_loss_ambiguous=if_loss_ambiguous,
                         if_single_pathology=if_single_pathology,
                         nb_gpus=nb_gpus,
                         )
        df.loc[count] = [pathology, test_auc]
        count+=1
        df.to_csv(path_to_model_save+"test_aucs.csv", index=False)
#    print(df)
    
    return df


def read_comandline(arg_dict, 
                    str_args, int_args, bool_args, list_args, float_args,
                    total_args, comandline):
    count=0
    for arg_name in total_args:
        if re.match('^'+arg_name, comandline):
            count += 1
    #        value = re.search('(?<=^'+arg_name+'=)\S+', comandline).group(0)
            if arg_name in str_args:
                value = re.search('(?<=^'+arg_name+'=)\S+', comandline).group(0)
                arg_dict[arg_name] = str(value)
            elif arg_name in int_args:
                value = re.search('(?<=^'+arg_name+'=)\d+', comandline).group(0)
                arg_dict[arg_name] = int(value)
            elif arg_name in float_args:
                value = re.search('(?<=^'+arg_name+'=)\S+', comandline).group(0)
                arg_dict[arg_name] = float(value)
            elif arg_name in bool_args:
                value = re.search('(?<=^'+arg_name+'=)\w+', comandline).group(0)
                if value == "True":
    #                    arg_dict['if_list'][arg_name] = True
                    arg_dict[arg_name] = True
                else:
    #                    arg_dict['if_list'][arg_name] = False
                    arg_dict[arg_name] = False
            elif arg_name in list_args:
                value = re.search('(?<=^'+arg_name+'=)\w+', comandline).group(0)
                arg_dict[arg_name].append(value)
                arg_dict[arg_name]=list(set(arg_dict[arg_name]))
    assert count==1, "count = {0}".format(count)
#            elif arg_name in func_args:
#                value = re.search('(?<=^'+arg_name+'=)\w+', comandline).group(0)
#                if value == "True":
#                    arg_dict[arg_name] = True
#                else:
#                    arg_dict[arg_name] = False

def main():
    
#    pathologies = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
#                   'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']          
    arg_nih={}
    arg_nih['pathologies']=[]#['Edema', 'Effusion', 'Consolidation', 'Atelectasis', 'Hernia', 'Cardiomegaly', 'Infiltration', 'Fibrosis']
    arg_nih['network']="DenseNet121"
    arg_nih['path_to_image_dir']=base_dir+"../nih_data/images/"
    arg_nih['batch_size']=64
    arg_nih['epochs']=128    
    arg_nih['val_num']=2048
    arg_nih['patience']=16
    arg_nih['nb_gpus']=1
    arg_nih['input_shape']=256
    arg_nih['ratio_train']=0.7
    arg_nih['ratio_validation']=0.1
    arg_nih['eps']=0.1
    arg_nih['if_batch_from_df']=False
    arg_nih['if_duplicate']=True
    arg_nih['if_normalize']=True
    arg_nih['if_augment']=True
    arg_nih['if_train']=True
    arg_nih['if_datagen_self']=True
    arg_nih['if_loss_ambiguous']=True
    arg_nih['if_single_pathology']=True

    int_args = ['batch_size', 'epochs', 'val_num', 'patience', 'nb_gpus', 'input_shape']
    float_args = ['ratio_train', 'ratio_validation', 'eps']
    str_args = ['network', "path_to_image_dir"]
    list_args = ['pathologies']
    bool_args = ['if_batch_from_df', 'if_duplicate', 'if_normalize', 'if_augment', 'if_train', 'if_datagen_self', 'if_loss_ambiguous', 'if_single_pathology']
    total_args=int_args+str_args+bool_args+list_args+float_args

#    pathologies = ['Edema', 'Effusion', 'Consolidation', 'Atelectasis', 'Hernia', 'Cardiomegaly', 'Infiltration', 'Fibrosis']
#    network="VGG19"
#    path_to_image_dir="/lustre/jh170036h/share/chestxray_nihcc/images"
#    batch_size=64
#    epochs=128
#    val_num=2048
#    ratio=[0.7,0.1,0.2]
#    patience=16
#    if_batch_from_df=False
#    if_duplicate=True
#    if_normalize=True
#    if_augment=True
#    nb_gpus=1
    
    
    argvs=sys.argv[1:]
    argc=len(argvs)
    if argc > 0:
        for arg_index in range(argc):
            arg_input = argvs[arg_index]
#            for arg_name in total_args:
            read_comandline(arg_nih, 
                            str_args, int_args, bool_args, list_args, float_args,
                            total_args, arg_input)
    arg_nih['ratio']=[arg_nih['ratio_train'], arg_nih['ratio_validation'], 1-arg_nih['ratio_train']-arg_nih['ratio_validation']]
    
    
#    for input_shape in [256]:
    print(arg_nih)
#    sys.exit()
    train_pathologies(pathologies=arg_nih['pathologies'],
                      network=arg_nih['network'],
                      path_to_image_dir=arg_nih['path_to_image_dir'],
                      batch_size=arg_nih['batch_size'],
                      input_shape=arg_nih['input_shape'],
                      epochs=arg_nih['epochs'],
                      val_num=arg_nih['val_num'],
                      ratio=arg_nih['ratio'],
                      patience=arg_nih['patience'],
                      eps=arg_nih['eps'],
                      if_batch_from_df=arg_nih['if_batch_from_df'],
                      if_duplicate=arg_nih['if_duplicate'],
                      if_normalize=arg_nih['if_normalize'],
                      if_augment=arg_nih['if_augment'],
                      if_train=arg_nih['if_train'],
                      if_datagen_self=arg_nih['if_datagen_self'],
                      if_loss_ambiguous=arg_nih['if_loss_ambiguous'],
                      if_single_pathology=arg_nih['if_single_pathology'],
                      nb_gpus=arg_nih['nb_gpus'],
                      )
#    test_aucs={}
#    for pathology in pathologies:
#        test_aucs[pathology] = train(batch_size=32,
#                                     input_shape=(256,256,1),
#                                     epochs=32,
#                                     val_num=2048,
#                                     ratio=[0.7,0.1,0.2],
#                                     pathology=pathology,
#                                     patience=8,
#                                     if_batch_from_df=False,
#                                     if_duplicate=True,
#                                     if_normalize=True,
#                                     nb_gpus=1,
#                                     )
#        print(test_aucs)

        
if __name__ == '__main__':
    main()        
