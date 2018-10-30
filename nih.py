# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 16:57:22 2018

@author: murata
"""

import numpy as np
import pandas as pd
import csv, shutil, random
from PIL import Image

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
    
def get_first_image(path_to_nih_data_csv = "../nih_data/Data_Entry_2017_murata.csv",
                    path_to_png_dir = "../nih_data/pngs/",
                    path_to_images = "../nih_data/images.npy",
                    if_save = False,
                    ):
    df = pd.read_csv(path_to_nih_data_csv)
    images = np.zeros((len(df),1024,1024))
    count = 0
    for image_index in df['Image Index'].values:
        images[count]  = np.asarray(Image.open(path_to_png_dir+image_index).convert('L'))
        count += 1
    
    if if_save:
        np.save(path_to_images, images)

    return images
    
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
    
grouping()
#gts = set_label()
#move_images()
#gts = set_gts(if_save=True)
#images = 
#print(gts.shape)