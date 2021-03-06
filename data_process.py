# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:36:03 2018

@author: murata
"""

from PIL import Image, ImageOps, ImageDraw
import pandas as pd
import numpy as np
import os, re, shutil, math
from keras.utils import to_categorical

base_dir = os.getcwd()+"/"
if not re.search("nih_python", base_dir):
    base_dir = base_dir + "xray/nih_python/"

def grouping(path_to_nih_data_csv = base_dir+"../nih_data/Data_Entry_2017_murata.csv",
             path_to_bb = base_dir+"../nih_data/BBox_List_2017.csv",
             path_to_save_dir = "",
#             path_to_train_csv = "../nih_data/Data_Entry_2017_train.csv",
#             path_to_validation_csv = "../nih_data/Data_Entry_2017_validation.csv",
#             path_to_test_csv = "../nih_data/Data_Entry_2017_test.csv",
             if_duplicate=True,
             ratio = [0.7, 0.1, 0.2],
#             if_save = False,
             ):
#    path_to_save_dir = base_dir+"../nih_data/ratio_t%.2fv%.2ft%.2f/" % tuple(ratio) 
    if not os.path.exists(path_to_save_dir):
        os.makedirs(path_to_save_dir)
    path_to_group_csv = path_to_save_dir+ "%s.csv"
    
    df = pd.read_csv(path_to_nih_data_csv)
    total_num = len(df)
    train_num, validation_num = int(ratio[0]*len(df)), int(ratio[1]*len(df))
    
    df_bb = pd.read_csv(path_to_bb)
    bb_indices = df_bb["Image Index"].values
    bb_indices = list(set(list( map(lambda x: x[:-7]+"000.png", bb_indices) )))
#    test_num = len(df) - (train_num + validation_num)
    # BB のある患者は test に入れるので、df からそれらの患者を削除
    df_nonbb = df[~df["Image Index"].isin(bb_indices)]
    df_shuffle = df_nonbb.sample(frac=1)
    df_train, df_validation, df_test = df_shuffle[:train_num], df_shuffle[train_num:train_num+validation_num], df_shuffle[train_num+validation_num:]
    df_test = pd.concat([df_test, df[df["Image Index"].isin(bb_indices)]])
    assert total_num==(len(df_train)+len(df_validation)+len(df_test)), "{0},{1},{2},{3}".format(total_num, len(df_train), len(df_validation), len(df_test))
    
    if if_duplicate:
        # 重複を含んだリストを読み込む
        df_duplicate = pd.read_csv(base_dir+"../nih_data/Data_Entry_2017.csv")
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
    assert len(df_duplicate)==(len(df_train)+len(df_validation)+len(df_test)), "{0},{1},{2},{3}".format(len(df_duplicate), len(df_train), len(df_validation), len(df_test))
    # save to csv
    df_train.to_csv(path_to_group_csv % "train")
    df_validation.to_csv(path_to_group_csv % "validation")
    df_test.to_csv(path_to_group_csv % "test")


# 画像を読み込む関数
def load_images(df,
                input_shape=(128, 128, 1),
                path_to_image_dir = base_dir+"../nih_data/images/",
                if_rgb=False,
#                if_normalize=True,
                ):
    images = np.zeros((len(df),)+input_shape, dtype=np.uint8)
        
    count = 0
    for image_index in df['Image Index'].values:
        if if_rgb:
            for rgb in range(3):
                images[count,:,:,rgb]  = np.asarray(Image.open(path_to_image_dir+image_index).convert('L'))
        else:
            image = np.asarray( Image.open(path_to_image_dir+image_index).convert('L').resize(input_shape[:-1]) )
#            if if_normalize:
#                image = (image-image.mean()) / image.std()
            images[count] = image.reshape(input_shape)
        count += 1
#    images = images.reshape(images.shape+(1,))
    
    return images


# データセットを作成する関数
def make_dataset(df=[],
                 group="train",
                 path_to_image_dir="",
                 ratio=[0.7,0.1,0.2],
                 input_shape=(128, 128, 1),
                 data_num=128,
                 pathology="Effusion",
                 path_to_group_csv="",
                 if_rgb=False,
#                 if_normalize=True,
                 if_load_npy=False,
                 if_save_npy=False,
                 if_return_df=False,
                 if_load_df=False,
                 if_single_pathology=True,
                 ):
    size = input_shape[0]
    path_to_ratio = base_dir+"../nih_data/ratio_t%.2fv%.2ft%.2f/" % tuple(ratio)
    if not if_single_pathology:
        path_to_ratio = path_to_ratio[:-1] + "_multipathology/"
    path_to_data = path_to_ratio  + "%s_size%d_%s_data.npy" % (group, size, pathology)
    path_to_labels = path_to_ratio + "%s_size%d_%s_labels.npy" % (group, size, pathology)
#    path_to_group_csv = base_dir+"../nih_data/ratio_t%.2fv%.2ft%.2f/" % tuple(ratio) + "%s_%s.csv" % (group, pathology)
    path_to_group_csv = path_to_ratio+ "%s.csv" % (group)
    
    if len(df)==0:
        df = pd.read_csv(path_to_group_csv[:-4]+"_duplicate.csv")
    if data_num==-1:
        data_num=len(df)
    
    # csv をロード
    if if_load_df and os.path.exists(path_to_group_csv[:-4]+"_%s.csv" % pathology):
        df = pd.read_csv(path_to_group_csv[:-4]+"_%s.csv" % pathology)
#        elif os.path.exists(path_to_group_csv[:-4]+"_%s.csv" % pathology):
#            df = pd.read_csv(path_to_group_csv)
        
    if if_load_npy and os.path.exists(path_to_data):
        data = np.load(path_to_data)
        labels = np.load(path_to_labels)
#    df_deplicate = pd.read_csv()
    else:
        print("len(df), data_num =", len(df), data_num)
        print("df[Finding Labels], pathology = ", df["Finding Labels"], pathology)
        if if_single_pathology: # 該当する病変だけを含むかどうかで正負を判定する
            df = df[(df["Finding Labels"]=="No Finding") | (df["Finding Labels"]==pathology)]
#        df = df[(df["Finding Labels"]=="No Finding") | (df["Finding Labels"].str.contains(pathology))]
        data_num = min(data_num, len(df))
        print("len(df), data_num =", len(df), data_num)
#        df_shuffle = df.sample(frac=1)
        data = load_images(df[:data_num], path_to_image_dir=path_to_image_dir, input_shape=input_shape, if_rgb=if_rgb)#, if_normalize=if_normalize)
        labels = np.array(df["Finding Labels"].str.contains(pathology)*1.0)
        labels = to_categorical(labels[:data_num], num_classes=2)
    
    assert data.itemsize==1, print(data.dtype, data.itemsize)
    if if_save_npy and (not os.path.exists(path_to_data)):
        np.save(path_to_data, data)
        np.save(path_to_labels, labels)
    if not os.path.exists(path_to_group_csv[:-4]+"_%s.csv" % pathology):
        df[:data_num].to_csv(path_to_group_csv[:-4]+"_%s.csv" % pathology)
    
    if if_return_df:
        return data[:data_num], labels[:data_num], df[:data_num]
    else:
        return data[:data_num], labels[:data_num]


# 正常・異常が一対一になるように
def class_balance(data, labels):
    norm_indices = np.where(labels[:,1]==0)[0]
    sick_indices = np.where(labels[:,1]==1)[0]
    sick_num = len(sick_indices)
#    norm_num, sick_num = len(norm_indices), len(sick_indices)
    norm_indices = np.random.choice(norm_indices, sick_num, replace=False)
    indices = np.hstack((norm_indices, sick_indices))
    np.random.shuffle(indices)
        
    return data[indices], labels[indices]


def make_bb_images(path_to_bb = "../nih_data/BBox_List_2017.csv",
                   pathology="Effusion"):
    path_to_images = "../nih_data/images/"
    path_to_bb_images = "../nih_data/bb_images/" + pathology +"/"
    
    if not os.path.exists(path_to_bb_images):
        os.makedirs(path_to_bb_images)
    df = pd.read_csv(path_to_bb)
    df_pathology = df[df["Finding Label"]==pathology]    
    pathology_indices = df_pathology["Image Index"].values

    count=0
    for pathology_index in pathology_indices:
        df_index = df[df["Image Index"].isin([pathology_index])]
        multiplicity = len(df_index)
        if multiplicity==1:
#            pathology_unique_indices.append(pathology_index)
            image = Image.open(path_to_images+pathology_index).convert('L')
            image = ImageOps.colorize(image, black=[0,0,0], white=[255,255,255])
            draw = ImageDraw.Draw(image)
#            print(df_index.iloc[:,[2,3,4,5]].values[0])
            [x,y,w,h] = list(df_index.iloc[:,[2,3,4,5]].values[0])
            draw.rectangle([x, y, x+w, y+h], outline=(255,255,0))
            image.save(path_to_bb_images+pathology_index)
            count+=1
    return count


def move_cam_pngs_single(pathology, cam_dir,
                         path_to_bb_murata = base_dir + "../nih_data/bb_images/%s/murata_select/", # % pathology
                         path_to_cams=base_dir + "../nih_data/models/mm.../%s/cams/",
                         ):
    path_to_bb_murata=path_to_bb_murata % pathology
    path_to_cams=path_to_cams % pathology
    path_to_cam=path_to_cams+cam_dir
    path_to_cam_pngs = path_to_cam + "/TP/"
    path_to_cam_moved = path_to_cam + "/murata_select/"

    if not os.path.exists(path_to_cam_moved):
        os.makedirs(path_to_cam_moved)

    bb_pngs = []
    for png in os.listdir(path_to_bb_murata):
        if re.match(".*.png$", png): 
            bb_pngs.append(png)
    cam_pngs = []
    for png in os.listdir(path_to_cam_pngs):
        if re.match(".*.png$", png): 
            cam_pngs.append(png)

    for png in bb_pngs:
        if png in cam_pngs: 
            shutil.copyfile(path_to_cam_pngs+png, path_to_cam_moved+png[:-4]+"_cam.png")
            shutil.copyfile(path_to_bb_murata+png, path_to_cam_moved+png)


def move_cam_pngs_all(#cam_method, layer_name,
                      pathology,
                      path_to_bb_murata = base_dir + "../nih_data/bb_images/%s/murata_select/", # % pathology
                      path_to_cams=base_dir + "../nih_data/models/mm.../%s/cams/",
#                  path_to_cam_pngs = "../nih_data/models/mm11dd26_size%d_%s/%s/cams/%s_%s/TP/", # % (size, network, pathology, cam_method, layer_name) 
#                  path_to_cam_moved = "../nih_data/models/mm11dd26_size256/%s/cams/%s_%s/murata_select/",
                  ):
#    path_to_bb_murata=path_to_bb_murata % pathology
#    path_to_cams=path_to_cams % pathology
                
    for cam_dir in os.listdir(path_to_cams):
        if os.path.isdir(cam_dir):
#            path_to_cam_pngs = path_to_cam + "/TP/"
#            path_to_cam_moved = path_to_cam + "/murata_select/"
            move_cam_pngs_single(pathology, cam_dir, path_to_bb_murata, path_to_cams)
#        path_to_cam=path_to_cams+path_to_cam
#        if os.path.isdir(path_to_cam):
#            path_to_cam_pngs = path_to_cam + "/TP/"
#            path_to_cam_moved = path_to_cam + "/murata_select/"
#            move_cam_pngs_single(path_to_cam_pngs, path_to_cam_moved)
#    path_to_bb_murata=path_to_bb_murata % pathology
#    path_to_cam_pngs = (path_to_cams+"%s_%s/TP/") % (pathology, cam_method, layer_name) 
#    path_to_cam_moved = path_to_cam_pngs[:-3] + "murata_select/"


def glue_cams_single(pathology, size, 
                     path_to_cam_dir):
    path_to_cam_dir=path_to_cam_dir % pathology
    path_to_cam_pngs = path_to_cam_dir + "/murata_select/"
    path_to_glued_png = path_to_cam_dir+".png"
    pngs, cam_pngs = [], []
    for png in os.listdir(path_to_cam_pngs):
        if re.match(".*.png$", png): 
            if re.match(".*_cam.png$", png): 
                cam_pngs.append(png)
            else:
                pngs.append(png)
    pngs.sort(), cam_pngs.sort()
    assert len(pngs) == len(cam_pngs), print( len(pngs), len(cam_pngs) )
    row_num = math.ceil(np.sqrt(len(pngs)*2))
    half_column_num = math.ceil( len(pngs) / row_num )
    column_num = 2*half_column_num
    while row_num*half_column_num > len(pngs)+min(row_num, half_column_num):
        row_num -= 1
#        column_num = int( len(pngs)*2.0 / row_num - 0.01) + 2
#        row_num -= 1
    canvas = Image.new("RGB", (column_num*size, row_num*size))
    print(column_num, row_num, 2*len(pngs))
    r,c = 0,0
    for i in range(len(pngs)):
        print(c,r)
#        img = Image.open(path_to_cam_pngs % (pathology, cam_method, layer_name) +pngs[i]).convert('L').resize((128,128))
#        img_cam = Image.open(path_to_cam_pngs % (pathology, cam_method, layer_name) +cam_pngs[i]).convert('L').resize((128,128))
        img = Image.open(path_to_cam_pngs+pngs[i]).resize((size,size))
        img_cam = Image.open(path_to_cam_pngs+cam_pngs[i]).resize((size,size))
        canvas.paste(img, (c*size, r*size))
        canvas.paste(img_cam, ((c+1)*size, r*size))
        if c > column_num-3:
            r+=1
            c=0
        else:
            c+=2
    canvas.save(path_to_glued_png)

def glue_cams(#cam_method, layer_name, 
              pathology, size, 
              path_to_cams_dir="../nih_data/models/mm.../%s/cams/",
#              path_to_cam_pngs="../nih_data/models/mm11dd26_size%d_%s/%s/cams/%s_%s/murata_select/",  # % (pathology, cam_method, layer_name) 
#              path_to_cams = "../nih_data/models/mm11dd26_size%d_%s/%s/cams/%s_%s.png",  # % (pathology, cam_method, layer_name) 
              ):

    path_to_cams_dir=path_to_cams_dir % pathology
    for path_to_cam_dir in os.listdir(path_to_cams_dir):
        path_to_cam_dir=path_to_cams_dir+path_to_cam_dir
        if os.path.isdir(path_to_cam_dir):
#            path_to_cam_pngs = path_to_cam_dir + "/murata_select/"
#            path_to_glued_png = path_to_cam_dir+".png"
#            print(path_to_cam_pngs, path_to_glued_png)
            glue_cams_single(pathology, size, path_to_cam_dir)
#    path_to_cam_pngs=(path_to_cams+"%s_%s/murata_select/") % (pathology, cam_method, layer_name) 
#    path_to_cams_png = (path_to_cams+"%s_%s.png") % (pathology, cam_method, layer_name)


"""
def move_cam_pngs(cam_method, layer_name, pathology,
                  path_to_bb_murata = "../nih_data/bb_images/%s/murata_select/", # % pathology
                  path_to_cams="../nih_data/models/mm.../%s/cams/",
#                  path_to_cam_pngs = "../nih_data/models/mm11dd26_size%d_%s/%s/cams/%s_%s/TP/", # % (size, network, pathology, cam_method, layer_name) 
#                  path_to_cam_moved = "../nih_data/models/mm11dd26_size256/%s/cams/%s_%s/murata_select/",
                  ):
    path_to_bb_murata=path_to_bb_murata % pathology
    path_to_cam_pngs = (path_to_cams+"%s_%s/TP/") % (pathology, cam_method, layer_name) 
    path_to_cam_moved = path_to_cam_pngs[:-3] + "murata_select/"
    if not os.path.exists(path_to_cam_moved):
        os.makedirs(path_to_cam_moved)

    bb_pngs = []
    for png in os.listdir(path_to_bb_murata):
        if re.match(".*.png$", png): 
            bb_pngs.append(png)
    cam_pngs = []
    for png in os.listdir(path_to_cam_pngs):
        if re.match(".*.png$", png): 
            cam_pngs.append(png)

    for png in bb_pngs:
        if png in cam_pngs: 
            shutil.copyfile(path_to_cam_pngs+png, path_to_cam_moved+png[:-4]+"_cam.png")
            shutil.copyfile(path_to_bb_murata+png, path_to_cam_moved+png)
    
def glue_cams(cam_method, layer_name, pathology, size, 
              path_to_cams="../nih_data/models/mm.../%s/cams/",
#              path_to_cam_pngs="../nih_data/models/mm11dd26_size%d_%s/%s/cams/%s_%s/murata_select/",  # % (pathology, cam_method, layer_name) 
#              path_to_cams = "../nih_data/models/mm11dd26_size%d_%s/%s/cams/%s_%s.png",  # % (pathology, cam_method, layer_name) 
              ):
    path_to_cam_pngs=(path_to_cams+"%s_%s/murata_select/") % (pathology, cam_method, layer_name) 
    path_to_cams_png = (path_to_cams+"%s_%s.png") % (pathology, cam_method, layer_name)
    pngs, cam_pngs = [], []
    for png in os.listdir(path_to_cam_pngs):
        if re.match(".*.png$", png): 
            if re.match(".*_cam.png$", png): 
                cam_pngs.append(png)
            else:
                pngs.append(png)
    pngs.sort(), cam_pngs.sort()
    assert len(pngs) == len(cam_pngs), print( len(pngs), len(cam_pngs) )
    row_num = int(np.sqrt(len(pngs)*2))
    column_num = int( len(pngs)*2.0 / row_num - 0.01) + 2
    row_num -= 1
    canvas = Image.new("RGB", (column_num*size, row_num*size))
    print(column_num, row_num, 2*len(pngs))
    r,c = 0,0
    for i in range(len(pngs)):
        print(c,r)
#        img = Image.open(path_to_cam_pngs % (pathology, cam_method, layer_name) +pngs[i]).convert('L').resize((128,128))
#        img_cam = Image.open(path_to_cam_pngs % (pathology, cam_method, layer_name) +cam_pngs[i]).convert('L').resize((128,128))
        img = Image.open(path_to_cam_pngs+pngs[i]).resize((size,size))
        img_cam = Image.open(path_to_cam_pngs+cam_pngs[i]).resize((size,size))
        canvas.paste(img, (c*size, r*size))
        canvas.paste(img_cam, ((c+1)*size, r*size))
        if c==column_num-2:
            r+=1
            c=0
        else:
            c+=2
    canvas.save(path_to_cams_png)
#        r += 1
#        c += 1
#        r = r % row_num
#        c = c % column_num_half
"""

def main():
#    layer_names = ["block4_conv4", "block5_conv4", "block5_pool"]
#    cam_methods = ["grad_cam", "grad_cam_murata"]#, "grad_cam+", "grad_cam+2"]
    pathology="Effusion"
    size = 512
#    path_to_cam_pngs="../nih_data/models/mm12dd17_size512_VGG19/%s/cams/%s_%s/murata_select/"
#    path_to_cams = "../nih_data/models/mm12dd17_size512_VGG19/%s/cams/"
    

    path_to_cams = base_dir + "../nih_data/models/mm11dd26_size256/%s/cams/" # % pathology
    move_cam_pngs(pathology, path_to_cams=path_to_cams)
    glue_cams(pathology, size, path_to_cams)
#    move_cam_pngs(#cam_method, layer_name, 
#                  pathology,
#                  path_to_cams=path_to_cams,
##                          path_to_bb_murata = "../nih_data/bb_images/%s/murata_select/",
##                          path_to_cam_pngs = path_to_cam_pngs, # % (pathology, cam_method, layer_name),
#                          )
#    glue_cams(#cam_method, layer_name, 
#              pathology, size,
##                      path_to_cam_pngs=path_to_cam_pngs, # % (pathology, cam_method, layer_name), 
#              path_to_cams=path_to_cams, # % (pathology, cam_method, layer_name),
##              )
#    for group in ["train", "validation", "test"]:
#        make_dataset(df=[],
#                     group=group,
#                     path_to_image_dir=base_dir+"../nih_data/images/",
#                     ratio=[0.7,0.1,0.2],
#                     input_shape=(128, 128, 1),
#                     data_num=-1,
#                     pathology="Effusion",
#                     path_to_group_csv="",
#                     if_rgb=False,
#        #                 if_normalize=True,
#                     if_load_npy=False,
#                     if_save_npy=True,
#                     if_return_df=False,
#                     if_load_df=False,
#                     if_single_pathology=False,
#                     )
    
if __name__ == '__main__':
    main()
    