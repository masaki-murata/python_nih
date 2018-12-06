# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:36:03 2018

@author: murata
"""

from PIL import Image, ImageOps, ImageDraw
import pandas as pd
import numpy as np
import os, re, shutil

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

def move_cam_pngs(cam_method, layer_name, pathology,
                  path_to_bb_murata = "../nih_data/bb_images/%s/murata_select/", # % pathology
                  path_to_cam_pngs = "../nih_data/models/mm11dd26_size256/%s/cams/%s_%s/TP/", # % (pathology, cam_method, layer_name) 
#                  path_to_cam_moved = "../nih_data/models/mm11dd26_size256/%s/cams/%s_%s/murata_select/",
                  ):
    path_to_bb_murata=path_to_bb_murata % pathology
    path_to_cam_pngs = path_to_cam_pngs % (pathology, cam_method, layer_name) 
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
              path_to_cam_pngs="../nih_data/models/mm11dd26_size256/%s/cams/%s_%s/murata_select/",  # % (pathology, cam_method, layer_name) 
              path_to_cams = "../nih_data/models/mm11dd26_size256/%s/cams/%s_%s.png",  # % (pathology, cam_method, layer_name) 
              ):
    pngs, cam_pngs = [], []
    for png in os.listdir(path_to_cam_pngs % (pathology, cam_method, layer_name) ):
        if re.match(".*.png$", png): 
            if re.match(".*_cam.png$", png): 
                cam_pngs.append(png)
            else:
                pngs.append(png)
    pngs.sort(), cam_pngs.sort()
    assert len(pngs) == len(cam_pngs), print( len(pngs), len(cam_pngs) )
    row_num = int(np.sqrt(len(pngs)*2))
    column_num = int( len(pngs)*2.0 / row_num - 0.01) + 1
    canvas = Image.new("RGB", (column_num*size, row_num*size))
    print(row_num, column_num, len(pngs))
    r,c = 0,0
    for i in range(len(pngs)):
        print(r,c)
#        img = Image.open(path_to_cam_pngs % (pathology, cam_method, layer_name) +pngs[i]).convert('L').resize((128,128))
#        img_cam = Image.open(path_to_cam_pngs % (pathology, cam_method, layer_name) +cam_pngs[i]).convert('L').resize((128,128))
        img = Image.open(path_to_cam_pngs % (pathology, cam_method, layer_name) +pngs[i]).resize((size,size))
        img_cam = Image.open(path_to_cam_pngs % (pathology, cam_method, layer_name) +cam_pngs[i]).resize((size,size))
        canvas.paste(img, (c*size, r*size))
        canvas.paste(img_cam, ((c+1)*size, r*size))
        if c==column_num-1:
            r+=1
            c=0
        else:
            c+=2
    canvas.save(path_to_cams % (pathology, cam_method, layer_name) )
#        r += 1
#        c += 1
#        r = r % row_num
#        c = c % column_num_half


def main():
    layer_names = ["block4_conv4", "block5_conv4", "block5_pool"]
    cam_methods = ["grad_cam", "grad_cam+", "grad_cam+2", "grad_cam_murata"]
    pathology="Effusion"
    size = 1024
#    for layer_name, cam_method in zip(layer_names, cam_methods):
    for layer_name in layer_names:
        for cam_method in cam_methods:
            print(layer_name, cam_method)
            glue_cams(cam_method, layer_name, pathology, size)
#    pathologies = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
#                   'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']          
#                   
#    for pathology in pathologies:
#        count = make_bb_images(pathology=pathology)
#        print(pathology, count)

if __name__ == '__main__':
    main()
    