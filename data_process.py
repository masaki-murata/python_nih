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
#                  path_to_cam_moved = "./nih_data\models\mm11dd26_size256/%s/cams/%s_%s/murata_select/",
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
    

def main():
    layer_names = ["block4_conv4", "block5_conv4", "block5_pool"]
    cam_methods = ["grad_cam+"]
    pathology="Effusion"
#    for layer_name, cam_method in zip(layer_names, cam_methods):
    for layer_name in layer_names:
        for cam_method in cam_methods:
            move_cam_pngs(cam_method, layer_name, pathology)
#    pathologies = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
#                   'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']          
#                   
#    for pathology in pathologies:
#        count = make_bb_images(pathology=pathology)
#        print(pathology, count)

if __name__ == '__main__':
    main()
    