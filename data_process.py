# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:36:03 2018

@author: murata
"""

from PIL import Image, ImageOps, ImageDraw
import pandas as pd
import numpy as np

def make_bb_images(path_to_bb = "../nih_data/BBox_List_2017.csv",
                   pathology="Effusion"):
    path_to_images = "../nih_data/images/"
    path_to_bb_images = "../nih_data/bb_images/" + pathology +"/"
    df = pd.read_csv(path_to_bb)
    df_pathology = df[df["Finding Label"]==pathology]    
    pathology_indices = df_pathology["Image Index"].values

    count=0
    for pathology_index in pathology_indices:
        df_index = df[df["Image Index"].isin([pathology_index])]
        multiplicity = len(df_index)
        if multiplicity==1:
#            pathology_unique_indices.append(pathology_index)
            image = Image.open(path_to_images+pathology_index)
            image = ImageOps.colorize(image, black=[0,0,0], white=[255,255,255])
            draw = ImageDraw.Draw(image)
            [x,y,w,h] = df_index.iloc[:,[2,3,4,5]].values
            draw.rectangle([x, y, x+w, y+h], outline=(255,255,0))
            image.save(path_to_bb_images+pathology_index)
            count+=1
    return count

pathologies = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
               'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']          
               
for pathology in pathologies:
    count = make_bb_images(pathology=pathology)
    print(pathology, count)