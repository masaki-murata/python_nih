# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 16:57:22 2018

@author: murata
"""

import numpy as np
import pandas as pd
import csv
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
        
        
def test(aho="aho", **args):
    print(args)
    if "aho" in args:
        print("aho is here")
    else:
        print("no aho")
#gts = set_label()
gts = set_gts()
print(gts.shape)