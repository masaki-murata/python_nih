# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 16:57:22 2018

@author: murata
"""

import numpy as np
import pandas as pd
import csv
from PIL import Image

def set_label():
    path_to_nih_data_csv = "../nih_data/nih_data_000.csv"
    path_to_png_dir = "../nih_data/pngs/"
#    name_png = "%08d_000.png"
    num_pngs = len(pd.read_csv(path_to_nih_data_csv))
    nih_csv = open(path_to_nih_data_csv, 'r', encoding="utf-8")
    reader = csv.reader(nih_csv)
    header = next(reader)
    print("aho")
    no_findings = []
    for row in reader:
#        print(path_to_png_dir+row[0])
        img = np.asarray(Image.open(path_to_png_dir+row[0]).convert('L'))
        print(row[0], img.shape)
        if row[1] == "No Finding":
            print(row[1])
            no_findings.append(row[0])
    print(len(no_findings))
        

set_label()