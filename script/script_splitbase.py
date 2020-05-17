#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:03:44 2019

@author: inra-cirad
"""

import os
import shutil


dossier = "/home/inra-cirad/Bureau/apprentissage/gregory_folder/MyData/WW/"
lesImages = os.listdir(dossier)
partition = [80,20]

compteur=0
for taille in partition:
    debut = compteur
    compteur+= int((len(lesImages)*taille/100))
    os.mkdir(str(taille))
    for i in range(debut,compteur):
        shutil.copy(os.path.join(dossier,lesImages[i]),str(taille))
        
    




"""
import os
import gdal

in_path = '/home/inra-cirad/Bureau/MonDossier/aa/'
lesImages = os.listdir(in_path)
out_path = '/home/inra-cirad/Bureau/MonDossier/out/'
output_filename = 'tile_'

tile_size_x = 128
tile_size_y = 128

ds = gdal.Open(in_path + input_filename)
band = ds.GetRasterBand(1)
xsize = band.XSize
ysize = band.YSize

for i in range(0, xsize, tile_size_x):
    for j in range(0, ysize, tile_size_y):
        com_string = "gdal_translate -of GTIFF -srcwin " + str(i)+ ", " + str(j)+ ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + str(lesImages)+ " " + str(out_path) + str(i) + "_" + str(j) + ".tif"
        os.system(com_string)

"""