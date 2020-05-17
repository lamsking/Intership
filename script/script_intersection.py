#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 17:24:49 2019

@author: inra-cirad
"""
#creattion du script pour intersection des coordonnées
from shapely.geometry import Point, Polygon, LineString
import pandas as pd
import os
import numpy as np
from numpy import linalg
from shutil import copyfile

FICHIER_SORTIE = "coordinate/Label_Final.csv"
FOLDER_PATH = "DATA/"
SOUCE_DATA_PATH = "/home/inra-cirad/Bureau/MonDossier/Dtest/"
nameId = ''

file1 = open(FICHIER_SORTIE,"w") 
  
# \n is placed to indicate EOL (End of Line) 
file1.write("image,classe \n") 
def readDataFile(fileName, separator=','):
    dataFile = pd.read_csv(fileName, sep=separator, encoding="utf-8")
    dataFile = dataFile.fillna("")
    return dataFile

# ouvrir le fichier des lines
lineDataFrame = readDataFile("data_cibles.csv")
lineCoordinate = lineDataFrame[['pointID','lat', 'Long']]

polyDataFrame = readDataFile("coordinate/coord_Final.csv", "\t")
polyCoordinate = polyDataFrame[['point2', 'point3', 'point4', 'point1', 'image']]

lineList = []
for i in range(len(lineCoordinate)) : 
    lineList.append(lineCoordinate.iloc[i, 1])
    lineList.append(lineCoordinate.iloc[i, 2])
    #print("Line", i)
    if len(lineList) == 4: 
        myLine = LineString([(lineList[0],lineList[1]), (lineList[2],lineList[3])])
        for j in range(len(polyCoordinate)):
            p0 = polyCoordinate.iloc[j, 0].split()
            p1 = polyCoordinate.iloc[j, 1].split()
            p2 = polyCoordinate.iloc[j, 2].split()
            p3 = polyCoordinate.iloc[j, 3].split()
        
            coords = [(float(p0[0]),float(p0[1])),  (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])),  (float(p3[0]),float(p3[1]))]
            poly = Polygon(coords)
            if myLine.intersects(poly):
                print("labelisation en cours ...")
                #print("image:",polyCoordinate.iloc[j, 4] ,"correspond à :", lineCoordinate.iloc[i, 0])
                #print("End")
                
                #print("######################")
                 
                s = str(lineCoordinate.iloc[i, 0])
                
                label = ''.join(c for c in s if c.isupper())
                imag = str(polyCoordinate.iloc[j, 4])
                nameImg = imag.split('.')
                print(s,label,imag)
                exit()
                if not os.path.isdir(FOLDER_PATH+label):
                    os.makedirs(FOLDER_PATH+label)
                

                if nameId not in(None, '') :
                    #nameImg = nameImg[0]+'_'+nameId+'.'+nameImg[1]
                    nameImg = nameImg[0]+'_'+nameId+'_'+label+'.'+nameImg[1]
                else:
                    nameImg = '.'.join(nameImg)
            
                file1.write(nameImg+","+s+"\n") 

                copyfile(SOUCE_DATA_PATH+imag, FOLDER_PATH+label+"/"+nameImg)
                
            else:
                pass
                #print("image pas dans intervalle")
        lineList = []
file1.close()
print("Finished!!!")

