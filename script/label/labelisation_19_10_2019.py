#creattion du script pour intersection des coordonnées
from shapely.geometry import Point, Polygon, LineString
import pandas as pd
import os
import numpy as np
from numpy import linalg
from shutil import copyfile

FICHIER_SORTIE = "Data_Out/FichierAll_intersection1.csv"
FOLDER_PATH = "DATA/"
SOUCE_DATA_PATH = "vol/vol1/database/"
nameId = 'v3'

file1 = open(FICHIER_SORTIE,"a") 
  
# \n is placed to indicate EOL (End of Line) 
file1.write("image\tWD\tOW\tWW\n") 
def readDataFile(fileName, separator=','):
    dataFile = pd.read_csv(fileName, sep=separator, encoding="utf-8")
    dataFile = dataFile.fillna("")
    return dataFile

# ouvrir le fichier des lines
lineDataFrame = readDataFile("data_cibles.csv")
lineCoordinate = lineDataFrame[['pointID','lat', 'Long','WT']]

polyDataFrame = readDataFile("Data_Out/coordonnees_vol1.csv", "\t")
polyCoordinate = polyDataFrame[['point2', 'point3', 'point4', 'point1', 'image']]

lineList = []

for j in range(len(polyCoordinate)):
    p0 = polyCoordinate.iloc[j, 0].split()
    p1 = polyCoordinate.iloc[j, 1].split()
    p2 = polyCoordinate.iloc[j, 2].split()
    p3 = polyCoordinate.iloc[j, 3].split()
    
    coords = [(float(p0[0]),float(p0[1])),  (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])),  (float(p3[0]),float(p3[1]))]
    poly = Polygon(coords)
    dictLabelImage={'WD':0,'OW':0,'WW':0}
    for i in range(len(lineCoordinate)) : 
        lineList.append(lineCoordinate.iloc[i, 1])
        lineList.append(lineCoordinate.iloc[i, 2])
        #print("Line", i)
        if len(lineList) == 4: 
            myLine = LineString([(lineList[0],lineList[1]), (lineList[2],lineList[3])])
            if myLine.intersects(poly) or poly.contains(myLine):
                classe = str(lineCoordinate.iloc[i, 3])
                dictLabelImage[classe]=1
            
            lineList = []
    if dictLabelImage['WD']==0 and dictLabelImage['OW']==0:
        file1.write(SOUCE_DATA_PATH+str(polyCoordinate.iloc[j,4])+"\t"+str(dictLabelImage['WD'])+"\t"+str(dictLabelImage['OW'])+"\t1\n")
    else:
        file1.write(SOUCE_DATA_PATH+str(polyCoordinate.iloc[j,4])+"\t"+str(dictLabelImage['WD'])+"\t"+str(dictLabelImage['OW'])+"\t"+str(dictLabelImage['WW'])+"\n")

file1.close()