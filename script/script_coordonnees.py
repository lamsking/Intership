# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 11:00:01 2019

@author: INRA
"""
'''
Script d'extraction des metadonnées de chaque image de la base de données et les
stock dans un fichier soit format .txt oubien .csv
'''
import tifffile
import os
from geo.sphere import distance, destination, bearing
from math import *
exf_tag = {}
gps_tag = {}
home = "/home/inra-cirad/Bureau/MonDossier/"
enrg = "/home/inra-cirad/Bureau/MonDossier/sortie_image3/output"
#dataset = "/home/inra-cirad/Bureau/MonDossier/out_vol1"
dataset = "/home/inra-cirad/Bureau/MonDossier/datav3/"
#angles = [45, 135, 225, 315]
filn = open(os.path.join(os.path.join(home, enrg), "coord_file3.csv"),"w")
filn.write("image"+"\t"+"latitude"+"\t"+"longititude"+"\t"+"heightFootprint"+"\t"+"widthFootprint"+"\t"+"bearing"+"\t"+"point2"+"\t"+"point3"+"\t"+"point4"+"\t"+"point1")
filn.write("\n")
datasetFolder = os.path.join(home,dataset)
#xSensor = 4.8
#sySensor = 3.6
fileList = os.listdir(datasetFolder)
fileList.sort()

def getFileInformation(filename):
    xSensor = 4.8 if "nm" in filename else 10.9
    ySensor = 3.6 if "nm" in filename else 8.7
    with tifffile.TiffFile(os.path.join(datasetFolder, filename)) as tif:
        tif_tags = {}
        exf_tag = tif.pages[0].tags["ExifTag"]
        gps_tag = tif.pages[0].tags["GPSTag"]
        
        for tag in tif.pages[0].tags.values():
            name, value = tag.name, tag.value
            tif_tags[name] = value
        image = tif.pages[0].asarray()
    myInfo = {}
    myFV = {}
    for key, value in exf_tag.value.items():
        if str(key) in "FocalLength":
            focalLenght = value[0]
            fVW = 2*atan(xSensor/(2*focalLenght))
            fVT = 2*atan(ySensor/(2*focalLenght))
            myFV["fVW"] = fVW
            myFV["fVT"] = fVT
        else:
            print(str(key)+":"+str(value)+"\n")

    print("******GPS tags******")
    myInfo["image"] = filename
    count = 0
    for key, value in gps_tag.value.items():
        if str(key) == "GPSLatitude":
            myInfo["latitude"] = (value[0]/value[1])+((value[2]/value[3])/60)+((value[4]/value[5])/3600)
        elif key in "GPSLongitude": 
            myInfo["longitude"] = (value[0]/value[1])+((value[2]/value[3])/60)+((value[4]/value[5])/3600)
        elif key in "GPSAltitude":
            altitude = value[0]
            bottom = altitude*tan(-0.5*myFV["fVW"])
            top = altitude*tan(0.5*myFV["fVW"])
            left = altitude*tan(-0.5*myFV["fVT"])
            right = altitude*tan(0.5*myFV["fVT"])
            myInfo["heightFootprint"] = right - left
            myInfo["widthFootprint"] = top - bottom
            myInfo["distance"] = sqrt((right - left)**2 + (top - bottom)**2)/2
        else:
            print(key, value)
    if len(myInfo) == 6:
        print(myInfo)
        return myInfo


def getAllInformation():
    for fileIndex in range(len(fileList)):
        # get bearing between two files
        nextFile = fileIndex + 1
        myDestination = []
        if nextFile  < len(fileList):
            currentFileInformation = getFileInformation(fileList[fileIndex])
            nextFileInformation  = getFileInformation(fileList[nextFile])
            currentFileInformation['bearing'] = bearing((currentFileInformation['latitude'],currentFileInformation['longitude']),(nextFileInformation['latitude'],nextFileInformation['longitude']))
            currentFileInformation['betha2'] = 90 -  degrees(asin((currentFileInformation["heightFootprint"]/2)/(currentFileInformation["widthFootprint"]/2)))
            currentFileInformation["betha3"] = 90 + currentFileInformation["betha2"]
            currentFileInformation["betha4"] = 180+ currentFileInformation["betha2"]
            currentFileInformation["betha1"] = 180+ currentFileInformation["betha3"]
            myDestination.append( destination((currentFileInformation["latitude"], currentFileInformation["longitude"]),currentFileInformation["distance"], currentFileInformation["betha2"]+currentFileInformation["bearing"]))
            myDestination.append( destination((currentFileInformation["latitude"], currentFileInformation["longitude"]),currentFileInformation["distance"], currentFileInformation["betha3"]+currentFileInformation["bearing"]))
            myDestination.append( destination((currentFileInformation["latitude"], currentFileInformation["longitude"]),currentFileInformation["distance"], currentFileInformation["betha4"]+currentFileInformation["bearing"]))
            myDestination.append( destination((currentFileInformation["latitude"], currentFileInformation["longitude"]),currentFileInformation["distance"], currentFileInformation["betha1"]+currentFileInformation["bearing"]))


            
            filn.write(currentFileInformation["image"]+"\t"+ str(currentFileInformation["latitude"])+"\t"+ str(currentFileInformation["longitude"])+"\t"+ str(currentFileInformation["heightFootprint"])+"\t"+ str(currentFileInformation["widthFootprint"])+"\t"+ str(currentFileInformation["bearing"])+"\t"+str(myDestination[0][0])+" "+str(myDestination[0][1])+"\t"+str(myDestination[1][0])+" "+str(myDestination[1][1])+"\t"+str(myDestination[2][0])+" "+str(myDestination[2][1])+"\t"+str(myDestination[3][0])+" "+str(myDestination[3][1]))
        filn.write("\n")
    filn.close()
        


def main():
    getAllInformation()
    
            
if __name__=="__main__":
    main()
