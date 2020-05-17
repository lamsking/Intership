#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
import os
import gdal

in_path = '/home/inra-cirad/Bureau/MonDossier/'
input_filename = '0_v1_lwir.tif'

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
        com_string = "gdal_translate -of GTIFF -srcwin " + str(i)+ ", " + str(j)+ ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + str(in_path)+ str(input_filename) + " " + str(out_path) + str(output_filename)+ str(i) + "_" + str(j) + ".tif"
        os.system(com_string)
        
'''

from PIL import Image

xsize = 32
ysize = 32
filename = '0_v1_lwir.tif'

img = Image.open(filename)
for x in range(img.size[0]//xsize):
    for y in range(img.size[1]//ysize):
        imgpiece = Image.crop((xsize, ysize, (x+1)*xsize, (y+1)*ysize))
        imgpiece.save(filename[:-4] + '' + str(x) + '' + str(y) + filename[-4:]) 

"""



import os
import gdal

#in_path = '/home/inra-cirad/vol/vol1/'
PATH_IMAGE = '/home/inra-cirad/Bureau/MonDossier/datav3'
k=0
fichier_path='/home/inra-cirad/Bureau/MonDossier/sortie_image3/output/coord_file3.csv'
out_path = '/home/inra-cirad/Bureau/MonDossier/sortie_image3/file/'
output_filename = 'tile_'

#fonction de lecture de fichier

def readfile(pathfile):
    fichier=open(pathfile,'r')
    return fichier.readlines()
    

def getmetadataimage(fichier,nameimage):
    lines=readfile(fichier)
    for ln in lines:
        l=ln.strip('\n').split('\t')
        if nameimage in l[0]:
            return ln
def getLongLal(metadonnee):
    return metadonnee[6],metadonnee[7],metadonnee[8],metadonnee[9]

def getMilieu(p1,p2):
    latMilieu = 0
    longMilieu = 0
    point1 = p1.split(' ')
    point2 = p2.split(' ')
    
    latP1 = float(point1[0])
    longP1 = float(point1[1])
    
    latP2 = float(point2[0])
    longP2 = float(point2[1])
    
    if latP1>latP2:
        latMilieu = ((latP1-latP2)//2)+latP2
    else:
        latMilieu = ((latP2-latP1)//2)+latP1
    
    if longP1 > longP2:
        longMilieu = ((longP1-longP2)//2)+longP2
    else:
        longMilieu = ((longP2-longP1)//2)+longP1
    return latMilieu,longMilieu
    


def get_image_paths(path_image):
    folder = path_image
    files = os.listdir(folder)
    files.sort()
    files = ['{}/{}'.format(folder, file) for file in files]
    return files
fichierAnnotation = open(out_path+"newAnnotation3.txt","w")
fichierAnnotation.write("image\tpoint2\tpoint3\tpoint4\tpoint1\n")
def split_image(image_path,k):
    
    image=image_path.split('/')
    imagename=image[len(image)-1]
    metadonnee=getmetadataimage(fichier_path,imagename)
    metadonnee=metadonnee.strip('\n').split('\t')
   
    tile_size_x = 320
    tile_size_y = 256
    
    ds = gdal.Open(image_path)
    band = ds.GetRasterBand(1)
    
    xsize = band.XSize                         
    ysize = band.YSize
    
    nbPoint=0
    for i in range(0, xsize, tile_size_x):
        latPoint1,longPoint1 = getMilieu(metadonnee[9],metadonnee[6])
        latPoint2,longPoint2 = getMilieu(metadonnee[9],metadonnee[8])
        latPoint3,longPoint3 = getMilieu(metadonnee[7],metadonnee[8])
        latPoint4,longPoint4 = getMilieu(metadonnee[6],metadonnee[7])
        latPoint5,longPoint5 = getMilieu(metadonnee[9],metadonnee[7])
        for j in range(0, ysize, tile_size_y):
            k=k+1
            print(i,j)
            com_string = "gdal_translate -of GTIFF -srcwin " + str(i)+ ", " + str(j)+ ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + str(image_path) + " " + str(out_path)+ str(output_filename) + str(i) + "_" + str(j) +"_"+ str(k)+".tif"
            com_list = com_string.split("/")
            nomImage = com_list[len(com_list)-1]
            
            if nbPoint == 0:
                fichierAnnotation.write(nomImage+"\t"+ str(latPoint1)+" "+str(longPoint1)+"\t"+str(latPoint5)+" "+str(longPoint5)+"\t"+str(latPoint2)+" "+str(longPoint2)+"\t"+metadonnee[9]+"\n")
            if nbPoint == 1:
                fichierAnnotation.write(nomImage+"\t"+ str(latPoint5)+" "+str(longPoint5)+"\t"+str(latPoint3)+" "+str(longPoint3)+"\t"+metadonnee[8]+"\t"+str(latPoint2)+" "+str(longPoint2)+"\n")
            if nbPoint == 2:
                fichierAnnotation.write(nomImage+"\t"+ metadonnee[6]+"\t"+str(latPoint4)+" "+str(longPoint4)+"\t"+str(latPoint5)+" "+str(longPoint5)+"\t"+str(latPoint1)+" "+str(longPoint1)+"\n")
            if nbPoint == 3:
                fichierAnnotation.write(nomImage+"\t"+str(latPoint4)+" "+str(longPoint4)+"\t"+ metadonnee[7]+"\t"+str(latPoint3)+" "+str(longPoint3)+"\t"+str(latPoint5)+" "+str(longPoint5)+"\n")
            nbPoint = nbPoint + 1
            os.system(com_string)
            


def main():
    k=0
    liste_image=get_image_paths(PATH_IMAGE)
    for image in liste_image:
        split_image(image,k)
        k=k+1
        print("#########################################################################################")
    fichierAnnotation.close()

if __name__ == "__main__":
    main() 



"""
import os
import shutil


dossier = "/home/inra-cirad/Bureau/MonDossier/greg/WD/"
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




























