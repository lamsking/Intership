
    
import os
os.chdir('/home/inra-cirad/Bureau/MonDossier/out_vol1')  
i=0
for file in os.listdir():
    src = file
    #dst ="imv1g"+str(i)+"_lwir.tif"
    dst ="imgv1_"+str(i)+"_lwir"+".tif"
    os.rename(src,dst)
    i+=1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
