
    
import os
os.chdir('/home/inra-cirad/vol/vol3/D3')  
i=893
for file in os.listdir():
    src = file
    #dst ="imv1g"+str(i)+"_lwir.tif"
    dst =str(i)+"_v2_lwir"+".tif"
    os.rename(src,dst)
    i+=1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
