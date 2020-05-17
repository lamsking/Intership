'''
script qui permet de fusioner plusieur image en une seule de façons horizontale
et verticale
'''

import numpy as np
import PIL

list_im = ['C:\\Users\\INRA\\Desktop\\workshop\\0098_530nm.tif', 'C:\\Users\\INRA\\Desktop\\workshop\\0098_570nm.tif',
           'C:\\Users\\INRA\\Desktop\\workshop\\0102_675nm.tif', 'C:\\Users\\INRA\\Desktop\\workshop\\0102_730nm.tif',
           'C:\\Users\\INRA\\Desktop\\workshop\\0102_850nm.tif']
imgs    = [ PIL.Image.open(i) for i in list_im ]
# pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )

# save that beautiful picture
imgs_comb = PIL.Image.fromarray( imgs_comb)
imgs_comb.save( 'C:/Users/INRA/Desktop/workshop/0098.tif' )    

# for a vertical stacking it is simple: use vstack
imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
imgs_comb = PIL.Image.fromarray( imgs_comb)
imgs_comb.save( 'C:/Users/INRA/Desktop/workshop/0098_verti.tif' )



























