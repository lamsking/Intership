#!/bin/python
'''
import os
import sys
from PIL import Image
from PIL.ExifTags import TAGS

image = sys.argv[1]

for (tag,value) in Image.open(image)._getexif().iteritems():
        print( '%s = %s' % (TAGS.get(tag), value))
'''
'''
import exifread
import numpy as np 
# Open image file for reading (binary mode)
f = open('ok.tif', 'rb')
#f = open('foo.jpg', 'rb')
#print(np.shape(f))

# Return Exif tags
tags = exifread.process_file(f)

# Print the tag/ value pairs
for tag in tags.keys():
    if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
        print("Key: %s, value %s" % (tag, tags[tag])) 
'''

import tifffile
with tifffile.TiffFile('ok.tif') as tif:
    tif_tags = {}
    for tag in tif.pages[0].tags.values():
        name, value = tag.name, tag.value
        tif_tags[name] = value
    image = tif.pages[0].asarray()
    print(tif_tags)
    