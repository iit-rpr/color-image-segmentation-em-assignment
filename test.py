# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import copy
import matplotlib.pyplot as plt
import os
from os.path import join
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
from skimage.color import rgb2gray
from skimage.color import label2rgb
from skimage.filters import gaussian
from sklearn.cluster import KMeans


imgNames = ['water_coins','jump','tiger']
segmentCounts = [2,3,4,5]
list_of_images=[]
row=3
column=4
i=1
for imgName in imgNames:
    for SegCount in segmentCounts:
        inputpath = join(''.join(['Input/', imgName , '.png']));
        inputpath = join(''.join(['Output/', str(SegCount), '_segments/', str(imgName) ,'/', '0.png']));
        #inputpath=join(''.join(['Output/',str(SegCount), '_segments/', imgName , '/', '0.png']));
        img= mpimg.imread(inputpath)
        plt.subplot(row, column, i)
        i=i+1
        plt.imshow(img)
        plt.title(imgName + "segment_"+ str(SegCount))
        plt.xticks([])
        plt.yticks([])
        #list_of_images.append(img)
plt.show()     

#np.sum(list4,axis=0)
#print(list4)
#print(list2[19][2],"    ", list1[4][3][2]  )