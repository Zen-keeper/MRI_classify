# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:22:12 2021

@author: jsc
"""
import nibabel as nib
import os
from matplotlib import pyplot as plt
from skimage import morphology
import cv2
show_flag = 0
size  =2
def findconnecttours(path):
    images = os.listdir(path)
    for image in images:
        p = os.path.join(path,image)
        imgnii = nib.load(p).get_data()
        # imgnii[imgnii<0.2] = 0
        # imgnii = (imgnii*255).astype(int)
        for i in range(5, 60):
            array_gray = imgnii[:, :, i]
            # array_gray[array_gray<0.2] = 0
            ret, th1 = cv2.threshold(array_gray, 0.2, 1, cv2.THRESH_BINARY_INV)
            mask = morphology.remove_small_objects(th1.astype(int), min_size=30,connectivity=2)
#            array_gray = morphology.remove_small_holes(array_gray, size)
            array_gray[mask==1] = 0
            if show_flag==0 :
                plt.imshow(array_gray, cmap='gray')
                plt.waitforbuttonpress()
#
#        #二值化
#        ret,th1 = cv2.threshold(array_gray,244,255,cv2.THRESH_BINARY_INV)
#        if show_flag==0 :
#            plt.figure("Image")
#            plt.imshow(th1,cmap="gray")
#            plt.axis("off")
#            plt.show()
#        
#        label_list = []
#        _, labels, stats, centroids = cv2.connectedComponentsWithStats(th1,4)
#        #得到连通域小于100的索引
#        for label_ in range(stats.shape[0]):
#            if (stats[label_][4])<100:
#                label_list.append(label_)
#        #小于100个像素点的连通域去除
#        for x in range(labels.shape[0]):
#            for y in range(labels.shape[1]):
#                for ll in label_list:
#                    if labels[x][y]==ll:
#                        array_gray[x][y]=255
#        if show_flag==0 :
#            plt.figure("Image")
#            plt.imshow(array_gray,cmap="gray")
#            plt.axis("off")
#            plt.show()
        
findconnecttours("D:/work/AD_V3/image_class/ori/residual/AD")