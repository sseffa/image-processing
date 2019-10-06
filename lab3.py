# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:30:35 2019

@author: Sefa
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


"""img=mpimg.imread('assets/sample.png')"""
img=mpimg.imread('assets/ankara.jpg')


plt.subplot(1,2,1),plt.imshow(img)
plt.subplot(1,2,2),plt.imshow(255-img)         #  inverse burada ; 255-I 

def my_histogram(image):
    
    my_H_R={}
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if(image[i,j,0] in my_H_R.keys()):
                my_H_R[image[i,j,0]]+=1
            else:
                my_H_R[image[i,j,0]]=1
    
    my_H_G={}
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if(image[i,j,1] in my_H_G.keys()):
                my_H_G[image[i,j,1]]+=1
            else:
                my_H_G[image[i,j,1]]=1
    my_H_B={}
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if(image[i,j,2] in my_H_B.keys()):
                my_H_B[image[i,j,2]]+=1
            else:
                my_H_B[image[i,j,2]]=1
    return (my_H_R,my_H_G,my_H_B)
                
        
my_histogram=my_histogram(img)



x=[]
y=[]
for i in my_histogram[0].keys():
    x.append(i)
    y.append(my_histogram[0][i])
plt.subplot(1,3,1),plt.plot(x,y)    

x=[]
y=[]
for i in my_histogram[1].keys():
    x.append(i)
    y.append(my_histogram[1][i])
plt.subplot(1,3,2),plt.plot(x,y)
x=[]
y=[]

for i in my_histogram[2].keys():
    x.append(i)
    y.append(my_histogram[2][i])
plt.subplot(1,3,3),plt.plot(x,y)