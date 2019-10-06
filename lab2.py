# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

img=plt.imread("assets/sample.png")

plt.imshow(img)
plt.show()

img[:,:,0]=img[:,:,0]+50
plt.imshow(img)
plt.show()


im_2 = plt.imread('assets/ankara.jpg')
def my_func(im_100,s=5):
    
    m,n,p = im_100.shape
    new_image = np.zeros((m,n,3),dtype=int)
    
    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            new_image[m,n,0] = img[m,n,0] + 10 
            new_image[m,n,1] = img[m,n,1]
            new_image[m,n,2] = img[m,n,2]
            
    return new_image

im_3 = my_func(im_2)
plt.imshow(im_3)

def my_func_2(im_500):
    
    m,n,p = im_500.shape
    new_m = int(m/2)
    new_n = int(n/2)
    
    im_600 = np.zeros((m,n),dtype=int)
    
    for m in range(new_m):
        for n in range(new_n):
            s0 = (im_500[m*2,n*2,0] + im_500[m*2,n*2,1] + im_500[m*2,n*2,2])/3
            im_600[m,n] = int(s0)
            
    return im_600


im_4 = plt.imread('assets/ankara.jpg')
plt.imshow(im_4)

im_5 = my_func_2(im_4)
plt.imshow(im_5,cmap='gray')
plt.show()



def my_func_3(im_20):
    m,n,p = im_20.shape
    new_m = int(m/2)
    new_n = int(n/2)
    
    im_30 = np.zeros((m,n,3),dtype=int) 
    
    for m in range(new_m):
        for n in range(new_n):
            im_30[m,n,0] = int(im_20[m*2,n*2,0])
            im_30[m,n,1] = int(im_20[m*2,n*2,1])
            im_30[m,n,2] = int(im_20[m*2,n*2,2])
    return im_30

im_6 = plt.imread('assets/ankara.jpg')
plt.imshow(im_6)
plt.show()

"""
im_7 = my_func_2(im_6)
plt.imshow(im_7)
plt.show()
"""