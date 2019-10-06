# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 19:58:35 2019

@author: Sefa
"""

import numpy as np
import matplotlib.pyplot as plt

img=plt.imread("assets/sample.png")


plt.imshow(img)
plt.show()

print("Resmin boyutu = ",img.ndim) 
print("Resmin shape değeri = ",img.shape)
print("Red için min değer = ",img[:,:,0].min()) 
print("Red için max değer = ",img[:,:,0].max()) 

print("en kucuk kirmizi renk degeri : ",np.min(img[:,:,0]))
print("en kucuk kirmizi renk degeri : ",np.max(img[:,:,0]))
print("en kucuk yesil renk degeri : ",np.min(img[:,:,1]))
print("en kucuk yesil renk degeri : ",np.max(img[:,:,1]))
print("en kucuk mavi renk degeri : ",np.min(img[:,:,2]))
print("en kucuk mavi renk degeri : ",np.max(img[:,:,2]))


print(img.ndim)
print(img.shape)

m, n, p = img.shape

print(m,n,p)


new = np.zeros((m,n),dtype=float)


for i in range(m):
    for j in range(n):
        s = (img[i,j,0] + img[i,j,1] + img[i,j,2]) / 3
        new[i,j] = s
           
plt.imshow(new)
plt.show()        
        
plt.imshow(new,cmap='gray')
plt.show()


plt.imsave('assets/new.png',new,cmap='gray')


"""-----------------------------"""

transpozed = np.zeros((n,m), dtype=float)

for i in range(m):
    for j in range(n):
        s = (img[i,j,0] + img[i,j,1] + img[i,j,2]) / 3
        transpozed[j,i] = s
		
plt.imshow(transpozed)
plt.show()





















