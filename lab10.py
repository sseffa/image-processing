# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:57:35 2019

@author: Sefa
"""

import random

def sumTwoVec(v1,v2):    
    s = len(v1)
    resultVec = []    
    for i in range(s):
        temp = v1[i] + v2[i]
        resultVec.append(temp)
    return resultVec


def createVec(m=5,n=2):
    vec = []
    for i in range(m):
        vec.append([])
        for j in range(n):
            vec[i].append(random.randint(-10,10))        
    return vec

def findCenter(v1,v2):
    s = len(v1)
    resultVec = []    
    for i in range(s):
        temp = (v1[i] + v2[i]) / 2
        resultVec.append(temp)
    return resultVec
    

def distance(v1,v2):
     s = len(v1)
     t = 0
     
     for i in range(s):
         t = t + (v1[i] - v2[i]) ** 2

     return  t**0.5

v1 = [0,4,0]
v2 = [3,0,0]

dis = distance(v1,v2)

print(dis)
