# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 13:56:31 2019

@author: Sefa
"""

import numpy as np 


def generateData():
    x = [180,175,185,160,175,150,170,170,155,175]
    y = [70,70,85,80,75,55,60,60,65,60]
    cov_mat = np.stack((x, y), axis = 0)  
    return cov_mat

def getCovMatrix(cov_mat):
    sigma = np.cov(cov_mat)
    return sigma

data = generateData()
result = getCovMatrix(data)


def univariate_normal(x, mean, variance):
    """pdf of the univariate normal distribution."""
    return ((1. / np.sqrt(2 * np.pi * variance)) * 
            np.exp(-(x - mean)**2 / (2 * variance)))
    
    
def multivariate_normal(x, d, mean, covariance):
    """pdf of the multivariate normal distribution."""
    x_m = x - mean
    return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) * 
            np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))
    

x1 = [165,75]
d1 = 2
data = generateData()
mean1 =np.array([np.mean(data[0,:]), np.mean(data[1,:])])     
covariance1 = getCovMatrix(data)

result = multivariate_normal(x1, d1, mean1, covariance1)

print(result)
