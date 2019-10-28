import numpy as np
import matplotlib.pyplot as plt

image_size = 28  # width and length
no_of_different_labels = 10  # i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
train_data = np.loadtxt("assets/mnist/mnist_train.csv", delimiter=",", skiprows=1)
"""test_data = np.loadtxt("assets/mnist/mnist_test.csv", delimiter=",", skiprows=1)"""

""" print(test_data[:10]) """

print(train_data.ndim, train_data.shape)  # 785 = 28*28 + 1

print(train_data[10, 0])  # 10. satırın ilk elemanı

im3 = train_data[10, :]
im4 = im3[1:]
im5 = im4.reshape(28, 28)
"""
plt.imshow(im5, cmap='gray')
plt.show()
"""
"""m, n = train_data.shape"""

"""
def numCounter(k=0):
    s = 0
    for i in range(m):
        if (train_data[i, 0] == k):
            s = s + 1
    return s


for i in range(10):
    c = numCounter(i)
    print(i, " ", c)
"""

import math
eps = np.finfo(float).eps

def pdf(x, mu=0.0, sigma=1.0):
    
    try:
        x = float(x - mu) / sigma
        return math.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi) / sigma
    except:
      return 0


# print(pdf(10, 1, 3)) # test pdf function

def getMeanAndStd(k=0, l=350):
    
    try:
        m, n = train_data.shape
        # k, l = 0, 350
        s, t = 0, 0
        """
        k =
        s =
        t = intensity info
        l = location
        """
        for i in range(m):
            if (train_data[i, 0] == k):
                s = s + 1
                t = t + train_data[i, l + 1]
    
                """digit_class = train_data[i, 0]
                top_left = train_data[i, 1]
                bottom_right = train_data[i, 784]
                print(digit_class, top_left, bottom_right)
                print(digit_class, end=" ")
                print(top_left, end=" ")
                print(bottom_right, end="\n")"""
    
        # print(t / s)  # ortalaması
        mean1 = t / s
    
        s = 0
        t = 0
        for i in range(m):
            if (train_data[i, 0] == k):
                s = s + 1
                diff1 = train_data[i, l + 1] - mean1
                t = t + diff1 * diff1
        var1 = t / (s - 1)
        std1 = np.sqrt(var1)
    
        # print(mean1, std1)  # standart sapması
        return mean1, std1
    except:
      return 0,0
  
m1, std1 = getMeanAndStd(1, 100)

print(m1, std1)

# x = train_set[100,:] | train_set[200:400]
#print(train_data[100,:])


print(pdf(45.8, 4.0, 2.0))

#print(pdf(40, m1, std1)) # 40 intensity değerinin bulunma olasılığı


img = plt.imread('assets/one.jpg')
plt.imshow(img)
plt.show()

shape = img.shape

print(shape)
print(img[14,:])

img2 = img[:,:,0]

print(img2.shape)
print(img2[14,:])

img3 = img2.reshape(1,784)

print(img3)


m2, std2 = getMeanAndStd(2, 100)
#♠print(pdf(test_value, m2, std2)) # 40 intensity değerinin bulunma olasılığı


pdfVal = 0


myList = []
maxPdfVal = 0

for i in range(10):
    pdfVal = 0
    for j in range(784):
        x=img3[0,j]
        mean1, stand1 = getMeanAndStd(i,j)
        pdfVal = pdf(x,mean1, stand1)
        print("pdf", pdfVal)
        myList.append(pdfVal);
        
    
    
    maxPdfVal = max(myList)
        