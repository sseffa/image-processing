import numpy as np
import matplotlib.pyplot as plt


img=plt.imread("assets/one.jpg")
img.shape


img2=np.zeros((28,28), dtype=np.uint8)
img2.shape
img2=img[:,:,0]
img3=np.zeros((28,28), dtype=np.uint8)
# img3=img[:,:,0]
plt.imshow(img2, cmap='gray')
plt.show()


m,n=img2.shape


for i in range(1,m-1):
    for j in range(1,n-1):
        s = img2[i-1,j-1]/9 + img2[i-1,j]/9 + img2[i-1,j+1]/9 + img2[i,j-1]/9 + img2[i,j]/9 + img2[i,j+1]/9 + img2[i+1,j-1]/9 + img2[i+1,j]/9 + img2[i+1,j+1]/9
        s=int(s)
        
        # s=img2[i-1,j+1]
        # print(s, end=' * ')
        img3[i,j]=s


plt.subplot(1,2,1)
plt.imshow(img2, cmap='gray')

plt.subplot(1,2,2)
plt.imshow(img3, cmap='gray')


plt.imsave("assets/temp.jpg", img2,cmap='gray')
plt.imsave("assets/temp2.jpg", img3,cmap='gray')