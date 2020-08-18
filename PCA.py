import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA # PCA library from sklearn 
import pandas as pd 
import math 
import random 

# consider case where y = n * x ; so relation of x and y is scalable.

n = 1 # amount of correlation 
x = np.random.uniform(1,2,1000) # taking generic 1000 samples from uniform random variable

y = x.copy() * n # here I making y = n * x

# centring of data for PCA to make it work better.

x = x - np.mean(x) # centre the x and remove its mean 
y = y - np.mean(y) # centre y and remove its mean. 

# create dataframe with x and y 

data = pd.dataFrame({'x': x, 'y': y}) 

# plot the original un corelated data 

plt.scatter(data.x, data.y)

# initiate the PCA and choose 2 output variables 

pca = PCA(n_components=2)

# create transformation model for this data to get it to rotate(use of rotation matrices)

pcaTr = pca.fit(data)
rotatedData = pcaTr.transform(data) # transofrm data base on rotation of matrix of pcaTr.

# create dataframe with new variables as PC1 and PC2.

dataPCA = pd.dataFrame(data = rotatedData, columns = ['PC1' , 'PC2'])

# plot the transformed data 

plt.scatter(dataPCA.PC1 , dataPCA.PC2)
plt.show()


# Rotation matrices 

"""
pcaTr.components_ has the rotation matrix
pcaTr.explained_variance_ has the explained variance of each principal component

"""

print('Eigenvectors or principal component : First row must be in direction of [1,n]')
print(pcaTr.components_)

print('...............')

print('Eigenvalues or explained variance')
print(pcaTr.explained_variance_)