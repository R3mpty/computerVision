import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

# Step 1: Obtian faces images as training data (Currently using a library)

# Download Olivetti faces dataset
trainningData = fetch_olivetti_faces()
dolphin = plt.imread('dolphin.jpeg')
trainningData = [] #TODO: automate this process for all data

# Step 2: Represeant every image I as a vector i
# (Flatening the image)
def flatternImage(image):
    return image.flatten('F') # This F is important, it flatterns it the way we want it


# Step 3: Calculating the mean face image
trainningData.append(flatternImage(dolphin)) #TODO: automate this process for all data

def meanFaceVector(trainingData):
    M = len(trainingData)
    return (1/M) * np.sum(trainingData)  

# Step 4: Subtract the mean image from the original image vector
subtracted = [] # difference between each image and the mean
for image in trainningData:
    subtracted.append(image - meanFaceVector(trainningData))

# Step 5: Calculating the covariance matrix
A = np.concatenate(subtracted)
# print(A)
C = (1/len(trainningData)) * (A * np.matrix.transpose(A))
# print(C)

# Step 6: Compute the eigen values of eigen vecotrs of C
def findingEigenvalues(c):
    return np.linalg.eig(c)

eigen_vals, eigen_vecs = findingEigenvalues(c)


# Step 7:





# Step 8:









