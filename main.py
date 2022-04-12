from locale import normalize
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

# Get pass odd mac security issue
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

### Algorithm ####
# Step 1: Obtain faces images as training data (Currently using a library)
# Download Olivetti faces dataset
rawData = fetch_olivetti_faces().images
trainingData = rawData[:-20]
testingData = rawData[-20:]
flattenedImages = []

# Step 2: Represeant every image I as a vector i
# (Flattening the image)
for image in trainingData:  
    flattenedImages.append(image.flatten('F')) # This F is important, it flatterns it the way we want it

# Step 3: Calculating the mean face image
meanFaceVector = (1 / len(flattenedImages)) * np.sum(flattenedImages)  

# Step 4: Subtract the mean image from the original image vector
subtracted = [] # difference between each image and the mean
for image in flattenedImages:
    subtracted.append(image - meanFaceVector)
subtracted = np.asarray(subtracted)

# Step 5: Calculating the covariance matrix
A = np.matrix.transpose(subtracted)
C = (1/len(trainingData)) * np.matmul(A, np.matrix.transpose(A))

# Step 6: Compute the eigenvalues and eigenvectors of C
eigenVals, eigenVecs = np.linalg.eig(C)
print(eigenVals.shape)
print(eigenVecs.shape)
eigenData = {}
eiginVecsList = np.nditer(eigenVecs)
for index, val in enumerate(eigenVals):
    eigenData[val] = eigenVecs[index]

# Select best k eigenvectors (k largest eigence values)
BEST_K = 6
eigenValsSorted = sorted(eigenData.items(), key=lambda x: x[0], reverse=True)
bestKEigenvectors = eigenValsSorted[:BEST_K]

# Representing faces onto this basis

# Step 7: How to reconstruct image I <--
# I = meanFaceVector + eigen_vecs * weight

# Step 8: Collecting weight
# Weight is the product of image (vector) with each of the eigen vectors [9:08]
weights = []

for img in flattenedImages:
    faceMinusAvg = img - meanFaceVector
    weights.append(np.linalg.norm(faceMinusAvg - meanFaceVector))

print(weights[:10])
print(len(weights))

# Step 9: Euclidean distance between the mean face and the test image <-- Not in the implementation section?
# def euclideanDistance(x,y):
#     return np.linalg.norm(x-y)


### Testing ###
# Step 1: Normalize the test image -> I = Test_Image - Average_Face_Vector
# normalize
# for img in testingData:
#     flattened = image.flatten('

# Step 2: Find weights of test image


# Step 3: Calculate error between weights of test image and weights of all images in training set.


# Step 4: Choose the image of training set which has minimum error.

