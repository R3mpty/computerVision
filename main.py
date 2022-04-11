import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

# Get pass odd mac security issue
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

### Algorithm ####
# Step 1: Obtain faces images as training data (Currently using a library)
# Download Olivetti faces dataset
trainingData = fetch_olivetti_faces().images
flattenedImage = []

# Step 2: Represeant every image I as a vector i
# (Flatening the image)
for image in trainingData:  
    flattenedImage.append(image.flatten('F')) # This F is important, it flatterns it the way we want it

# Step 3: Calculating the mean face image
meanFaceVector = (1 / len(flattenedImage)) * np.sum(flattenedImage)  

# Step 4: Subtract the mean image from the original image vector
subtracted = [] # difference between each image and the mean
for image in flattenedImage:
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
eigenValsSorted = sorted(eigenData.items(), key=lambda x: x[1], reverse=True)
bestKEigenvectors = eigenValsSorted[:BEST_K]

# Representing faces onto this basis <--

# Step 7: How to reconstruct image I <--
# I = meanFaceVector + eigen_vecs * weight

# Step 8: Collecting weight
# Weight is the product of image (vector) with each of the eigen vectors [9:08]
weights = np.empty([4096, 400])


# Step 9: Euclidean distance between the mean face and the test image <-- Not in the implementation section?
# def euclideanDistance(x,y):
#     return np.linalg.norm(x-y)


### Testing ###



