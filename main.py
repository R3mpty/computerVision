import numpy as np
from matplotlib import pyplot as plt

# Step 1: Obtian faces images as training data 
dolphin = plt.imread('dolphin.jpeg')
plt.imshow(dolphin)
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
subtracted = []
for image in trainningData:
    subtracted.append(image - meanFaceVector(trainingData))



# Step 5:






