#import opencv for images
import cv2
import matplotlib.pyplot as plt
import numpy as np

def loadImage(image_path):
    img = cv2.imread(image_path)

    #return arrays
    return img

def showImage(img):
    plt.imshow(img[:,:,::-1])
    plt.show()

def processImage(img):
    #define a
    pass

def createUniformKernel(rowSize,colSize):
    #creates a uniform array
    return np.full((rowSize,colSize), 1/(rowSize*colSize))

if __name__ == "__main__":
    img = loadImage("images/number_zero.jpg")

    #define a kernel
    kernel = createUniformKernel(3,3)
    print(kernel)
    #process image

    showImage(img)
