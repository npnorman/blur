#import opencv for images
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

def loadImage(image_path):
    img = cv2.imread(image_path)

    #return arrays
    return [img,img.copy()]

def showImage(img):
    plt.imshow(img[:,:,::-1])
    plt.show()

def processImage(kernel, img, img_original):
    #input: kernel: region of interest(roi), image
    #output: image that has been blurred

    #number of rows and cols
    count = img.shape

    #limit (row, col)
    limit = kernel.shape
    limit = list(limit)
    for i in range(0, len(limit)):
        limit[i] = math.floor(limit[i]/2)

    #for each pixel in the image
    for row in range(0, len(img)):
        for col in range(0, len(img[row])):
            #get new Pixel of Interest (POI)

            #check to make sure it is the correct size
            if (row - limit[0] >= 0 and row + limit[0] <= count[0] - 1):
                if (col - limit[1] >= 0 and col + limit[1] <= count[1] - 1):
                    img_original[row,col] = newPOI(row,col,kernel, limit, img)

def newPOI(pixRow:int, pixCol:int, kernel, limit:int, img):
    #input: pixel coordinates, kernel, region of interest
    #output: uint8 np array 0 to 255, 3 elements

    #get roi from kernel
    roi = img[pixRow - limit[0]:pixRow + limit[0] + 1, pixCol - limit[1]:pixCol + limit[1] + 1]

    #make roi into floats
    roi = roi.astype(float)

    #multiply by the kernel
    roi *= kernel

    #sum each entry in the roi
    sum = np.sum(roi, axis=(0,1))

    #make sure total is between 0 and 255
    sum = list(sum)
    for i in range(0,len(sum)):
        #round
        sum[i] = round(sum[i])

        #between 0 and 255
        if sum[i] > 255:
            sum[i] = 255
        elif sum[i] < 0:
            sum[i] = 0

    #return total as new poi
    sum = np.uint8(sum)
    return sum

def createUniformKernel(rowSize,colSize):
    #creates a uniform array
    if rowSize % 2 == 0:
        rowSize += 1
        print("adjusting row size to odd:",rowSize)
    
    if colSize % 2 == 0:
        colSize += 1
        print("adjusting col size to odd:",colSize)

    return np.full((rowSize,colSize, 3), 1/(rowSize*colSize))

def gaussianKernel(size, sigma):
    #based on guass' funtion

    # Make sure kernel size is odd
    if size % 2 == 0:
        size += 1

    # Create a coordinate grid centered at zero
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)

    # Calculate the Gaussian function
    kernel = (1 / (2 * np.pi * sigma**2)) * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    # Normalize the kernel to make the sum = 1
    kernel /= np.sum(kernel)

    expandedKernel = np.repeat(kernel[:, :, np.newaxis], 3, axis=2)

    return expandedKernel

if __name__ == "__main__":
    print("progress = 0%")

    #img = loadImage("images/number_zero.jpg")
    [img, img_original] = loadImage("images/campus-center-night-trident.png")
    [gimg, gimg_original] = loadImage("images/campus-center-night-trident.png")

    #define a kernel
    #box blur
    kernel = createUniformKernel(101,101)

    #gaussian blur
    gKernel = gaussianKernel(101,100)
    

    #process image
    processImage(kernel, img, img_original)
    print("progress = 50%")
    processImage(gKernel, gimg, gimg_original)
    print("progress = 90%")

    fig = plt.figure(figsize=(15,8))
    plt.subplot(2,2,1)
    plt.subplot(2,2,1).set_title("Box Blur")
    plt.imshow(img_original[:,:,::-1])

    plt.subplot(2,2,2)
    plt.imshow(img[:,:,::-1])

    plt.subplot(2,2,3)
    plt.subplot(2,2,3).set_title("Guass")
    plt.imshow(gimg_original[:,:,::-1])

    plt.subplot(2,2,4)
    plt.imshow(cv2.GaussianBlur(gimg, (101,101), 100))

    print("progress = 100%")

    plt.show()
    #showImage(img)