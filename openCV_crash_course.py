#import opencv for images
import cv2
import matplotlib.pyplot as plt
import numpy as np

#basic function for catching overflow

#grayscale
image_path = "images/number_zero.jpg"

#read miage to matrix
img = cv2.imread(image_path,0)

#show matix
print(img)

#properties
print("Data type = {}\n".format(img.dtype))
print("Object type = {}\n".format(type(img)))
print("Image Dimensions = {}\n".format(img.shape))

# openCV => imshow()

#index is [y,x] in array, or [row, column]
print(img[0,0])
print(img[1,1])
print(img[2,2])

#I want 249 at x=4, y=2, note: start at 0
print(img[2,4])

#region of interest
#roi = img[startY:endY, startX:endX]
roi = img[1:12, 3:9]
print(roi)

#modify pixel values
img[0,0] = 255

#region of interest
img[11:12,2:8] = 180

#matplotlib, show image
#plt.imshow(img, cmap='gray')
#plt.show()
print(img)

### color images
img2 = cv2.imread(image_path)

img2[0,0] = (0,255,255)
#blue
img2[1,1] = (255,0,0)

#region of interest
img2[11:12,2:8] = (0,255,255)

plt.imshow(img2[:,:,::-1])
plt.show()

#adding
img2[1,0] += np.uint8((50,50,50))

#multiplying
print(img2[3,0])
img2[3,0] *= np.uint8((50,50,50))
print(img2[3,0])

plt.imshow(img2[:,:,::-1])
plt.show()