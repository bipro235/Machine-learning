# import the dependencies

import cv2
import numpy as np

image = cv2.imread(r'C:\Users\bipro\OneDrive\Pictures\test_image.webp')
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)   

# perform the edge detection 
# Here we are using canny edge detection

output_image = cv2.Canny(image, 80, 120)

# Display the image

cv2.imshow('Canny', output_image)
cv2.waitKey()
