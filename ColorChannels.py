## Basic Libraries Required ##
import numpy as np
import skimage.io
import skimage.viewer
import matplotlib.pyplot as plt

# Read Image
image = skimage.io.imread(fname="images/rgb.jpg")

# Image Dimensions
print(image.shape)

# View Color Channel (RGB)
red_channel = image * [1,0,0]
green_channel = image * [0,1,0]
blue_channel = image * [0,0,1]

plt.imshow(red_channel)
plt.show()

# Get Actual Color Channel (RGB)
red_channel = image[:,:,0]
green_channel = image[:,:,1]
blue_channel = image[:,:,2]

plt.imshow(red_channel, cmap='gray')
plt.show()

# Save Image
skimage.io.imsave(fname="images/temp.jpg", arr=red_channel)