# Project : Combining Images Test
# Author : Tej Pandit

## Basic Libraries Required ##
import numpy as np
import skimage.io
import skimage.viewer
import matplotlib.pyplot as plt

# Read Image
image = skimage.io.imread(fname="images/objects.jpg")

# Get Actual Color Channel (RGB)
red_channel = image[:,:,0]
green_channel = image[:,:,1]
blue_channel = image[:,:,2]

# Show a color channel
plt.imshow(blue_channel, cmap='gray')
plt.show()

# Combine color channels
combined = np.dstack((red_channel, green_channel, blue_channel))

# Show recombined image
plt.imshow(combined)
plt.show()
