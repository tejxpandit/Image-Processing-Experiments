## Basic Libraries Required ##

# For accessing and manipulating pixels
import numpy as np

# For reading and writing to images
import skimage.io

# For viewing images
import skimage.viewer

# For plotting statistics
import matplotlib.pyplot as plt

# Read Image
image = skimage.io.imread(fname="images/cells1.jpg")

# Plot Image
plt.imshow(image)

# Display Image
plt.show()

# Image Dimensions
print(image.shape)