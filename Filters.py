## Basic Libraries Required ##
import numpy as np
import skimage.io
import skimage.viewer
import matplotlib.pyplot as plt

# Read Image
image = skimage.io.imread(fname="images/noisystars1.jpg")

# Convert to Grayscale
g_image = skimage.color.rgb2gray(image)

# Display Image
plt.imshow(g_image, cmap="gray")
plt.show()

# BLURRING : Gaussian Filter
sigma = 20.0 # blur amount
blurred = skimage.filters.gaussian(g_image, sigma=(sigma, sigma), truncate=3.5, multichannel=True)

plt.imshow(blurred, cmap="gray")
plt.show()

# EDGE DETECTION : Sobel Filter
edges = skimage.filters.sobel(g_image)

plt.imshow(edges, cmap="gray")
plt.show()

# NOISE REDUCTION : Median Filter
less_noise = skimage.filters.median(g_image)

plt.imshow(less_noise, cmap="gray")
plt.show()
