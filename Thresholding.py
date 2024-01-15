## Basic Libraries Required ##
import numpy as np
import skimage.io
import skimage.viewer
import matplotlib.pyplot as plt

# Read Image
image = skimage.io.imread(fname="images/noisystars1.jpg")

# Convert to Grayscale
g_image = skimage.color.rgb2gray(image)

# Image Dimensions
print(g_image.shape)

# Display Image
plt.imshow(g_image, cmap="gray")
plt.show()

# Create Histogram
hist, bin_edges = np.histogram(g_image, bins=256, range=(0, 1))

plt.plot(bin_edges[0:-1], hist)
plt.show()

# Thresholding (0.2-0.5)
th_image = g_image > 0.2

plt.imshow(th_image, cmap="gray")
plt.show()

# Automatic Thresholding
auto_threshold = skimage.filters.threshold_otsu(g_image)
ath_image = g_image > auto_threshold

plt.imshow(ath_image, cmap="gray")
plt.show()