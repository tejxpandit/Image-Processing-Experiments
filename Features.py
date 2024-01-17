## Basic Libraries Required ##
import numpy as np
import skimage.io
import skimage.viewer
import matplotlib.pyplot as plt

# Read Image
image = skimage.io.imread(fname="images/shapes.jpg")

# Convert to Grayscale
g_image = skimage.color.rgb2gray(image)

# Display Image
plt.imshow(g_image, cmap="gray")
plt.show()

# EDGE DETECTION : Canny Filter
edges = skimage.feature.canny(g_image)

plt.imshow(edges, cmap="gray")
plt.show()

# ORIENTATION DETECTION : HOG Filter (histogram of oriented gradients)
fd, hog_image = skimage.feature.hog(image, orientations=8, pixels_per_cell=(16, 16),
                                    cells_per_block=(1, 1), visualize=True, channel_axis=-1)

hog_image_rescaled = skimage.exposure.rescale_intensity(hog_image, in_range=(0, 10))

plt.imshow(hog_image_rescaled, cmap="gray")
plt.show()