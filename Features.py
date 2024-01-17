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

# CORNER DETECTION
# Sheared checkerboard
tform = skimage.transform.AffineTransform(scale=(1.3, 1.1), rotation=1, shear=0.7,
                        translation=(110, 30))
image = skimage.transform.warp(skimage.data.checkerboard()[:90, :90], tform.inverse,
             output_shape=(200, 310))
# Ellipse
rr, cc = skimage.draw.ellipse(160, 175, 10, 100)
image[rr, cc] = 1
# Two squares
image[30:80, 200:250] = 1


coords = skimage.feature.corner_peaks(skimage.feature.corner_harris(image), min_distance=5, threshold_rel=0.02)
coords_subpix = skimage.feature.corner_subpix(image, coords, window_size=13)

fig, ax = plt.subplots()
ax.imshow(image, cmap=plt.cm.gray)
ax.plot(coords[:, 1], coords[:, 0], color='cyan', marker='o',
        linestyle='None', markersize=6)
ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
ax.axis((0, 310, 200, 0))
plt.show()