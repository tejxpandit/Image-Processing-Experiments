## Basic Libraries Required ##
import numpy as np
import skimage.io
import skimage.viewer
import matplotlib.pyplot as plt

# More libraries
from tabulate import tabulate
from matplotlib.patches import Rectangle

# Too Small
def tooSmall(minr, minc, maxr, maxc, min_size):
    w = maxc - minc
    h = maxr - minr
    return(w<min_size and h<min_size)

# Read Image
image = skimage.io.imread(fname="images/coinsUSA_set.jpg")

# Convert to Grayscale
g_image = skimage.color.rgb2gray(image)

# Display Image
plt.imshow(g_image, cmap="gray")
plt.show()

# Masking
mask = g_image < 0.9
plt.imshow(mask, cmap="gray")
plt.show()

# Blobs
blobs = skimage.measure.label(mask > 0)
plt.imshow(blobs, cmap="tab10")
plt.show()

# Blob Analysis
properties =['area','bbox','convex_area','bbox_area',
             'major_axis_length', 'minor_axis_length',
             'eccentricity']

data = skimage.measure.regionprops_table(blobs, properties = properties)
print(tabulate(data, headers='keys'))



