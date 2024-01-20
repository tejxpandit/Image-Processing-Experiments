## Basic Libraries Required ##
import matplotlib.pyplot as plt
import skimage

# More libraries
from skimage import data
from matplotlib.patches import Rectangle

# Read Image
image = skimage.io.imread(fname="images/Akerkars Colorized.jpg")

# Load Trained Face Cascade Model
trained = data.lbp_frontal_face_cascade_filename()

# Initialize Detector
detector = skimage.feature.Cascade(trained)

# Face Detection
faces = detector.detect_multi_scale(img=image,
                                    scale_factor=1.2,
                                    step_ratio=1,
                                    min_size=(60, 60),
                                    max_size=(123, 123))

fig, ax = plt.subplots()
plt.imshow(image)

# Bounding Box
for face in faces:
    ax.add_patch(Rectangle((face['c'], face['r']), face['width'], face['height'], fill=False, color='r', linewidth=2))

plt.show()

# # Display Faces
# fig, ax = plt.subplots(1, len(faces), figsize=(15,5))
# ax_n = 0

# for face in faces:
#     minX, minY, maxX, maxY = face['c'], face['r'], face['c']+face['width'], face['r']+face['height']

#     # Filter Out Small Errors
#     ax[ax_n].imshow(image[minX:maxX, minY:maxY])
#     ax[ax_n].get_xaxis().set_visible(False)
#     ax[ax_n].get_yaxis().set_visible(False)
#     ax_n += 1

# fig.tight_layout()
# plt.show()