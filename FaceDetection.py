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

