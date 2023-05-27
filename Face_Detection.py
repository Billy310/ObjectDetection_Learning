import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data,exposure

image = data.astronaut()

fd,hot_image = hog(image,orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),visualize=True,channel_axis=True)
# multichannel_output=False
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,6),sharex=True,sharey=True)

ax1.axis("off")

ax1.imshow(image,cmap=plt.cm.gray)

ax1.set_title('Input image')

hot_image_rescaled = exposure.rescale_intensity(hot_image,in_range=(0,10))

ax2.axis("off")

ax2.imshow(hot_image_rescaled,cmap=plt.cm.gray)

ax2.set_title('Histogram of Oriented Gradients')

plt.show()