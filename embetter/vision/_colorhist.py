import numpy as np
import skimage
import matplotlib.pyplot as plt
 
image = skimage.data.chelsea()
image_red, image_green, image_blue = image[:,:,0], image[:,:,1], image[:,:,2]
 
fig, ax = plt.subplots(2,3)
ax[0,0].imshow(image_red, cmap='gray')
ax[0,1].imshow(image_green, cmap='gray')
ax[0,2].imshow(image_blue, cmap='gray')
 
bins = np.arange(-0.5, 255+1,1)
ax[1,0].hist(image_red.flatten(), bins = bins, color='r')
ax[1,1].hist(image_green.flatten(), bins=bins, color='g')
ax[1,2].hist(image_blue.flatten(), bins=bins, color='b')
