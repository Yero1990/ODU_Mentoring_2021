import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


#read image file
img = mpimg.imread('./stinkbug.png')

height, width = img.shape

#print image dimensions
print('height, width = ', height,', ',width)
print('pixels = height x width = ', height*width)

#print image
plt.imshow(img)
plt.show()

# print array representation of image
# index [irow, jcol] = [0,0] : top left corner pixel
print(img)
