'''
This exercise script is based on the blog by Victor Zhou:

CNNs, Part 1: An Introduction to Convolutional Neural Networks
(https://victorzhou.com/blog/intro-to-cnns-part-1/)

'''

import numpy as np
import matplotlib.pyplot as plt

class Conv3x3:
    # A convolution layer using 3x3 filters

    def __init__(self, num_filters):
        print('Calling __init__ of Conv3x3 class . . .')
        self.num_filters = num_filters

        # filters are a 3d array with dimensions (num_filters, 3, 3)
        # We divide by 9 to reduce the variance of our initial values
        self.filters = np.random.rand(num_filters, 3, 3) / 9  # num_filters random 3x3 matrices 


    def iterate_regions(self, image):
        '''
        Generates all possible 3x3 image regions using valid padding.
        (Move the 3x3 filter across the image regions)
        -image is a 2d numpy array
        '''

        # get the height (h) and width(w) of the image (2D array)
        h, w = image.shape

        # a 3x3 filter on an image (h,w) can ONLY cover total steps: (h-2), (w-2)
        # (draw filter overlay with (h,w) image on paper to convince ourselves)
        for i in range(h - 2):
            for j in range(w - 2):
                # select the region from the image covered by the 3x3 filter
                # for example, image[0:3, 0:3] only selects the 1st 3 elements (0, 1, 2) of (height, width)
                im_region = image[i:(i+3), j:(j+3)]  

        
# instantiate the Conv3x3 class
conv = Conv3x3(8)

print('number of filters = ', conv.num_filters)
print('matrices = ', conv.filters)


# (testing) create a dummy image of nxn pixels from random numbers
# image = np.random.rand(60,28) # (y, x) -> (height, width)
# h, w = image.shape
# print('h,w=',h,', ',w)
# plt.imshow(image)
# plt.show()
