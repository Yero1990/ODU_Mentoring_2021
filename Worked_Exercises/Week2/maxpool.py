'''
 Neural Network: Pooling
 
 Description: Neighboring pixels in images tend to have similar values,
 so conv layers will typically also produce similar values for neighboring pixels in outputs. 
 As a result, much of the information contained in a conv layer’s output is redundant. 
 Pooling layers reduce the size of the input it’s given by pooling values together in the input. 
 The pooling is usually done by a simple operation like max, min, or average. 
 Pooling divides the input’s width and height by the pool size.
'''
# We'll implement a MaxPool2 class with the same methods as conv class

import numpy as np

class MaxPool2:
    # A Max Pooling layer using a pool size of 2

    def iterate_regions(self, image):
        '''
        Generates non-overlapping 2x2 image regions to pool over.
        - image is a 2d numpy array
        '''

        # extract the height and width of the convolved image input for a specified kth filter
        h, w, _ = image.shape
        new_h = h // 2   # the // is a floor division which is a normal division, except it returns the largest whole number. For example, 15 / 1.19  = 12.605,  whereas 15 // 1.19 = 12 (no roundup)
        new_w = w // 2

        
        # loop over the cut-by-half, (h/2,w/2) dimensions due to pooling,
        # the new_h and new_w represent the indices of the 2x2 pool square to be scanned across the image
        for i in range(new_h):
            for j in range(new_w):
                # define the im_region as that of the original image size sliced into the (i-th,j-th) 2x2
                # portions of the original image, and access the 2x2 portions of the original image
                # via the selection of image array elements in height and width, which correspond to the pool 
                # the syntax [h1:h2, w1:w2] represent the range in height and width that define a pool square of 2x2
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]  
                yield im_region, i, j   # yields the 2x2 region of the image corresponding to the pool size, at the (i-th, j-th) coordinates of the pool

    def forward(self, input):
        '''
        Performs a forward pass of the maxpool layer using the given input.
        Returns a 2d numpy array with dimensions (h/2, w/2, num_filters)
        - input is a 3d numpy array with dimensions (h, w, num_filters)
        '''

        print('Step 2: Pooling (h,w,num_filters) -> (h/2, w/2, num_filters)')
        
        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        # loop over each 2x2 pool at the (i,j) location of the image, and extract the max value out of said 2x2 pool
        for im_region, i, j in self.iterate_regions(input):

            #extract the max value of the 2x2 pool at the (i,j) location of the image
            # axis = (0,1) -> (h, w) is specified to ONLY find the maximum along the first 2 dimension (h,w), and not a third (num_filters)
            output[i, j] = np.amax(im_region, axis=(0,1))  
            
        return output
