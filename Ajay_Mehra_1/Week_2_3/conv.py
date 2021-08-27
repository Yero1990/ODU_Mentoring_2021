# Conv.py takes input number of filters and its function forward takes the image array as input and 
# overall returns 3d array with filters applied,
# input : 8   28*28 array
# Output: 26*26*8 array
import numpy as np
class conv3x3:
    def __init__(self,num_filters):
        self.num_filters=num_filters
        self.filters=np.random.rand(num_filters,3,3)

    def iterate_regions(self,image):
        d1,d2=image.shape
        for i in range(d1-2):
            for j in range(d2-2):
                region=image[i:(i+3),j:(j+3)]
                yield region,i,j

    def forward(self,input):
        d1,d2=input.shape
        self.last_input=input
        output=np.zeros((d1-2,d2-2,self.num_filters))
        for region,i,j in self.iterate_regions(input):
            output[i,j]=np.sum(region*self.filters,axis=(1,2))
        return output
    
    def backprop(self, d_L_d_out, learn_rate):

        d_L_d_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        # Update filters
        self.filters -= learn_rate * d_L_d_filters

        return None