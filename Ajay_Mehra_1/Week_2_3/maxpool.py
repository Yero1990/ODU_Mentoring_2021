#Maxpool's function forward takes 26*26*8 input and do maxpool and return 13*13*8 array.
import numpy as np
class max_pool:
    def iterate_regions(self,image):
        d1,d2,d3=image.shape
        for i in range(d1//2):
            for j in range(d2//2):
                region=image[2*i:2*(i+1),2*j:2*(j+1)]
                yield region,i,j
    def forward(self,input):
        d1,d2,d3=input.shape
        self.last_input = input
        output=np.zeros((d1//2,d2//2,d3))
        for region,i,j in self.iterate_regions(input):
            output[i,j]=np.max(region,axis=(0,1))
        return output
    def backprop(self, d_L_d_out):
        d_L_d_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))

        for i2 in range(h):
            for j2 in range(w):
                for f2 in range(f):
                    # If this pixel was the max value, copy the gradient to it.
                    if im_region[i2, j2, f2] == amax[f2]:
                        d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

        return d_L_d_input