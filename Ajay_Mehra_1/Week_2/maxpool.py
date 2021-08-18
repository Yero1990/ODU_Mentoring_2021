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
        output=np.zeros((d1//2,d2//2,d3))
        for region,i,j in self.iterate_regions(input):
            output[i,j]=np.max(region,axis=(0,1))
        return output