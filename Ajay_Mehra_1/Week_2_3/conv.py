import numpy as np
class conv3x3:
    def __init__(self,num_filters):
        self.d3=num_filters
        self.filters_8=np.random.rand(num_filters,3,3)

    def iterate_regions(self,image):
        d1,d2=image.shape
        for i in range(d1-2):
            for j in range(d2-2):
                region=image[i:(i+3),j:(j+3)]
                yield region,i,j

    def forward(self,input):
        d1,d2=input.shape
        output=np.zeros((d1-2,d2-2,self.d3))
        for region,i,j in self.iterate_regions(input):
            output[i,j]=np.sum(region*self.filters_8,axis=(1,2))
        return output