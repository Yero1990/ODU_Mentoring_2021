# This class soft_max takes input of (number of elements in new maxpooled img "13*13*8=1352") 
# and (number of nodes 10) and forward function takes input of the img array
# and returns a array 10 with probablities for each one.
import numpy as np
class soft_max:
    def __init__(self,input_len,num_nodes):
        self.weights=np.random.randn(input_len,num_nodes)/input_len
        self.biases=np.zeros(num_nodes)
    def forward(self,input):
        input=input.flatten()
        #calculaing wX+b
        feed=np.dot(input,self.weights)+self.biases
        #final output
        expx=np.exp(feed)
        output=expx/np.sum(expx,axis=0)
        return output