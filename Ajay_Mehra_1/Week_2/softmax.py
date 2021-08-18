import numpy as np
class soft_max:
    def __init__(self,input_len,num_nodes):
        self.weights=np.random.randn(input_len,num_nodes)/input_len
        self.biases=np.zeros(num_nodes)
    def forward(self,input):
        input=input.flatten()
        input_len,num_nodes=self.weights.shape
        feed=np.dot(input,self.weights)+self.biases
        expx=np.exp(feed)
        output=expx/np.sum(expx,axis=0)
        return output