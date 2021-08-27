# This class soft_max takes input of (number of elements in new maxpooled img "13*13*8=1352") 
# and (number of nodes 10) and forward function takes input of the img array
# and returns a array 10 with probablities for each one.
import numpy as np
class soft_max:
    def __init__(self,input_len,num_nodes):
        self.weights=np.random.randn(input_len,num_nodes)/input_len
        self.biases=np.zeros(num_nodes)
    def forward(self,input):
        self.last_input_shape = input.shape
        input=input.flatten()
        self.last_input = input
        #calculaing wX+b
        totals=np.dot(input,self.weights)+self.biases
        self.last_totals=totals
        #final output
        expx=np.exp(totals)
        output=expx/np.sum(expx,axis=0)
        return output
    def backprop(self,d_L_d_out,learn_rate):
        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:
                continue

            # e^totals
            t_exp = np.exp(self.last_totals)

            # Sum of all e^totals
            S = np.sum(t_exp)

            # Gradients of out[i] against totals
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            # Gradients of totals against weights/biases/input
            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights

            # Gradients of loss against totals
            d_L_d_t = gradient * d_out_d_t

            # Gradients of loss against weights/biases/input
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t * d_t_d_b
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            # Update weights / biases
            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b
            return d_L_d_inputs.reshape(self.last_input_shape)
