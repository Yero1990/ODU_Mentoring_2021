'''
Neural Network: Activation function, Softmax

Description: To complete our CNN, we need to give it the ability to actually 
make predictions. We’ll do that by using the standard final layer for a 
multiclass classification problem: the Softmax layer, a fully-connected (dense)
layer that uses the Softmax function as its activation. 

What softmax really does is help us quantify how sure we are of our prediction,
which is useful when training and evaluating our CNN. More specifically, using 
softmax lets us use cross-entropy loss, which takes into account how sure we 
are of each prediction.

'''

import numpy as np

class Softmax:
    # A standard fully-connected layer with softmax activation

    def __init__(self, input_len, nodes):
        # We divide by input_len to reduce the variance of our initial values
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        '''
        Performs a forward pass of the softmax layer using the given input.
        Returns a 1d numpy array containing the respective probability values.
        - input can be any array with any dimensions
        '''

        input = input.flatten()
        
        input_len, nodes = self.weights.shape

        totals = np.dot(input, self.weights) + self.biases

        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)
        
