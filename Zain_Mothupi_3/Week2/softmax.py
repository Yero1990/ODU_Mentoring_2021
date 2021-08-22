import numpy as np

class Softmax:
    # A standard fully-connected layer with softmax activation


    def __init__(self, input_len, nodes):
        # we divide by input_len to reduce the variance of our initial values

        self.weights = np.random.randn(input_len, nodes)/ input_len

        self.biases = np.zeros(nodes)


    def forward(self, input):
        ''' Performs a forward pass of the softmax layer using the given input.
            Returns a 1d numpy array containing the respective probabillity values.
            - input can be an array with any dimensions '''

        input = input.flatten() # we flatten the input to make it easier to work with since we no longer need its shape

        input_len, nodes = self.weights.shape

        totals = np.dot(input, self.weights) + self.biases
        exp = np.exp(totals)
        return exp/np.sum(exp, axis=0)
