# Following along with Victor Zhou's tutorial series on neural networks
# https://victorzhou.com/blog/intro-to-neural-networks/

import numpy as np

def sigmoid(x):
    # Activation function: f(x) = 1 / [1 + exp(-x)]
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedForward(self, inputs):
        # Weigh inputs, add bias, then use the activation function
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

weights = np.array([0, 1])      # w_1 = 0, w_2 = 1
bias = 4                        # b = 4

n = Neuron(weights, bias)

x = np.array([2, 3])            # x_1 = 2, x_2 = 3

#print(n.feedForward(x))         # 0.9990889488055994


class OurNeuralNetwork:
    '''

    A neural network with:
        - 2 inputs
        - a hidden layer with 2 neurons (h1, h2)
        - an ouput layer with 1 neuron (o1)

    Each neuron has the same weights and bias:
        - w = [0, 1]
        - b = 0

    '''

    def __init__(self):
        weights = np.array([0, 1])
        bias = 0

        # Using the Neuron class from the previous example
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedForward(self, x):
        out_h1 = self.h1.feedForward(x)
        out_h2 = self.h2.feedForward(x)

        # The inputs for o1 are the outputs from neurons h1 and h2
        out_o1 = self.o1.feedForward(np.array([out_h1, out_h2]))

        return out_o1


# Initializing neural network
network = OurNeuralNetwork()

x = np.array([2, 3])

print(network.feedForward(x))       # Expected: 0.7216325609518421
