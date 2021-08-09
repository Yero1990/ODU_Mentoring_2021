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

print(n.feedforward(x))         # 0.9990889488055994
