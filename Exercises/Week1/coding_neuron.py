import numpy as np
import matplotlib.pyplot as plt

# Machine Learning for Beginners: An Introduction to Neural Networks
# 1. Building Blocks: Neurons
# This code example has been modified from:
# https://victorzhou.com/blog/intro-to-neural-networks/

def sigmoid(x):
    
    #activation function
    return 1 / (1 + np.exp(-x))

#x = np.linspace(-10, 10, 100)
#plt.plot(x, sigmoid(x))
#plt.show()

class Neuron:
    # 1. the __init__ function is called a constructor, or initializer,
    #    and is automatically called when you create a new instance of a class
    # 2. within the __init__ function, the newly created object (i.e., obj = Neuron (...) )
    #    is assigned to the parameter self
    def __init__(self, weights, bias):
        print('Calling __init__ function . . .')
        self.weights = weights
        self.bias    = bias

        print('weights = ', weights)
        print('bias = ', bias)

    def feedforward(self, inputs):
        print('Calling feedforward . . .')
        # Weight the inputs, add bias, then use activation function
        # e.g. total = w1*x1 + w2*x2 + bias
        total = np.dot(self.weights, inputs) + self.bias

        # return a call to the activation function
        return sigmoid(total)

# define weights and bias
weights = np.array([0, 1])  # w1 = 0, w2 = 1
bias = 4                    # b = 4

# create a new instance of the class 'Neuron'        
n = Neuron(weights, bias)

# define inputs
x = np.array([2,3])         # x1 = 2, x2 = 3

print(n.feedforward(x))
