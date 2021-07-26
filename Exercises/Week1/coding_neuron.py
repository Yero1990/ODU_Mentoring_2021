import numpy as np
import matplotlib.pyplot as plt

# Machine Learning for Beginners: An Introduction to Neural Networks

#-----------------------------
# 1. Building Blocks: Neuron
#-----------------------------
def sigmoid(x):
    
    #activation function
    return 1 / (1 + np.exp(-x))

#x = np.linspace(-10, 10, 100)
#plt.plot(x, sigmoid(x))
#plt.show()

class Neuron:
    '''
    1. the __init__ function is called a constructor, or initializer,
        and is automatically called when you create a new instance of a class
    2. within the __init__ function, the newly created object (i.e., obj = Neuron (...) )
        is assigned to the parameter self, self is an instance of the class via which 
        attributes and methods can be initialized and accessed later on.
    '''
    def __init__(self, weights, bias):
        print('Calling Neuron() __init__ function . . .')
        self.weights = weights
        self.bias    = bias

        print('weights = ', weights)
        print('bias = ', bias)

    def feedforward(self, inputs):
        print('Calling Neuron() feedforward . . .')
        # Weight the inputs, add bias, then use activation function
        # e.g. total = w1*x1 + w2*x2 + bias
        total = np.dot(self.weights, inputs) + self.bias

        # return a call to the activation function
        return sigmoid(total)


# define inputs
x = np.array([2,3])         # x1 = 2, x2 = 3

# define weights and bias
weights = np.array([0, 1])  # w1 = 0, w2 = 1
bias = 4                    # b = 4

# create a new instance of the class 'Neuron'        
n = Neuron(weights, bias)

#print(n.feedforward(x))

#--------------------------------------------
# 2. Combining Neurons into a Neural Network
#--------------------------------------------

class OurNeuralNetwork:
    '''
    A neural network with:
      - 2 inputs
      - a hidden layer with 2 neurons (h1, h2)
      - an output layer with 1 neuron (o1)
    Each neuron has the same weights and bias:
       - w = [0,1]
      - b = 0
    '''
    
    def __init__(self):
        weights = np.array([0,1])
        bias = 0

        # Call the Neuron Class to instantiate 2 hidden layer neurons and 1 output neuron layer
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)  # output of neuron h1 layer (input of 'output' o1 layer)
        out_h2 = self.h2.feedforward(x)

        # The inputs for o1 are the outputs from h1 and h2
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1


# define instance of the neural network
network = OurNeuralNetwork()

# define the inputs to the neural network (i.e., data)
x = np.array([2,3])

print(network.feedforward(x))
