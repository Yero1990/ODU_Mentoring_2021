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
        #print('Calling Neuron() __init__ function . . .')
        self.weights = weights
        self.bias    = bias

        #print('weights = ', weights)
        #print('bias = ', bias)

    def feedforward(self, inputs):
        #print('Calling Neuron() feedforward . . .')
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

        # Initialize an instance of the Neuron Class for 2 hidden layers and 1 output layer
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

#print(network.feedforward(x))


#--------------------------------------------
# 3. Training a Neural Network, Part 1
#--------------------------------------------

'''
 Say we have the following measurements we can train to predict someone's gender given their weight/height

  Name      Weight(lb)   Height(in)    Gender
-------------------------------------------------
  Alice     133          65            F
  Bob       160          72            M
  Charlie   152          70            M
  Diana     120          60            F


Definitions
Loss: Before we train our network, we first need a way to quantify how "good" it's doing
so that it can try to do "better". That's what loss is.

mean squared error (MSE) loss:

  MSE = 1 / n * SUM_(i=1)->(i=n) ( y_true - y_pred)^2

n      : number of samples, which is 4 (Alice, Bob, Charlie, Diana)
y      : represents the variable being predicted, which is Gender in this case
y_true : is the true value of the variable (the "correct answer"), for example, y_true for Alice would be 1
y_pred : is the predicted value of the variable. It's whatever our networks outputs.

(y_true - y_pred)^2 is the squared error, and the Loss function is the average over all squared errors

Better predictions by Neural Network = Lower loss

Training a network = trying to minimize its loss 

'''

# put the mean squared error into code
def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length
    mse = ((y_true - y_pred)**2).mean()
    return mse

y_true = np.array([1, 0, 0, 1])
y_pred = np.array([0, 0, 0, 0])

# print(mse_loss(y_true, y_pred)) # 0.5


#--------------------------------------------
# 4. Training a Neural Network, Part 2
#--------------------------------------------

'''
 Goal: minimize the loss of the neural network (i.e., improve its predictive power)
 This can be done by changing the network's weights and biases to influence its predictions,
 but how do we do so in a way that decreases loss? Use multivariate calculus to minimize the loss

 L (weights, bias) :  take partial derivatives of loss with respect to weights, biases and set to zero to
 determine which combination of (weights, biases) minimizes the loss, i.e., parameter optimization. An example
 of the math is done on the actual blog by Victor Zhou.

 To optimize the loss function, the "Stochaistic Gradient Descent" method is used, which tells us how
 to change the weights and biases to minimize loss, via the simple replacement of the current weight by:

 w --> w - n * dL/dw, where n is a constant called the learning rate. All we're doing is subtracting n * dL/dw from w:

 1. If dL/dw > 0, w will decrease, which makes L decrease
 2. If dL/dw < 0, w will increase, which makes L decrease

 This process of subtracting n * dL/dw from w is done iteratively such that the loss will eventually approximate zero,
 which is directly related to the success of the neural network training





'''
