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

    def __init__(self,)
