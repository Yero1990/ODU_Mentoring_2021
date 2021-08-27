import numpy as np
import matplotlib.pyplot as plt
import mnist
from conv_week2 import Conv3x3
from maxpool_week2 import MaxPool2
from softmax_week2 import Softmax

'''
This is the steering script to runs the various layers
of the CNNs. This script uses the methods from each of the
imported hidden layer classes, Conv3x3, MasPool2 and Softmax
'''

# The mnist package handles the MNIST dataset for us!
# Learn more at https://github.com/datapythonista/mnist

# Read training images (used for training the neural network)
train_images = mnist.train_images()[:1000]
train_labels = mnist.train_labels()[:1000]

# Read test images (used for testing the neural network accuray in making predictions)
# We only use the first 1k testing examples (out of 10k total)
# in the interest of time. Feel free to change this if you want.

test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]


# Initialize each of the hidden layers in the Neural Network
conv = Conv3x3(8)                  # 28x28x1 -> 26x26x8  (convolution layer)
pool = MaxPool2()                  # 26x26x8 -> 13x13x8  (pooling layer)
softmax = Softmax(13*13*8, 10)     # 13x13x8 -> 10 (softmax activation function layer)

def forward(image, label):
    '''
    Completes a forward pass of the CNN and calculates the accuray and cross-entropy loss.
    - image is a 2d numpy array
    - label is a digit
    '''

    # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier to
    # work with. This is standard practice.
    out = conv.forward((image/255) - 0.5)
    out = pool.forward(out)
    out = softmax.forward(out)    # size-10 1d-array of probabilities outputed by softmax
    
    # calculate cross-entropy loss and accuracy. np.log() is the natural log
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0   #accuracy is 1 if (argmax returns index of maximum argument,, i.e., highest probability)
                                                #if the index of the max prob. matches the label, then the prediction is correct (100 accuracy)
    return out, loss, acc

def train(im, label, lr=.005):
    '''
    Completes a full training step on the given image and label.
    Returns the cross-entropy loss and accuracy.
    - image is a 2d numpy array
    - label is a digit
    - lr is the learning rate
    '''

    # Forward propagation
    out, loss, acc = forward(im, label)

    # Calculate initial gradient (Loss = -ln( out ), dLoss / dout = -1/out )
    # out is a probability function outputted by softmax
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    # Backward propagation
    gradient = softmax.backprop(gradient, lr)  # 10 -> 13 x 13 x 8
    gradient = pool.backprop(gradient)         # 13 x 13 x 8 -> 26 x 26 x 8
    gradient = conv.backprop(gradient, lr)     # 26 x 26 x 8
    
    return loss, acc
    
print('MNIST CNN initialized!')

# Train the CNN for 3 epochs (loop over each batch of training data images)
for epoch in range(3):
    print('--- Epoch %d ---'% (epoch+1))

    # Shuffle the training data
    permutation = np.random.permutation(len(train_images))  # this changes the ordering of the indices of the training images
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]
    
    
    # Train
    loss = 0
    num_correct = 0

    # loop over the test images, and for each test image, check how well the CNN does in predictin the image
    # IMPORTANT: RECALL, this is without having trained the netwrk yet. We are making predictions based on ONLY the
    # randomly-initialized parameters

    for i, (im, label) in enumerate(zip(train_images, train_labels)):   # zip returns a zip object, i.e. a paired object (ith-test_image, ith-test_labels)
        if i % 100 == 99: # check Loss and accuracy every 100 steps (images), then reset counters. num_correct is the number of correct predictions out of every 100 images,
                          # calculated every 100 images, then naturally, num_correct represents the accuracy of the neural network.
                          # the modulo (%) operator of  x % y returns the remainder of the division, when x=y, 2y, etc, it returns 0 (resets)
            print('[Step %d] Past 100 steps: Average Loss %.3f | Accuracy = %.2f %%' % (i + 1, loss / 100, num_correct) )

            # Reset loss and num_correct counters at the end of every multiple of 100th step
            loss = 0
            num_correct = 0

            # For every ith image (im), train the network, and increment the loss
            # and the number of times the neural network made correct prediction (up to 100 images before resetting)

        # even though loss and num_correct are reset every 100 steps, the training of the images is NOT, and therefore, constantly improves the accuray with every image trained
        l, acc = train(im, label)
        loss += l
        num_correct += acc


# Test the CNN
print('\n---- Testing the CNN ---- ')
loss = 0
num_correct = 0

# loop over each test image
for im, label in zip(test_images, test_labels):
    out, l, acc = forward(im, label)
    loss += l
    num_correct += acc   # increment number of correct predictions made by the neural network

    print('predicted digit = ', np.argmax(out) )
    print('real digit = ', label)
    plt.imshow(-1.*im, cmap='gray')
    plt.show()
    
num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)

































