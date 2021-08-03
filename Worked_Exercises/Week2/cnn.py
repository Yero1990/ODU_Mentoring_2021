import numpy as np
import matplotlib.pyplot as plt
import mnist
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax

#select which training image to view
img_num = 4

# The mnist package handles the MNIST dataset for us!
# Learn more at https://github.com/datapythonista/mnist

# Read training images (used for training the neural network)
#train_images = mnist.train_images()
#train_labels = mnist.train_labels()

# Read test images (used for testing the neural network accuray in making predictions)
# We only use the first 1k testing examples (out of 10k total)
# in the interest of time. Feel free to change this if you want.
test_images = mnist.test_images()[:1]
test_labels = mnist.test_labels()[:1]


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
    out = softmax.forward(out)
    
    # calculate cross-entropy loss and accuracy. np.log() is the natural log
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0   #accuracy is 1 if (argmax returns index of maximum argument,, i.e., highest probability)
                                                #if the index of the max prob. matches the label, then the prediction is correct (100 accuracy)
    return out, loss, acc

print('MNIST CNN initialized!')

loss = 0
num_correct = 0

# loop over the test images, and for each test image, check how well the CNN does in predictin the image
# IMPORTANT: RECALL, this is without having trained the netwrk yet. We are making predictions based on ONLY the
# randomly-initialized parameters
for i, (im, label) in enumerate(zip(test_images, test_labels)):   # zip returns a zip object, i.e. a paired object (ith-test_image, ith-test_labels)


    # --------- testing / checks (C. Yero) -------------
    print('i = ', i)
    #print('im_shape, label = ', im.shape,', ',label)
    #print('im matrix = ', -1*im)

    # plot image of number (in black & white)
    plt.imshow(-1*im, cmap='gray')
    plt.show()

    # pass the image and label to the CNN
    iout, iloss, iacc = forward(im, label)

    print('iout = ', iout)   # flatten array of 10 outputs, each representing the probablity of the digit
    print('index w/ highest prob. = ', np.argmax(iout), ' predicted index = ', label)
        
    #print('iloss = ', iloss)
    #print('iacc = ', iacc)   # if acc =1 , the prediciton is correct, if acc = 0, the prediction is NOT correct

    loss += iloss
    num_correct += iacc
    print('loss_sum = ', loss)
    print('acc_sum = ', num_correct/100. )


    # My Comments: At this stage, since the CNN has NOT been trained, the predictions are made
    # based on the random numbers generated for the parameters/weights in each of the layers.
    # Once would need to train the CNN to optimize the parameters (minimize the loss), and improve
    # the predictive power of the CNN
    
    # ==================================================
    
    
    # ----- Forward Pass as done in blog -----
    # Do a forward pass.
    #_, l, acc = forward(im, label)
    #loss += l   # calculate the loss: loss += l is equivalent to : loss = loss + l (each time it gets updated with + l)
    #num_correct += acc

    # Print stats every 100 steps.
    #if i % 100 == 99:
    #    print('[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' % (i + 1, loss / 100, num_correct) )


    #    loss = 0
    #    num_correct = 0
