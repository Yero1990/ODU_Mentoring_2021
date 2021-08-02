import numpy as np
import matplotlib.pyplot as plt
import mnist

#impor the classes from the various hidden layers (where most of the work is done)
from conv import Conv3x3
from maxpool import MaxPool2

#select which training image to view
img_num = 4

# The mnist package handles the MNIST dataset for us!
# Learn more at https://github.com/datapythonista/mnist
train_images = mnist.train_images()
train_labels = mnist.train_labels()

# print shape of 1st training image
# print('train_images[img_num] shape = ', train_images[img_num].shape)

# plot train_images[img_num] to see how it looks like
# plt.imshow(train_images[img_num])
# plt.show()


# instantiate our Convolusion 3x3 class, and initiate it with 8 3x3 filters
conv = Conv3x3(8)
# instantiate our 2x2 Pooling class
pool = MaxPool2()

#pass a training image  to the forward method in the Conv3x3 class 
output = conv.forward(train_images[img_num])
print(output.shape) # shape of the output image is (26, 26, 8)

# forward pass the convolved (h,w,num_filters) image given by the above output into the pooling class
# the new output is:
output = pool.forward(output)
print(output.shape) # shape of the output image is (13, 13, 8)


