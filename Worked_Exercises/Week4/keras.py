'''
Script extarcted from blog by Victor Zhou:
Keras for Beginners: Implementing a Convolutional Neural Network

https://victorzhou.com/blog/keras-cnn-tutorial/

'''

import numpy as np
import mnist
import matplotlib.pyplot as plt
import idx2numpy  #use this incase loading images with mnist does not work ( I have had HTTP Error 503 using mnist before)
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical


# User Input (train or test model)
analysis = 'train'  # 'train' or 'test'


# The first time you run this might be a bit slow, since the
# mnist package has to download and cache the data.

# load images/labels directly from  the MNIST website using the mnist module (may give HTTP Error )
'''
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()
'''

# load test images in .idx format (this would require you to have downloaded the mnist files locally, from http://yann.lecun.com/exdb/mnist/)
# Make sure to change the path to the .idx files accordingly once you have downloaded these files
train_images = idx2numpy.convert_from_file('/Users/nuclear/MNIST/train-images.idx') 
train_labels = idx2numpy.convert_from_file('/Users/nuclear/MNIST/train-labels.idx') 

test_images = idx2numpy.convert_from_file('/Users/nuclear/MNIST/test-images.idx') 
test_labels = idx2numpy.convert_from_file('/Users/nuclear/MNIST/test-labels.idx') 


#print(train_images.shape) # (60000, 28, 28)
#print(train_labels.shape) # (60000,)

# Normalize the images (easier to work with small numbers than large numbers)
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Reshape the images.
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

print(train_images.shape) # (60000, 28, 28, 1)
print(test_images.shape)  # (10000, 28, 28, 1)


#--------------------------
# Building the Model
# (Using Sequential Class)
#--------------------------
'''
Every Keras model is either built using the Sequential class, which represents a 
linear stack of layers, or the functional Model class, which is more customizeable. 
We’ll be using the simpler Sequential model, since our CNN will be a linear stack of layers.
'''



num_filters = 8
filter_size = 3
pool_size   = 2

model = Sequential([
    Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=pool_size),
    Flatten(),
    Dense(10, activation='softmax'),
])


if analysis=='test':
    
    # Load the model's saved weights. (assumes this code has already been run and weights saved to cnn.h5)
    model.load_weights('cnn.h5')
    
    # Predict on the first 5 test images.
    predictions = model.predict(test_images[:5])
    pred_labels = np.argmax(predictions, axis=1)  # max probability for each image represents the model's prediction

    # Print our model's predictions.
    print('keras model predictions = ', pred_labels) # [7, 2, 1, 0, 4]
    
    # Check our predictions against the ground truths.
    print('actual digits in images = ', test_labels[:5]) # [7, 2, 1, 0, 4]
    
    


if(analysis=='train'):

    #---------------------
    # Compiling the Model
    #---------------------
    model.compile(
        optimizer="adam",
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    #--------------------
    # Training the Model
    #--------------------
    
    history = model.fit(
        train_images,
        to_categorical(train_labels),
        epochs=5,
        validation_data=(test_images, to_categorical(test_labels)),
    )

    #--------------------
    # Saving the Model
    #--------------------
    
    # save optimized (trained) weights for later use
    model.save_weights('cnn.h5')

    #-------------------------------------
    # Plot the Neural Network Performance
    #-------------------------------------
    
    # Plot the accuracy and loss vs. epochs to determine how well the network has been trained
    loss = np.array(history.history['loss'])
    acc = np.array(history.history['accuracy'])

    # validated accuracy and loss using validation data (actual test data used at the end of each epoch) which allows us to monitor our model’s
    # progress over time during training, which can be useful to identify overfitting and even support early stopping.
    val_loss = np.array(history.history['val_loss'])
    val_acc = np.array(history.history['val_accuracy'])

    plt.subplot(121)
    plt.plot(acc, linestyle='-',   color='g',  label='model accuracy')
    plt.plot(val_acc, linestyle='-',   color='b',  label='validation accuracy')

    plt.subplot(122)
    plt.plot(loss, linestyle='--', color='g',  label='model loss')
    plt.plot(val_loss, linestyle='--', color='b',  label='validation loss')

    plt.legend()
    plt.show()

