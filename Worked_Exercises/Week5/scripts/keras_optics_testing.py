'''
Script extarcted and modified from blog by Victor Zhou:
Keras for Beginners: Implementing a Convolutional Neural Network

https://victorzhou.com/blog/keras-cnn-tutorial/

This script is adapted to analyze the actual JLab Hall C optics data
'''

import numpy as np
import matplotlib.pyplot as plt
#import mnist
import h5py   # module to load binary data format (.h5)
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
import codecs  # this module is used to decode binary strings to normal form

# user input, to either traing the neural network, or test it
analysis = 'train_data' 
#analysis = 'test_data' 

# create empty lists to append arrays of data

title = []
train_images = []
train_labels = []
train_tunes = []
history = []
loss = []
acc = []

#train_images.append()

# Open training data binary data file
f1 = h5py.File('optics_training_data.h5', 'r')

# loop over each key
for key in f1['images'].keys():

    # append all training images/labels/tunes corresponding to each key (i.e., a key is: 'xfp_vs_yfp', or 'xpfp_vs_yfp', etc.) to a list
    train_images.append( f1['images'][key][:] )
    train_labels.append( f1['labels'][key][:] ) 
    train_tunes.append(  f1['tunes'][key][:] ) 
    title.append(key)
    
# Normalize the images (easier to work with small numbers than large numbers)
for i in range(len(train_images)):

    # access the ith set of train images and normalize
    train_images[i] = (train_images[i] / 255) - 0.5

    # Reshape the images.
    train_images[i] = np.expand_dims(train_images[i], axis=3)

    print(train_images[i].shape) # (60000, 28, 28, 1)


'''
# Open testing data binary data file
f2 = h5py.File('optics_testing.h5', 'r')
    
test_images = f2['images'][:]  # mnist.train_images()
test_labels = f2['labels'][:]  # mnist.train_labels()
test_tunes  = f2['tunes'][:]   # not necessary for analysis, but helps to identify which image and tune it predicted (corresponding to a label)

# Normalize the images (easier to work with small numbers than large numbers)
test_images = (test_images / 255) - 0.5

# Reshape the images.
test_images = np.expand_dims(test_images, axis=3)

print(test_images.shape) 
'''
    

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
    Conv2D(num_filters, filter_size, input_shape=(200, 200, 1)),
    MaxPooling2D(pool_size=pool_size),
    Flatten(),
    Dense(31, activation='softmax'), 
])


if analysis == 'train_data':

    #---------------------
    # Compiling the Model
    #---------------------
    model.compile(
        optimizer="adam",
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    color_arr = ['k', 'b', 'g', 'r', 'm', 'violet']
    # loop over each of the 6 training set (set of images for xfp_vs_yfp, xpfp_vs_yfp, etc. . .)
    for i, key in enumerate(f1['images'].keys()):
                            
        #--------------------
        # Training the Model
        #--------------------
        
        ihist = model.fit(
            train_images[i],
            to_categorical(train_labels[i]),
            epochs=5,
            #validation_data=(test_images, to_categorical(test_labels)),
        )
        
        history.append(ihist)
        
        #--------------------
        # Saving the Model
        #--------------------
        
        # save optimized (trained) weights for later use
        model.save_weights('optics_weights_%s.h5' % (key))

        #-------------------------------------
        # Plot the Neural Network Performance
        #-------------------------------------
    
        # Plot the accuracy and loss vs. epochs to determine how well the network has been trained
            
        ith_loss = np.array(history[i].history['loss'])/100.
        ith_acc = np.array(history[i].history['accuracy'])

        loss.append(ith_loss)
        acc.append(ith_acc)
        
        plt.plot(acc[i], linestyle='-',   color=color_arr[i],  label='accuracy: '+title[i])
        plt.plot(loss[i], linestyle='--', color=color_arr[i],  label='loss: '+title[i])

    
    
    plt.title('Model Performance')
    plt.ylabel('accuracy, loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()

'''
elif analysis == 'test_data':

    # --------------------------------------------
    # Load the model's saved weights.
    # (assumes the weights have already been saved)
    # -------------------------------------------
    
    model.load_weights('optics_weights.h5')
    
    # Predict on the first 10 test images.
    predictions = model.predict(test_images[:10])
    
    # ---- Print our model's predictions -----
    
    # print predicted label (max output probability corresponding to each of the input test_images)
    predicted_labels = np.argmax(predictions, axis=1) 
    predicted_img    = train_images[predicted_labels]
    predicted_tunes  = train_tunes[predicted_labels] 

    true_labels      = np.arange(predicted_labels.size)  # 1st 10 images
    true_img    = test_images[true_labels]
    true_tunes       = test_tunes[true_labels]
    
    print('keras model predictions = ', predicted_labels)

    # plot precited (left plot) and trues (right plot) patterns


    for idx in range(predicted_labels.size):

        fig, ax = plt.subplots(1)

        # left plot (predicted)
        plt.subplot(1, 2, 1) 
        plt.imshow(predicted_img[idx], cmap='gray_r')
        plt.title(codecs.decode(predicted_tunes[idx]))
        ax.text(5, 5, 'your legend', bbox={'facecolor': 'white', 'pad': 10})

        # right plot (true)
        plt.subplot(1, 2, 2) 
        plt.imshow(true_img[idx], cmap='gray_r')
        plt.title(codecs.decode(true_tunes[idx]))

        plt.savefig('image_pattern_%i.png'%idx)
        
        
    #row2
    #plt.subplot(3, 2, 3) #left
    #plt.imshow(predicted_img[1], cmap='gray_r')
    #plt.title(codecs.decode(predicted_tunes[1]))

    #plt.subplot(3, 2, 4) # right
    #plt.imshow(true_img[1], cmap='gray_r')
    #plt.title(codecs.decode(true_tunes[1]))

    #row3
    #plt.subplot(3, 2, 5) #left
    #plt.imshow(predicted_img[2], cmap='gray_r')
    #plt.title(codecs.decode(predicted_tunes[2]))

    #plt.subplot(3, 2, 6) # right
    #plt.imshow(true_img[2], cmap='gray_r')
    #plt.title(codecs.decode(true_tunes[2]))

    #plt.tight_layout()
    #plt.show()
        
        
    # Check our predictions against the ground truths.
    #print('actual digits in images = 'test_labels[:5]) # [7, 2, 1, 0, 4]

    # ------------------------------------------
'''
