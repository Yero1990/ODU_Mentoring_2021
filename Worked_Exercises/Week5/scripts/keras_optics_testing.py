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

test_images = []
test_labels = []
test_tunes = []

history = []
loss = []
acc = []


# Open training data binary data file
f1 = h5py.File('optics_training_data.h5', 'r')

# loop over each key
for i, key in enumerate(f1['images'].keys()):

    print('i = ',i,', key =', key)
    print('sizeof f1[images][ikey] -> ',  f1['images'][key].shape)
    # append all training images/labels/tunes corresponding to each key (i.e., a key is: 'xfp_vs_yfp', or 'xpfp_vs_yfp', etc.) to a list
    train_images.append( f1['images'][key][:] )
    train_labels.append( f1['labels'][key][:] ) 
    train_tunes.append(  f1['tunes'][key][:] ) 
    title.append(key)

    # normalize train images
    train_images[i] = (train_images[i] / 255) - 0.5

    # Reshape the images.
    train_images[i] = np.expand_dims(train_images[i], axis=3)


# Open testing data binary data file
f2 = h5py.File('optics_testing_data.h5', 'r')
    
# loop over each key
for i, key in enumerate(f2['images'].keys()):

    print('i = ',i,', key =', key)

    # append all training images/labels/tunes corresponding to each key (i.e., a key is: 'xfp_vs_yfp', or 'xpfp_vs_yfp', etc.) to a list
    test_images.append( f2['images'][key][:] )
    test_labels.append( f2['labels'][key][:] ) 
    test_tunes.append(  f2['tunes'][key][:] ) 

    # normalize test images
    test_images[i] = (test_images[i] / 255) - 0.5

    # Reshape the images.
    test_images[i] = np.expand_dims(test_images[i], axis=3)
    
    print('test_images_shape = ', test_images[i].shape) # (60000, 28, 28, 1)
    

#--------------------------
# Building the Model
# (Using Sequential Class)
#--------------------------
'''
Every Keras model is either built using the Sequential class, which represents a 
linear stack of layers, or the functional Model class, which is more customizeable. 
We’ll be using the simpler Sequential model, since our CNN will be a linear stack of layers.
'''


num_filters = 12   #optimum
filter_size = 6    #optimum
pool_size   = 6    #optimum

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
        optimizer="adam",    # adam, RMSprop, SGD, Nadam, Adamax ('adam' is the best optimizer for these studies)
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    color_arr = ['k', 'b', 'g', 'r', 'm', 'violet']
    
    # loop over each of the 6 training sets (set of images for xfp_vs_yfp, xpfp_vs_yfp, etc. . .)
    for i, key in enumerate(f1['images'].keys()):
                            
        #--------------------
        # Training the Model
        #--------------------
        
        ihist = model.fit(
            train_images[i],
            to_categorical(train_labels[i]),
            epochs=100,
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


elif analysis == 'test_data':

    # --------------------------------------------
    # Load the model's saved weights.
    # (assumes the weights have already been saved)
    # -------------------------------------------


    # loop over each key type of images (i.e., xfp_vs_yfp, etc. . . )
    for i, key in enumerate(f2['images'].keys()):

        # load the optimized weights for each key
        model.load_weights('optics_weights_%s.h5' % (key))
    
        # Predict labels of all test images per ith key
        predictions = model.predict(test_images[i])
    
        # ---- Print our model's predictions -----
    
        # print predicted label (max output probability corresponding to each of the input test_images)
        predicted_labels = np.argmax(predictions, axis=1) 
        predicted_img    = train_images[i][predicted_labels]
        predicted_tunes  = train_tunes[i][predicted_labels] 
        
        true_labels      = np.arange(predicted_labels.size)
        true_img         = test_images[i][true_labels]
        true_tunes       = test_tunes[i][true_labels]
    
        print('%s keras model predictions' % (key))
        print('============================')
        print('predicted_labels = ', predicted_labels)
        print('predicted_images_shape = ', predicted_img.shape)
        print('predicted_tunes_shape = ', predicted_tunes.shape)
        
        print('----------------------------')
        print('true_labels = ', true_labels)
        print('true_images_shape = ', true_img.shape)
        print('true_tunes_shape = ', true_tunes.shape)

        fig, ax = plt.subplots(figsize=(12,12))
        plt.subplots_adjust(left=0.01, bottom=0.025, right=0.99, top=0.95, wspace=0, hspace=0.4)
        # create subplot figure
        for idx in range(predicted_labels.size):

            
            #  idx   npad_odd = 2*(idx+1) - 1       npad_even = 2*idx + 2
            #  0              1                     2
            #  1              3                     4
            #  2              5                     6
            #  3              7                     8

            # define pad numbering per idx to be: (1,2), (3,4), (5,6), . . . etc.  --> (npad_odd, npad_even) --> (predicted, true)
            npad_odd = 2*(idx+1) - 1
            npad_even = (2*idx) + 2
            plt.suptitle(title[i])
            # left plot (predicted)
            plt.subplot(5, 4, npad_odd) 
            plt.imshow(predicted_img[idx], cmap='gray_r')
            plt.title(codecs.decode(predicted_tunes[idx]), fontsize=8)
            plt.plot([], color='k', marker='', label='predicted')
            plt.legend()

            # right plot (true)
            plt.subplot(5, 4, npad_even) 
            plt.imshow(true_img[idx], cmap='gray_r')
            plt.title(codecs.decode(true_tunes[idx]), fontsize=8)
            plt.plot([], color='k', marker='', label='true')
            plt.legend()

        plt.savefig('final_results_%s.png'%(key)) # change the resolution of the saved image    
        #plt.show()
        

