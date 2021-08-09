'''
Script extarcted from blog by Victor Zhou:
Keras for Beginners: Implementing a Convolutional Neural Network

https://victorzhou.com/blog/keras-cnn-tutorial/

'''

import numpy as np
import mnist
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical

# The first time you run this might be a bit slow, since the
# mnist package has to download and cache the data.
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

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

# Load the model's saved weights. (assumes this code has already been run and weights saved to cnn.h5)
model.load_weights('cnn.h5')

# Predict on the first 5 test images.
predictions = model.predict(test_images[:5])

# Print our model's predictions.
print('keras model predictions = ', np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]

# Check our predictions against the ground truths.
print('actual digits in images = 'test_labels[:5]) # [7, 2, 1, 0, 4]

'''
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

model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=3,
    validation_data=(test_images, to_categorical(test_labels)),
)

#--------------------
# Using the Model
#--------------------

# save optimized (trained) weights for later use
model.save_weights('cnn.h5')

'''
