import numpy as np
import mnist
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical




train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

#print(train_images.shape)# (60000, 28, 28)
#print(test_images.shape) #  (10000, 28, 28)

# Normalize the images
train_images = (train_images/255) - 0.5
test_images = (test_images/255) - 0.5

# Reshape the images
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

#print(train_images.shape) #(60000, 28, 28, 1)
#print(test_images.shape)  #(10000, 28, 28, 1)

# Building the model

''' Every Keras model is either built using the Sequential class (which represents a linear stack of layers) or the functional Model class(which is more customizeable). We'll be using the simpler Sequential model, since our CNN will be a linear stack of layers '''

num_filters = 8
filter_size = 3
pool_size = 2

''' The first layer in any Sequential model must specify the input_shape, so we do son on Conv2D. Once this input is specified, Keras will automatically
    infer the shapes of inputs for later layers. '''

model = Sequential([
    Conv2D(num_filters, filter_size, input_shape = (28, 28, 1)), # num_filters, pool_size, filter_size are our hyperparameters
    MaxPooling2D(pool_size = pool_size),
    Flatten(),
    Dense(10, activation = 'softmax'),
    ]) # The output Softmax layer has 10 nodes, one for each class


# Compiling the Model
model.compile(
    'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'], 
    )

# Training the Model

''' There’s one thing we have to be careful about: Keras expects the training targets to be 10-dimensional vectors, since there are 10 nodes in our 
    Softmax output layer. Right now, our train_labels and test_labels arrays contain single integers representing the class for each image

    Conveniently, Keras has a utility method that fixes this exact issue: to_categorical. It turns our array of class integers into an array of
    one-hot vectors instead. For example, 2 would become [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] (it’s zero-indexed) '''

model.fit(
    train_images,
    to_categorical(train_labels),
    epochs = 3,
    validation_data = (test_images, to_categorical(test_labels)),
    )

#Now that we have a working, trained model, let’s put it to use. The first thing we’ll do is save it to disk so we can load it back up anytime

model.save_weights('cnn.h5')






