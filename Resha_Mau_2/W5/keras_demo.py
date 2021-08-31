
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
import numpy as np
import mnist    # import the MNIST dataset
from tensorflow import keras    # import the Keras package to use
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalize the images
# NOTE: Normalizing the image pixel values from [0, 255] to [-0.5, 0.5] makes it easier to train the network
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Reshape the images
# NOTE: It is necessary to reshape each image from (28, 28) to (28, 28, 1) because Keras requires the third dimension
train_images = np.expand_dims(train_images, axis = 3)
test_images = np.expand_dims(test_images, axis = 3)

#print(train_images.shape) # (60000, 28, 28, 1)
#print(test_images.shape)  # (10000, 28, 28, 1)

# NOTE: Every Keras model can be built using either the Sequential class or the Model class.
#   - Sequential class: groups a linear stack of layers into a tf.keras.Model
#   - Model class: groups layers into an object with training and inference features
#   - The Sequential model will be used because the CNN we're working with will just be a linear stack of layers.

num_filters = 8
filter_size = 3
pool_size = 2

model = Sequential([
Conv2D(num_filters, filter_size, input_shape=(28,28,1)),
MaxPooling2D(pool_size = pool_size),
Flatten(),
Dense(10, activation = 'softmax'),
])

# Training a model in Keras involves only invoking the method fit() and putting in some parameters.
# Parameters that will be supplied to our demo example:
#       - training data: images and labels, X and Y
#       - number of epochs: number of iterations over the entire dataset
#       - test data: used to evaluate the neural network's performance

# Compiling the model
model.compile(
    'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'],
)

# Training the model
model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=3,
  validation_data=(test_images, to_categorical(test_labels)),
)

# Save the trained model to the hard drive for future use
model.save_weights('cnn.h5')

# To use the trained model to make predictions, pass an array of inputs into the method "predict()".
# Passing these inputs as parameters into the method predict() will return an array of outputs.
# Output: 10 probabilities (use np.argmax() to convert them to digits)

# Make predictions on the first 5 test images
predictions = model.predict(test_images[:5])

# Print the model's predictions on those first 5 images
print(np.argmax(predictions, axis = 1))     # [7, 2, 1, 0, 4]

# Check predictions
print(test_labels[:5])      # [7, 2, 1, 0, 4]
