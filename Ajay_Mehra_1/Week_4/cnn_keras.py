import numpy as np
import mnist
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images=mnist.test_images()
test_labels=mnist.test_labels()

#Normalising the images
test_images=(test_images/255)-0.5
train_images=(train_images/255)-0.5

#Reshaping the each images from 28*28 to 28*28*1
#expand_dims converts axis 3 [0,0,0...] to [[0],[0],[0]...]
train_images=np.expand_dims(train_images,axis=3)
test_images=np.expand_dims(test_images,axis=3)

num_filters = 8
filter_size = 3
pool_size = 2

###1: Building model

#we choose Sequential model
#Conv2D > 28*28 to 26*26*8
#Maxpooling > 26*26*8 to 13*13*8
#flatten + Softmax as activation function > 13*13*8 to 1352 to 10
model=Sequential([
    Conv2D(num_filters,filter_size,input_shape=(28,28,1)),
    MaxPooling2D(pool_size=pool_size),
    Flatten(),
    Dense(10, activation="softmax"),
])

###2: configuring training process

#optimizer: adam based
#loss function: c_c
#metrics: what we want*
model.compile(
    'adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

###3: Training model

# *we need[0,0,0,4,0,0,0,0,0,0] instead of [4] from labels
model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=10,
    validation_data=(test_images, to_categorical(test_labels)),
)

### 4: Saving weights

model.save_weights('cnn.h5')

predictions = model.predict(test_images[:5])
print(np.argmax(predictions, axis=1))
print(test_labels[:5])