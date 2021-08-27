from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import mnist

###***
test_images=mnist.test_images()
test_images=np.expand_dims((test_images/255)-0.5,axis=3)
test_labels=mnist.test_labels()

### we skipped part 3 as we exported the weights

num_filters=8
filter_size=3
pool_size=2

model=Sequential([
    Conv2D(num_filters,filter_size,input_shape=(28,28,1)),
    MaxPooling2D(pool_size=pool_size),
    Flatten(),
    Dense(10,activation='softmax'),
])

n=10000
model.load_weights('cnn.h5')
prediction = model.predict(test_images[:n])
max_pred=np.argmax(prediction,axis=1)

DIFF=(max_pred)-test_labels[:n]
win=len(DIFF[DIFF==0])
print("total_acuracy=",(100*win)/n,"%")

for i in np.random.randint(1,6000,20):
    plt.imshow(test_images[i])
    print(np.argmax(prediction,axis=1)[i],"\n\n\n")
    plt.show()