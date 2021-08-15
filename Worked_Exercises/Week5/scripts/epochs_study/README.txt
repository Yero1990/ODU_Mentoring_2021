This directory summarizes the studies done
to determine the depencence of the neural network
predictive power on the number of epochs.

A plot is made of the ratio: correct_predictions / (total )
versus each of the focal plane plots, for each of the epoch
configurations

This study serves as a sanity check to show that increasing the
number of epochs does indeed improve the model, however, too many
epoch can result in overfitting the model and can even have detrimental
effects.


In the keras_optics.py script, the following parameters were used:

fixed parameters:
------------------
num_filters = 8
filter_size = 3
pool_size = 2
input_image=(200,200,1)
activation='softmax'

optimizer='adam'
loss='categorical_crossentropy'
metrics=['accuracy']


variable parameters:
---------------------
epochs = 10, 50 or 100



