'''
script to test how to save image 
data arrays to file format h5py
'''

import numpy as np
import h5py
import matplotlib.pyplot as plt


a = np.random.random(size=(100,20))
b = np.random.random(size=(100,20))

h5f = h5py.File('data.h5', 'w')
h5f.create_dataset('label_1', data=a)
h5f.create_dataset('label_2', data=b)

h5f.close()

#----------

# load binary data

h5f = h5py.File('data.h5','r')
h5f.keys()  # this will show the labels of each stored image

# access the labeled image as follows:
h5f['label_1']  # access the 1s5 5 pixels of the image on heigh and width

plt.imshow(h5f['label_1'], cmap='gray_r')

plt.show()
