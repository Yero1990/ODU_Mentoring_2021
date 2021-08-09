## Neural Network Applications Project

**Description:** <br>
This project consists of using actual JLab Experimental Hall C available data on spectrometer Optics, and utilizing it to train a convolutional neural network (CNN) to recognize specific patterns. The using the trained (optimized) parameters to  reognize test Optics images and determine with what accuracy is the CNN able to make predictions about the known pattern

**Pre-requisites:** <br>
Since these optics data exist in ROOTfiles, they must be extracted into an array-like format, similar to how MNIST stores training and test images (e.g. m x m pixels). An array of labels must also be created to identify the corresponding images, similar to the training and test labels used in MNIST. 

**Directory Structure:**<br>

`./ROOTfiles`: ROOTfiles containing the raw data. The files with the "_hist.root" extension contain the relevant histogram objects with the 2D spectrometer focal-plane correlations. These are the histogram objects which have to be converted to a 2D numpy arrayto be given as input in the neural network. <br>

`./scripts/make_2Doptics.C`: This ROOT C++ script takes a raw data ROOTfile as input and make six 2D correlation histogram objects which are stored the files with the "_hist.root" extension. <br>

`./scripts/save2binary.py`: This python script does the following: 1) reads the histogram objects for each "_hist.root" file, 2) converts the histogram objects to a 2D numpy array, and 3) saves the 2D numpy arrays, and the corresponding image labels to a binary file format (.h5), either under the name *optics\_training.h5* or *optics\_test.h5*, depending on whether the data will be used for training or testing the Keras CNN model. <br>

`./scripts/keras_optics.py`: This python script reads the .h5 binary files, trains the neural network and saves the optimized weights under the name *optics\_weights.h5*, so that these may be used when testing the model with new images/patterns. The script also make plots of the loss and accuray as a function of the number of epochs after the training, so that the learning curve of the neural network can actually be monitored. 