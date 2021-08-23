## Neural Network Applications Project

**Description:** <br>
This project consists of utilizing actual JLab Experimental Hall C simulated data on spectrometer optics to train a convolutional neural network (CNN) to recognize specific patterns. Then using the trained (optimized) parameters, a new set of test images is used to determine how well can the trained CNN model predict the tunes of the optics pattern given.

**Pre-requisites:** <br>
Since these optics data exist in ROOTfiles, they must be extracted into an array-like format, similar to how MNIST stores training and test images (e.g. m x m pixels). An array of labels must also be created to identify the corresponding images, similar to the training and test labels used in MNIST.  I have already extracted the images to a binary format readable by the main code.

**Directory Structure:**<br>

`./ROOTfiles`: ROOTfiles containing the raw data. The files with the "_hist.root" extension contain the relevant histogram objects with the 2D spectrometer focal-plane correlations. These are the histogram objects which have to be converted to a 2D numpy arrayto be given as input in the neural network. <br>

`./scripts/make_2Doptics.C`: This ROOT C++ script takes a raw data ROOTfile as input and make six 2D correlation histogram objects which are stored in ROOTfiles with the "_hist.root" extension. <br>

`./scripts/save2binary.py`: This python script does the following: 1) reads the histogram objects for each "_hist.root" file, 2) converts the histogram objects to a 2D numpy array, and 3) saves the 2D numpy arrays, and the corresponding image labels to a binary file format (.h5), either under the name *optics\_training.h5* or *optics\_test.h5*, depending on whether the data will be used for training or testing the Keras CNN model. <br>

`./scripts/keras_optics.py`: This python script reads the .h5 binary files, trains the neural network and saves the optimized weights under the name *optics\_weights_{ext}.h5*, where {ext} is descriptive name of the 2D optics correlation plot (e.g., xfp\_vs\_yfp, xpfp\_vs\_yfp, etc.) These saved weights are used when testing the model with new test images. The script also make plots of the loss and accuray as a function of the number of epochs after the training, so that the learning curve of the neural network can actually be monitored. 