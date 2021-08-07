## Neural Network Applications Project

**Description:** <br>
This project consists of using actual JLab Experimental Hall C available data on spectrometer Optics, and utilizing it to train a convolutional neural network (CNN) to recognize specific patterns. The using the trained (optimized) parameters to  reognize test Optics images and determine with what accuracy is the CNN able to make predictions about the known pattern

**Pre-requisites:** <br>
Since these optics data exist in ROOTfiles, they must be extracted into an array-like format, similar to how MNIST stores training and test images (e.g. m x m pixels). An array of labels must also be created to identify the corresponding images, similar to the training and test labels used in MNIST. 

**Steps-to-take:** <br>
Follow the steps to train the CNN covered in Week 4, using python Keras implementation in CNNs. 
