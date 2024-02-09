## Neural Network Applications Project

**Description:** <br>
This project consists of utilizing actual JLab Experimental Hall C simulated data on spectrometer optics to train a convolutional neural network (CNN) to recognize specific patterns. Then using the trained (optimized) parameters, a new set of test images is used to determine how well can the trained CNN model predict the tunes of the optics pattern given.

**Pre-requisites:** <br>
Since these optics data exist in ROOTfiles, they must be extracted into an array-like format, similar to how MNIST stores training and test images (e.g. m x m pixels). An array of labels must also be created to identify the corresponding images, similar to the training and test labels used in MNIST.  I have already extracted the images to a binary format readable by the main code.

**Directory Structure:**<br>

`./ROOTfiles`: ROOTfiles containing either the "training" or "testing" the raw data. The files with the "_hist.root" extension contain the relevant histogram objects with the 2D spectrometer focal-plane correlation plots. These are the histogram objects which have to be converted to a 2D numpy arrayto be given as input in the neural network. <br>

`./scripts/make_2Doptics.C`: This ROOT C++ script takes raw data ROOTfiles as input and make six 2D correlation histogram objects per input file (of a specific 'Q1,Q2,Q3' tune) which are stored in ROOTfiles with the "_hist.root" extension. <br>

`./scripts/save2binary.py`: This python script does the following: 1) reads the histogram objects in each "_hist.root" file, 2) converts the histogram objects to a 2D numpy array, and 3) saves the 2D numpy arrays, and the corresponding image labels to a binary file format (.h5), either under the name *optics\_training.h5* or *optics\_test.h5*, depending on whether the data will be used for training or testing the Keras CNN model. <br>

`./scripts/keras_optics.py`: This python script reads the .h5 binary files, trains the neural network and saves the optimized weights under the name *optics\_weights_{ext}.h5*, where {ext} is descriptive name of the 2D optics correlation plot (e.g., xfp\_vs\_yfp, xpfp\_vs\_yfp, etc.) These saved weights are used when testing the model with new test images. The script also make plots of the loss and accuray as a function of the number of epochs after the training, so that the learning curve of the neural network can actually be monitored. 

## Specific Instructions for Students
To run the `save2binary.py` code requires the user to import ROOT. Remember that ROOT CERN will need to be installed (see [ROOT\_CERN Installation](https://root.cern/install/). You can give it a try and run the
code to see if you don't get any errors as you might have it installed already. I can help out with the installation if needed. 

###Optional Task
Once you have uncompressed the Week_5.zip file in your respective work directories, make sure to make a symbolic link to the ROOTfiles directory from directory where this README.md file is located (for example, from a command-line using MacOS or Linux machines, you would do: `ln -s ../../Worked_Exercises/ROOTfiles` which automatically sets the symbolic link to the `ROOTfiles` directory.  This is REQUIRED in order to run the `make_2Doptics.C` code, which looks at the path of the `ROOTfiles`.  You can try this out after the ODU program ends (or even now) if you have the time, so you can become familiarized with ROOT C++ programming, which will be very useful in your future as a researcher, even outside of the science field. 