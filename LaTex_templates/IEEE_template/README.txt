This directory has a latex template based on the
IEEE format for producing a research paper.

I extracted the descriptions of the relevant sections below from
the following source:  https://libguides.bc.edu/edpaper/sections

For the purposes of writing the research paper
about "Neural Networks" as part of the ODU Mentoring Program,
the suggested sections to include are:


***************
 1. Abstract
***************

Often only 100 to 300 words, the abstract generally provides a broad overview and is never more than a page.
It describes the essence, the main theme of the paper. It includes the research question posed, its significance,
the methodology, and the main results or findings. Remember to take great care in composing the abstract. It's the
first part of the paper the instructor reads. It must impress with a strong content, good style, and general aesthetic appeal.

-----------------------------------
Example of how to start Abstract: (DO NOT COPY THE EXACT SAME SENTENCES, THIS IS JUST A GUIDE)
-----------------------------------
" The aim of this research paper is to develop a pattern recognition algorithm for predicting
simple image patterns using actual simulated data from Experimental Hall C at Jefferson Lab.
... Give a 1-2 sentence summary of what you did and your results (e.g., implemented Convolutional Neural Networks
using the Keras deep learning Application Programming Interface (API), the Keras models were trained with accuracy reaching
nearly 100 % or whichever accuracy you get, and loss of so an so . . .,   the models' predictive power was tested with a new
set of images with >60 % accuracy, . . .  )"


******************
 2. Introduction
******************
A good introduction states the main research problem and thesis argument. What precisely are you studying
and why is it important? How original is it? Will it fill a gap in other studies? Never provide a lengthy
justification for your topic before it has been explicitly stated.

-----------------------------------
Example of how to start Introduction: (DO NOT COPY THE EXACT SAME SENTENCES, THIS IS JUST A GUIDE)
-----------------------------------
The field of Artificial Intelligence (AI) was born in the 1950s and its goal at the time was to teach
computers how to actually think like humans.  Machine Learning (ML) was subsequently developed to study of
how to train a system to solve a problem rather than explicitly programming the rules. Deep Learning methods, which is
a subset of ML, are based on Artificial Neural Networks (ANNs) inspired by biological neural networks and how the
actual brain works to make decisions using its complex system of inter-connected neurons (See Fig. 1)

[Here you would put a picture, to illustarte the point]

This research study uses a class of ANNs, known as Convolutional Neural Networks (CNNs) in order to study and analyze visual
imagery. The logic of how a CNN work is as follows [1]:   (Note I gave a reference [1] to refer to the steps described in the "Neural Networks: Main Concepts" section of the cited [1] article)

     1)  Read the input data (2D array of image to be analyzed)
     2)  Make a prediction (after data has passed through all network layers)
     3)  Compare the prediction to the desired output (during data training, we do know the desired output beforehand)
     4)  Adjust the internal parameters (weights and biases) such as to minimize the loss function

A simple illustration of neural network architecture is shown in Fig. 2 [1]

[Here you would show a figure from the reference [1], whcih shows a dancing cartoon translated to "dancing" output ]


*****************
 3. Methodology
*****************
Discuss your research methodology. Did you employ qualitative or quantitative research methods?
Did you administer a questionnaire or interview people? Any field research conducted? How did you collect data?
Did you utilize other libraries or archives? And so on.

-----------------------------------
Example of how to start Methodology: (DO NOT COPY THE EXACT SAME SENTENCES, THIS IS JUST A GUIDE)
-----------------------------------

In this research, we analyze a total of 185 distinct simulated optics patterns from the Super High Momentum Spectrometer (SHMS) at Hall C of Jefferson Lab.
There were six different optics correlations (xfp_vs_yfp, xpfp_vs_ypfp, etc. you will see this when you do the final project . . .), each pattern had 31
optics images with varying optics tunes [Q1, Q2, Q3], corresponding to the spectrometer quadrupole magnets, summarized in Table 1:

------------------------------------
|       Range     |   Stepsize     |
------------------------------------
Q1    [0.90, 1.10]      0.02       |  
Q2    [0.95, 1.05]      0.01       |
Q3    [0.90, 1.10]      0.02       |
------------------------------------
Table 1: SHMS Quadrupoles Optics Tunes Configurations (training data) .

Each of the six 2D SHMS optics pattern correlations were trained separately, using 31 different optics tunes
per correlation plot which give a total of 6x31=185 images.

The optics patters for testing the network consisted of only varying Q2 from 0.945 to 1.055 in steps of 0.01, while
keeping Q1 and Q3 tunes fixed at unity.

To test the neural network after it had been trained, a set of 10 images were used for each 2D optics correlation, where Q1 and Q3
tunes were kept fixed at unity while Q2 was varied from 0.955 to 1.055 in steps of 0.01 for a total of 10 Q2 tunes.

++++++++++++++++++++++++++++
** IMPORTANT: Keep explaining what you did, or how was the data collected (I CAN HELP WITH THIS SINCE I WROTE THE CODES TO EXTRACT THE DATA IMAGES)
(See paragraph below, where I expalined roughly what I actually did. You can write this part as it is, and then put a citation that you got this
information through me. For example:  [3] Private communication. C. Yero. August 2021. (there is a template of how to do this in references.bib of Latex_Templates/IEEE directory.)
++++++++++++++++++++++++++++


The data with specific [Q1,Q2,Q3] tunes were simulated using the standard Hall C simulation program (mc-single-arm)
and the raw data output was written to a ROOTfile. A separate ROOT C++ script (make_2Doptics.C) was used to form each of
the six abovementioned 2D focal plane correlations correlations which were stored in a separate ROOTfile as histogram objects.
The 2D histograms were then converted to a 2D pixelated array and stored in binary format (.h5) via a Python code (save2binary.py)
array to be read by the Neural Network using Python Keras. Each optics image used was 200x200 pixels and was passed thorugh each of
the hidden layers of the network described in Section 4 of this article.


**************
 4. Main Body (use a different title than "Main Body", maybe "Data Analysis Procedure")
**************

This is generally the longest part of the paper. It's where the author supports the thesis and builds the argument.
It contains most of the citations and analysis. This section should focus on a rational development of the thesis with clear
reasoning and solid argumentation at all points. A clear focus, avoiding meaningless digressions, provides the essential unity
that characterizes a strong education paper.

-----------------------------------
Example of how to start Main Body: (DO NOT COPY THE EXACT SAME SENTENCES, THIS IS JUST A GUIDE)
-----------------------------------

Briefly explaing or introduce the name of each layer used in the CNN,

briefly explain what each layer of the CNN did:  conv2d layer, maxpooling, softmax, etc.
(You can divide this part into different subsections within the main body of the paper)

**** (feel free to use the blog you have been reading as guide) ****

 Subsection: Convolutional Layer
 Give brief description of what is a convolutional layer, and adapt the parameters that apply to the actual project you will be working on.
 For example, rather than inputing a 28x28 pixel image, you would say you input a 200x200 pixel image,  number of filters (you will find all these
 information when you actually run the keras code in Week5)

 Subsection: Pooling Layer
 Give brief description of what is a pooling layer and adapt the parameters,for examlple, what pooling size did you use
 (you will find all these information when you actually run the keras code in Week5)

 Subsection: Activation Function Layer
 Give brief description of what is an activation function layer is and adapt the parameters,for examlple, what activation did you used? Softmax, then maybe just give
 the formula for softmax, and explain that it outputs a vector with probabilities, and the highest probability is the actual network prediction, etc. 
 (you will find all these information when you actually run the keras code in Week5)

[Here you would put a picture of the CNN network layers used]  See image of Reference [2], which has a cartoon illustrating how the
image was trasnformed at each step. You would have to adapt that image to include the actual size of the optics image used (200x200 pixels, rather than 28x28), and so on


**IMPORTANT:  Don't worry about putting the details of the math (partial derivatives) that was done to actually carry out the forward/bacpropagation of the neural network.
Just focus on explaining the basics of each layer used, and just mention that once the image passed through the layers, the output was compared to the known result, and
a backpropagation method was done to minimize the loss by determining the optimum parameters. And mention that an epoch consists of a complete forward/backward propagation.
Then, the images were re-analyzed with the updated parameters in subsequent epochs to further optimize the parameters and minimize the loss.  ** You'll probably have to also
give a brief 1-sentence description of what the loss is in a neural network.

***************************
5. Results and Discussion
***************************

After spending a great deal of time and energy introducing and arguing the points in the main body of the paper,
the conclusion brings everything together and underscores what it all means. A stimulating and informative conclusion
leaves the reader informed and well-satisfied. A conclusion that makes sense, when read independently from the rest of
the paper, will win praise.

-------------------------------------------------
Example of how to start Results and Discussion: (DO NOT COPY THE EXACT SAME SENTENCES, THIS IS JUST A GUIDE)
-------------------------------------------------

The purpose of this research was to teach a machine to recognize optics patterns that would otherwise be
difficult to distinguish by the "human eye". With the help of Keras API, we were able to train and test a CNN
by providing simulated optics data from Jefferson Lab, Hall C.  Each of the six 2D optics correlation was trained with
31 optics tunes, and were able to reach and plateau at an accuracy of ~X %, and a loss of ~Y%, in Z epochs of training.
We used 10 test images per each of the six 2D optics correlations, and network was able to correctly predict
each the patterns with at least ~N% accuracy. The results of the training are shown in Fig. 3

[Here show the plot of Accuracy (and Loss) vs. epochs] to show how the training progresses.

The results of the test images is summarized in Table 2

--------------------------------------------------------------------------------------------
2D optics correlation   |  # of predicted patterns | total # of patterns  | Accuracy
xfp_vs_xpfp                     9                         10                  0.9 ( 90 %)
xfp_vs_yfp                      10                        10                  1.0 (100%)
xfp_vs_ypfp                     
xpfp_vs_yfp                     6                         10                  0.6 (60 %)
xpfp_vs_ypfp
ypfp_vs_yfp
----------------------------------------------------------------------------------------------
Table 2.  Performance of the trained CNN for each of the 2D optics correlation patterns



***************
6. References
***************

Here you would put the references with the corresponding reference numbers in the research paper:
You would need to determine what are you citing and the based on that, you can put your reference.

Suggestion: Currently, you explicitly have to write references at the end of the LaTex template, however,
an easier way would be to write the references on a separate file, called for example: references.bib,
and you simply determine which reference type you need, then fill in the required fields, and LaTeX
will automatically include the correct format of the reference. We can discuss this in the next meeting,
but basically you would do the follwoing:  1) fill in the fields in the references.bib file,
2) execute the commands: pdflatex main_latex_file.tex  (this is to compile and generate an main_latex_file.aux file),
then you would execute: bibtex main_latex_file.aux to compile the references, and finally, you would do:
pdflatex main_latex_file.tex  (do this two times as ti it required to compile cross-references, so it may
appear on both your paper and the reference at the bottom)


[1] Mesquita, D. Title, Month, Year Online: https://realpython.com/python-ai-neural-network/

[2] https://victorzhou.com/blog/intro-to-cnns-part-2/

[3]
 .
 .
 .



----------------------------

***********************
7. Appendix (Optional)
***********************

Here you would include additional detailed information your research, as long
as this information is NOT essential for understanding and interpreting the results
of your research. 
