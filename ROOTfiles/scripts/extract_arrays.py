import ROOT
from ROOT import *
import root_numpy #(make sure to do: pip install root_numpy, before importig this module)
import h5py       # module to help load/save data in binary format .h5
import matplotlib.pyplot as plt
import numpy as np
import os.path

from matplotlib import rc
rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"

# Loop over each quad tune
for Q1 in np.arange(0.90, 1.12, 0.02):
    for Q2 in np.arange(0.95, 1.06, 0.01):
        for Q3 in np.arange(0.90, 1.12, 0.02):

            # define file name
            ifname = "../shms_pointtarg_7p5deg_2gev_wc_mscat_vac_shms_vary_Q1_%.2f_Q2_%.2f_Q3_%.2f_hist.root" % (Q1, Q2, Q3)

            # check if file exists
            if os.path.exists(ifname):
                # Open ROOT file with histogram objects
                tf = TFile(ifname)
            else:
                continue

            print('fname = ', ifname)
'''
# Get histogram object
H_xfp_yfp   = gROOT.FindObject('hxfp_yfp')
H_xfp_ypfp  = gROOT.FindObject('hxfp_ypfp')
H_xfp_xpfp  = gROOT.FindObject('hxfp_xpfp')
H_xpfp_yfp  = gROOT.FindObject('hxpfp_yfp')
H_xpfp_ypfp = gROOT.FindObject('hxpfp_ypfp')
H_ypfp_yfp  = gROOT.FindObject('hypfp_yfp')

# convert ROOT histogram to numpy array
imgArr_xfp_yfp   = root_numpy.hist2array(H_xfp_yfp)
imgArr_xfp_ypfp  = root_numpy.hist2array(H_xfp_ypfp)
imgArr_xfp_xpfp  = root_numpy.hist2array(H_xfp_xpfp)
imgArr_xpfp_yfp  = root_numpy.hist2array(H_xpfp_yfp)
imgArr_xpfp_ypfp = root_numpy.hist2array(H_xpfp_ypfp)
imgArr_ypfp_yfp  = root_numpy.hist2array(H_ypfp_yfp)
'''

# Save data images in binary format
#h5f = h5py.File('optics_training.h5', 'w')
#h5f.create_dataset('images', data=np.array([imgArr_xfp_yfp, imgArr_xfp_ypfp, imgArr_xfp_xpfp]))   

# this is the tune configuration, which is specific for each tune, 0-> Q1_1.10, Q2_1.00, Q3_1.00, etc
# the issue is that if we specify a very complicated label into the neural network, such as Q1_1.10_Q2_1.00_Q3_1.00,
# then how would the network compare its own output to our complicated string?  Therefore is better to associate a
# specific tune with a digit, say 0, 1, 2, 3, 4, . . . Then on a separate label, we classify them accordingly
#h5f.create_dataset('label', data=np.array([0,0,0]))

#h5f.create_dataset('tune_config', data=['Q1=1.10, Q2=1.00, Q3=1.00','Q1=1.10, Q2=1.00, Q3=1.00','Q1=1.10, Q2=1.00, Q3=1.00'])   

#h5f.close()

# TODO: Loop over every (Q1,Q2,Q3) quad config setting, and store the images (optics plots), labels (0,1,2,...) and tune_config labels
# into an array, each. Then outside the loop, we open the h5f file, create and store a dataset for each.


'''
fig, axs = plt.subplots(2, 3)
plt.suptitle('SHMS Quads Tuned: Q1 1.10, Q2 1.00, Q3 1.00')
ax = plt.subplot(2, 3, 1)
ax.set_title(r'$x_{fp}$ vs. $y_{fp}$')
plt.imshow(imgArr_xfp_yfp, cmap='gray_r')

ax = plt.subplot(2, 3, 2)
ax.set_title(r'$x_{fp}$ vs. $y^{\prime}_{fp}$')
plt.imshow(imgArr_xfp_ypfp, cmap='gray_r')

ax = plt.subplot(2, 3, 3)
ax.set_title(r'$x_{fp}$ vs. $x^{\prime}_{fp}$')
plt.imshow(imgArr_xfp_xpfp, cmap='gray_r')

ax = plt.subplot(2, 3, 4)
ax.set_title(r'$x^{\prime}_{fp}$ vs. $y_{fp}$')
plt.imshow(imgArr_xpfp_yfp, cmap='gray_r')

ax = plt.subplot(2, 3, 5)
ax.set_title(r'$x^{\prime}_{fp}$ vs. $y^{\prime}_{fp}$')
plt.imshow(imgArr_xpfp_ypfp, cmap='gray_r')

ax = plt.subplot(2, 3, 6)
ax.set_title(r'$y^{\prime}_{fp}$ vs. $y_{fp}$')
plt.imshow(imgArr_ypfp_yfp, cmap='gray_r')

plt.show()
'''
