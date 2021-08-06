import ROOT
from ROOT import *
import root_numpy #(make sure to do: pip install root_numpy, before importig this module)
import matplotlib.pyplot as plt

from matplotlib import rc
rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"

# Open ROOT file with histogram objects
tf = TFile("shms_pointtarg_7p5deg_2gev_wc_mscat_vac_shms_vary_Q1_1.10_Q2_1.00_Q3_1.00.root_hist.root")

# Get histogram object
H_xfp_yfp = gROOT.FindObject('hxfp_yfp')
H_xfp_ypfp = gROOT.FindObject('hxfp_ypfp')
H_xfp_xpfp = gROOT.FindObject('hxfp_xpfp')
H_xpfp_yfp = gROOT.FindObject('hxpfp_yfp')
H_xpfp_ypfp = gROOT.FindObject('hxpfp_ypfp')
H_ypfp_yfp = gROOT.FindObject('hypfp_yfp')

# convert ROOT histogram to numpy array
imgArr_xfp_yfp = root_numpy.hist2array(H_xfp_yfp)
imgArr_xfp_ypfp = root_numpy.hist2array(H_xfp_ypfp)
imgArr_xfp_xpfp = root_numpy.hist2array(H_xfp_xpfp)
imgArr_xpfp_yfp = root_numpy.hist2array(H_xpfp_yfp)
imgArr_xpfp_ypfp = root_numpy.hist2array(H_xpfp_ypfp)
imgArr_ypfp_yfp = root_numpy.hist2array(H_ypfp_yfp)


fig, axs = plt.subplots(2, 3)
plt.suptitle('SHMS Quads Tuned: Q1 1.10, Q2 1.00, Q3 1.00')
ax = plt.subplot(2, 3, 1)
ax.set_title(r'$x_{fp}$ vs. $y_{fp}$')
plt.imshow(-1*imgArr_xfp_yfp, cmap='gray')

ax = plt.subplot(2, 3, 2)
ax.set_title(r'$x_{fp}$ vs. $y^{\prime}_{fp}$')
plt.imshow(-1*imgArr_xfp_ypfp, cmap='gray')

ax = plt.subplot(2, 3, 3)
ax.set_title(r'$x_{fp}$ vs. $x^{\prime}_{fp}$')
plt.imshow(-1*imgArr_xfp_xpfp, cmap='gray')

ax = plt.subplot(2, 3, 4)
ax.set_title(r'$x^{\prime}_{fp}$ vs. $y_{fp}$')
plt.imshow(-1*imgArr_xpfp_yfp, cmap='gray')

ax = plt.subplot(2, 3, 5)
ax.set_title(r'$x^{\prime}_{fp}$ vs. $y^{\prime}_{fp}$')
plt.imshow(-1*imgArr_xpfp_ypfp, cmap='gray')

ax = plt.subplot(2, 3, 6)
ax.set_title(r'$y^{\prime}_{fp}$ vs. $y_{fp}$')
plt.imshow(-1*imgArr_ypfp_yfp, cmap='gray')




#fig, axs = plt.subplots(3, 2)
#axs[0, 0].plot(imgArr_xfp_yfp)

#axs[0, 0].set_title('Axis [0, 0]')

#axs[0, 1].plot(x, y, 'tab:orange')
#axs[0, 1].set_title('Axis [0, 1]')
#axs[1, 0].plot(x, -y, 'tab:green')
#axs[1, 0].set_title('Axis [1, 0]')
#axs[1, 1].plot(x, -y, 'tab:red')
#axs[1, 1].set_title('Axis [1, 1]')

#for ax in axs.flat:
#    ax.set(xlabel='x-label', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
#for ax in axs.flat:
#    ax.label_outer()

    
#plt.imshow(-1*img_arr, cmap='gray')
plt.show()
