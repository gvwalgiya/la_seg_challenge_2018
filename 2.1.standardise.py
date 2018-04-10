import SimpleITK as sitk
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import skimage.segmentation as seg
from skimage import exposure, img_as_float
import os
import numpy as np
import sys

dir = sys.argv[1]

# Make all the folders if they're not there already
if not os.path.exists(dir+"/std/"):
    os.makedirs(dir+"/std/")

# this function loads .nrrd files into a 3D matrix and outputs it
# 	the input is the specified file path to the .nrrd file
def load_nrrd(full_path_filename):

	data = sitk.ReadImage( full_path_filename )
	data = sitk.Cast( sitk.RescaleIntensity(data), sitk.sitkUInt8 )
	data = sitk.GetArrayFromImage(data)

	return(data)

# Load the image file and reformat such that its axis are consistent with the MRI
lgemri = load_nrrd(dir+"/lgemri.nrrd")
laendo = load_nrrd(dir+"/laendo.nrrd")

# Rescale images to maintain similar contrast and intensity profile between different samples
for layer in range(lgemri.shape[0]):
    # Standardise data
    lgemri_std = (lgemri-np.average(lgemri)/np.std(lgemri)
    plt.imsave(dir+"/std/%s_layer.png" % (layer+1), seg.mark_boundaries(lgemri_std,laendo[layer,:,:]))
    # Show processed histograms for every layer - see if they're roughly similar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(lgemri_std.ravel(),bins=256, histtype='step', color='black')
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax.set_xlabel('Pixel intensity')
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    plt.savefig(dir+"/std/%s_hist.png" % (layer+1))
    plt.close()