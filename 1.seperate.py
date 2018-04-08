import SimpleITK as sitk
import matplotlib.pyplot as plt
import skimage.segmentation as seg
from skimage import exposure, img_as_float
import os
import numpy as np
import sys

dir = sys.argv[1]

# Make all the folders if they're not there already
if not os.path.exists(dir+"/boundaries/"):
    os.makedirs(dir+"/boundaries/")
if not os.path.exists(dir+"/laendo/"):
    os.makedirs(dir+"/laendo/")
if not os.path.exists(dir+"/lgemri/"):
    os.makedirs(dir+"/lgemri/")

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

# Save each layer separately
for layer in range(lgemri.shape[0]):
   plt.imsave(dir+"/lgemri/layer_%s.png" % (layer+1),lgemri[layer,:,:])
for layer in range(lgemri.shape[0]):
   plt.imsave(dir+"/laendo/layer_%s.png" % (layer+1),laendo[layer,:,:])

# Overlay the manual annotations on the MRIs
for layer in range(lgemri.shape[0]):
   plt.imsave(dir+"/boundaries/layer_%s.png" % (layer+1),seg.mark_boundaries(lgemri[layer,:,:],laendo[layer,:,:]))