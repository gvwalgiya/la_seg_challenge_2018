import SimpleITK as sitk
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import skimage.segmentation as seg
import os
import numpy as np
import sys
import cv2

# this function loads .nrrd files into a 3D matrix and outputs it
# 	the input is the specified file path to the .nrrd file
def load_nrrd(full_path_filename):
    data = sitk.ReadImage( full_path_filename )
    data = sitk.Cast( sitk.RescaleIntensity(data), sitk.sitkUInt8 )
    data = sitk.GetArrayFromImage(data)
    return(data)

def seperate_standardise(from_dir, to_dir):
    # Make all the folders if they're not there already
    if not os.path.exists(to_dir+"/boundaries/"):
        os.makedirs(to_dir+"/boundaries/")
    if not os.path.exists(to_dir+"/laendo/"):
        os.makedirs(to_dir+"/laendo/")
    if not os.path.exists(to_dir+"/lgemri/"):
        os.makedirs(to_dir+"/lgemri/")
    if not os.path.exists(to_dir+"/cv_norm/"):
        os.makedirs(to_dir+"/cv_norm/")

    # Load the image file and reformat such that its axis are consistent with the MRI
    lgemri = load_nrrd(from_dir+"/lgemri.nrrd")
    laendo = load_nrrd(from_dir+"/laendo.nrrd")

    # Save each layer separately
    for layer in range(31,36):
        plt.imsave(to_dir+"/lgemri/layer_%s.png" % (layer+1), lgemri[layer,:,:])
        # Standardise data
        lgemri_cv_norm = np.empty_like(lgemri[layer,:,:])
        cv2.normalize(lgemri[layer,:,:], lgemri_cv_norm, 0, 255, cv2.NORM_MINMAX)
        plt.imsave(to_dir+"/cv_norm/layer_%s.png" % (layer+1), lgemri_cv_norm)
        np.save(to_dir+"/cv_norm/layer_%s" % (layer+1), lgemri_cv_norm)

        np.save(to_dir+"/laendo/layer_%s" % (layer+1), laendo[layer,:,:])

        # Overlay the manual annotations on the MRIs
        plt.imsave(to_dir+"/boundaries/layer_%s.png" % (layer+1),seg.mark_boundaries(lgemri[layer,:,:],laendo[layer,:,:]))