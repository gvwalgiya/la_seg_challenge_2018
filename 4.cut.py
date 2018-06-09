import SimpleITK as sitk
import glob
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import skimage.segmentation as seg
import os
import sys
import cv2

# this function loads .nrrd files into a 3D matrix and outputs it
# 	the input is the specified file path to the .nrrd file
def load_nrrd(full_path_filename):
    data = sitk.ReadImage( full_path_filename )
    data = sitk.Cast( sitk.RescaleIntensity(data), sitk.sitkUInt8 )
    data = sitk.GetArrayFromImage(data)
    return(data)

def min_max():
    training_folders = glob.glob("./Training Set/*")

    max_x = 0
    max_y = 0

    min_x = 1000
    min_y = 1000

    for from_folder in training_folders:
        print(from_folder[-20:])

        laendo = load_nrrd(from_folder+"/laendo.nrrd")
        
        for layer in range(0,88):
            x_reduce = np.where(np.sum(laendo[33,:,:], 1))[0]
            y_reduce = np.where(np.sum(laendo[33,:,:], 0))[0]

            if x_reduce.size == 0:
                continue

            max_x_l = np.max(x_reduce)
            max_y_l = np.max(y_reduce)
            if max_x_l > max_x:
                max_x = max_x_l
            if max_y_l > max_y:
                max_y = max_y_l

            min_y_l = np.min(y_reduce)
            min_x_l = np.min(x_reduce)
            if min_x_l < min_x:
                min_x = min_x_l
            if min_y_l < min_y:
                min_y = min_y_l
        
        print("max_x: "+str(max_x))
        print("max_y: "+str(max_y))
        print("min_x: "+str(min_x))
        print("min_y: "+str(min_y))

    print(np.shape(laendo))

    return(max_x, max_y, min_x, min_y)

def seperate_standardise_cut(from_dir, to_dir):
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
        plt.imsave(to_dir+"/lgemri/layer_%s.png" % (layer+1), lgemri[layer,119:519,155:475])
        # Standardise data
        lgemri_cv_norm = np.empty_like(lgemri[layer,119:519,155:475])
        cv2.normalize(lgemri[layer,119:519,155:475], lgemri_cv_norm, 0, 255, cv2.NORM_MINMAX)
        plt.imsave(to_dir+"/cv_norm/layer_%s.png" % (layer+1), lgemri_cv_norm)
        np.save(to_dir+"/cv_norm/layer_%s" % (layer+1), lgemri_cv_norm)

        np.save(to_dir+"/laendo/layer_%s" % (layer+1), laendo[layer,119:519,155:475])

        # Overlay the manual annotations on the MRIs
        plt.imsave(to_dir+"/boundaries/layer_%s.png" % (layer+1),seg.mark_boundaries(lgemri[layer,119:519,155:475],laendo[layer,119:519,155:475]))

training_folders = glob.glob("./Training Set/*")

i = 1
for from_folder in training_folders:
    print(i)
    print(from_folder[-20:])
    seperate_standardise_cut(from_folder, './cut/'+str(i)+"_"+from_folder[-20:])
    i+=1