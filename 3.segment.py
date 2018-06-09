import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from skimage.segmentation import mark_boundaries, watershed, felzenszwalb
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import sys
import os

sample = sys.argv[1]

# Make all the folders if they're not there already
if not os.path.exists(sample+"/seg/"):
    os.makedirs(sample+"/seg/")

# std_methods = ["lgemri", "cv_norm", "std", "rescl"]
std_methods = ["cv_norm"]

random_slices = [33, 44, 55]

for method in std_methods:
    print(method)
    for slice_number in random_slices:
        print(slice_number)
        mri_slice = np.load(sample+"/"+method+"/layer_"+str(slice_number)+".npy")
        plt.imsave(sample+"/seg/slice_"+str(slice_number)+".jpg",mri_slice)

        blur = cv.GaussianBlur(mri_slice,(13,13),0)
        fel_slice = felzenszwalb(blur, scale=100)
        plt.imsave(sample+"/seg/"+str(slice_number)+"_fel_"+str(method)+".jpg",mark_boundaries(mri_slice,fel_slice))

        blur = cv.GaussianBlur(mri_slice,(25,25),0)
        cv_at = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,75,2)
        plt.imsave(sample+"/seg/"+str(slice_number)+"_cvat_"+str(method)+".jpg",mark_boundaries(mri_slice,cv_at))