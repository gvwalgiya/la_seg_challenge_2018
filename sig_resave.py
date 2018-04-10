'''
sig_save_df.py
Reorganise saved data from each method into dataframes and save as pickles for tests about normalisation.
'''
import numpy as np
import pandas as pd
import os

print(__doc__)

samples = ["sample_1", "sample_2", "sample_3"]
savedir = "reorganised_data/"

# Make all the folders if they're not there already
if not os.path.exists(savedir):
    os.makedirs(savedir)

print("lgemri")
for sample in samples:
    lgemri_list = []
    for layer in range(88):
        current = np.load(sample+"/lgemri/layer_"+str(layer+1)+".npy")
        lgemri_list.append(current.flatten())
    np.save(savedir+sample+"_lgemri", lgemri_list)
print("lgemri end")

print("rescl")
for sample in samples:
    rescl_list = []
    for layer in range(88):
        current = np.load(sample+"/rescl/layer_"+str(layer+1)+".npy")
        rescl_list.append(current.flatten())
    np.save(savedir+sample+"_rescl", rescl_list)
print("rescl end")

print("std")
for sample in samples:
    std_list = []
    for layer in range(88):
        current = np.load(sample+"/std/layer_"+str(layer+1)+".npy")
        std_list.append(current.flatten())
    np.save(savedir+sample+"_std", std_list)
print("std end")

print("cv_norm")
for sample in samples:
    cv_norm_list = []
    for layer in range(88):
        current = np.load(sample+"/cv_norm/layer_"+str(layer+1)+".npy")
        cv_norm_list.append(current.flatten())
    np.save(savedir+sample+"_cv_norm", cv_norm_list)
print("cv_norm end")

# gitignore everything in the new directory so we don't upload this to github
gitignore = open(savedir+".gitignore", "w")
gitignore.write("# Ignore everything in this directory\n*\n# Except this file\n!.gitignore")
gitignore.close()