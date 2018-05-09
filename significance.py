'''
significance.py
Test to see if there are significant differences in intensity values between samples and within samples for each method of standardising. 

There shouldn't be because the intensity/contrast does not actually mean anything in terms of where the LA is.

significance.ipynb didn't go very well. 
Turns out Jupyter uses a lot of memory and my laptop can't handle it so I'm back to using a normal script.

You have to enter the directory of the sample you want analysed
'''


import numpy as np
import pandas as pd
import os
import sys

print(__doc__)

sample = sys.argv[1]

methods = ["lgemri", "rescl", "std", "cv_norm"]
savedir = "reorganised_data/"

data = []
for method in methods:
    print("load %s" %method)
    data.append(np.load(savedir+sample+"_"+method+".npy"))
data = np.array(data).reshape((-1,len(methods)))

print(data.shape)

df = pd.DataFrame(data, columns=methods)

print(df.head)

print(df.summary)