'''
brute_force.py
Gonna cram everything into a NN and see what happens lol
1. Load a segmented data set
2. Pre-process that data set using the method specified
3. Cram the data set into an NN
4. Rinse and repeat for 75 of 100 data sets
5. Validate using the remaining datasets
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import glob


training_folders = glob.glob("./Training Set/*")