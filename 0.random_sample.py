from distutils.dir_util import copy_tree
import os
import sys
import glob
import random

"""  
Selects random people from the 100 samples in the training data set.
Used for exploratory stuff so we can visualise what's happening.
"""

# From script arguments decide where to copy sample to
toDirectory = sys.argv[1]+"/"

# If dir to copy does not exist, make the dir.
if not os.path.exists(toDirectory):
    os.makedirs(toDirectory)

# Randomly select a sample
training_folders = glob.glob("./Training Set/*")
random_index = random.randint(0, len(training_folders)-1)
print(random_index)
fromDirectory = training_folders[random_index]
print(fromDirectory)

# Copy random sample
copy_tree(fromDirectory, toDirectory)

# Make text file that says which sample was selected
text_file = open(toDirectory+fromDirectory[-20:], "w")
text_file.close()

# gitignore everything in the new copied directory so we don't upload this to github
gitignore = open(toDirectory+".gitignore", "w")
gitignore.write("# Ignore everything in this directory\n*\n# Except this file\n!.gitignore")
gitignore.close()