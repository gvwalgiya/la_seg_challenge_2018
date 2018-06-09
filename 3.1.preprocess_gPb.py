import glob

from seperate_std import seperate_standardise

training_folders = glob.glob("./Training Set/*")

i = 1
for from_folder in training_folders:
    print(i)
    print(from_folder[-20:])
    seperate_standardise(from_folder, './preprocessed/'+str(i)+"_"+from_folder[-20:])
    i+=1