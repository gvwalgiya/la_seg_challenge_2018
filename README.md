# la_seg_challenge_2018
My attempt at the LA Segmentation Challenge 2018

Training_set/ - training data folder, 100 people, unchanged from the zip file

sample_*/ - selected random samples for exploratory analysis and visualisation

0.random_sample.py - selects a random sample, places the sample in a folder (I use sample_1/, sample_2/, etc.)

1.seperate.py - separates the different layers out of the MRI and manual annotation .nrrd files. Also merges the MRI and annotations for every 

2.standardise.py - standardises histogram profiles for different samples. 

sample_code.py - came with the data zip file, used for submissions in the end

submission.csv - also came with the data zip, example submission file

train labels.csv - came with the data zip, not too sure what this is