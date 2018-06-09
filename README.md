# la_seg_challenge_2018
Attempt at the LA Segmentation Challenge 2018
Note of caution - Running everything, including the normalisation significance data reorganisation, will take about 10GB.

To use:
1. Download the training data from http://atriaseg2018.cardiacatlas.org/
2. Unzip the training data, there should be a folder called "Training_set/"
3. Copy everything in the unzipped "Training_set/" into the "Training_set/" folder in the repo
4. In the repo folder, run run_everything.sh. Install whatever missing packages it says you're missing.


Training_set/ - training data folder, 100 people, unchanged from the zip file

sample_*/ (e.g. sample_1/, sample_2/, etc.)- selected random samples for exploratory analysis and visualisation

0.random_sample.py - selects a random sample, places the sample in a folder (Jason: I use sample_1/, sample_2/, etc.)

1.seperate.py - separates the different layers out of the MRI and manual annotation .nrrd files. Also merges the MRI and annotations for every layer

2.standardise.py - standardises histogram profiles for different samples. I'm not sure the method is good though after looking at more histograms - will revisit.

sig_resave.py - data reorganisation for significance.py

significance.py - test to see if there are significant differences in intensity values between samples and within samples for each method of standardising. There shouldn't be because the intensity/contrast does not actually mean anything in terms of where the LA is.

sample_code.py - came with the data zip file, used for submissions in the end

submission.csv - also came with the data zip, example submission file

train labels.csv - came with the data zip, not too sure what this is
