import numpy as np
import SimpleITK as sitk
import pandas as pd

# this function loads .nrrd files into a 3D matrix and outputs it
# 	the input is the specified file path to the .nrrd file
def load_nrrd(full_path_filename):

	data = sitk.ReadImage( full_path_filename )
	data = sitk.Cast( sitk.RescaleIntensity(data), sitk.sitkUInt8 )
	data = sitk.GetArrayFromImage(data)

	return(data)
	
# this function encodes a 2D file into run-length-encoding format (RLE)
# 	the inpuy is a 2D binary image (1 = positive), the output is a string of the RLE
def run_length_encoding(input_mask):
	
	dots = np.where(input_mask.T.flatten()==1)[0]
	
	run_lengths,prev = [],-2
	
	for b in dots:
		if (b>prev+1): run_lengths.extend((b+1, 0))

		run_lengths[-1] += 1
		prev = b

	return(" ".join([str(i) for i in run_lengths]))
	
### a sample script to produce a prediction 

# load the image file and reformat such that its axis are consistent with the MRI
image = load_nrrd("lgemri.nrrd")



# *** your code goes here for predicting the mask:

mask = np.zeros(image.shape)
mask[image>200] = 1 # a very trivial solution is presented

# ***



# encode in RLE
image_ids = ["ExampleOnlyMRI_slice_"+str(i) for i in range(image.shape[0])]

encode_cavity = []
for i in range(mask.shape[0]):
	encode_cavity.append(run_length_encoding(mask[i,:,:]))

# output to csv file
csv_output = pd.DataFrame(data={"ImageId":image_ids,'EncodeCavity':encode_cavity},columns=['ImageId','EncodeCavity'])
csv_output.to_csv("ExampleOnlyLabels.csv",sep=",",index=False)