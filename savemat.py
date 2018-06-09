from scipy.io import savemat
import numpy as np
from glob import glob
import numpy as np

layerglob = glob('./sample_1/lgemri/*.npy')

layers = {}

for filepath in layerglob:
    if len(filepath) == 29:
        layers[filepath[18:25]] = np.load(filepath)
    else:
        layers[filepath[18:26]] = np.load(filepath)

savemat("sample_1",layers)