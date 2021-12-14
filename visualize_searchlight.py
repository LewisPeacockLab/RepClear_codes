#code to visualize the searchlight results:


#Imports
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import numpy as np
import nibabel as nib
import scipy as scipy
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import os
import fnmatch
import pandas as pd
import pickle

from nilearn import plotting


subIDs = ['002', '003', '004']
phases = ['rest', 'preremoval', 'study', 'postremoval']
runs = np.arange(6) + 1  
spaces = {'T1w': 'T1w', 
            'MNI': 'MNI152NLin2009cAsym'}
descs = ['brain_mask', 'preproc_bold']
ROIs = ['VVS', 'PHG', 'FG']
shift_sizes_TR = [5, 6]

stim_labels = {0: "Rest",
                1: "Scenes",
                2: "Faces"}
sub_cates = {
            "face": ["male", "female"],         #60
            "scene": ["manmade", "natural"],    #120
            }

operations=['maintain','replace','suppress']

workspace = 'scratch'
if workspace == 'work':
    data_dir = '/work/06873/zbretton/frontera/fmriprep/'
    event_dir = '/work/06873/zbretton/frontera/events/'
elif workspace == 'scratch':
    data_dir = '/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/'
    event_dir = '/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/'

for subID in subIDs:
    searchlight_dir=os.path.join(data_dir,'sub-%s' % subID,'searchlight')

    for operation in operations:
        img = nib.load(os.path.join(searchlight_dir,'Sub-%s_SL_%s_result.nii.gz' % (subID,operation)))
        plotting.plot_glass_brain(img,threshold=0.65,colorbar=True,cmap='Reds',plot_abs=False,output_file=(os.path.join(searchlight_dir,'%s_glass_brain.pdf' % operation)))

        plotting.plot_glass_brain(img,colorbar=True,cmap='twilight',vmin=0.5,plot_abs=False,output_file=(os.path.join(searchlight_dir,'%s_glass_brain_full.pdf' % operation)))