#code to find and grab the study design information for the subjects of interest, this will help us sort trials for later analyses


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

# global consts
subIDs = ['002', '003', '004']
phases = ['rest', 'preremoval', 'study', 'postremoval']

stim_labels = {0: "Rest",
                1: "Scenes",
                2: "Faces"}
sub_cates = {
            "face": ["male", "female"],         #60
            "scene": ["manmade", "natural"],    #120
            }
sub_folders = {subIDs[0]:'202110211',
                subIDs[1]:'202110221',
                subIDs[2]:'202110291'}

workspace = 'local'
if workspace == 'local':
    data_dir = '/Users/zb3663/Box/LewPeaLabBox/STUDY/RepClear/v2_fmri'
    out_dir = '/Users/zb3663/Box/LewPeaLabBox/STUDY/RepClear/Design/subject_designs'

def find(pattern, path): #find the pattern we're looking for
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
        return result

for sub in subIDs:

    #find the .mat file we want to use to load in all the designs for each phase:
    mat_files=find('dataMat*study*03*mat',os.path.join(data_dir,'repclear_%s/' % sub_folders[sub])) # for now this targets the study phase, run 3 (but each file contains the full subject design)
    sub_design=sio.loadmat(mat_files[0],simplify_cells=True)
    pre_localizer_data=sub_design['args']['design']['ph'][1]['matrix']
    pre_headers=sub_design['args']['design']['ph'][1]['header']
    pre_localizer_df=pd.DataFrame(data=pre_localizer_data,columns=pre_headers)

    #this was planned out by looking at the .mat file in matlab and determining the exact paths
    study_data=sub_design['args']['design']['ph'][2]['matrix']
    study_headers=sub_design['args']['design']['ph'][2]['header']
    study_df=pd.DataFrame(data=study_data,columns=study_headers)    

    post_localizer_data=sub_design['args']['design']['ph'][3]['matrix']
    post_headers=sub_design['args']['design']['ph'][3]['header']
    post_localizer_df=pd.DataFrame(data=post_localizer_data,columns=post_headers)

    #create a key to save each of the dataframes with the subject number and then name of the phase
    #pickling because that is the best way to save dataframes 
    df_map={'pre_localizer_design':pre_localizer_df,'study_design':study_df, 'post_localizer_design':post_localizer_df}
    for name, df in df_map.items():
        df.to_pickle(os.path.join(out_dir,f'{sub}_{name}.pkl'))
