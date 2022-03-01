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

workspace = 'scratch'
if workspace == 'work':
    data_dir = '/work/06873/zbretton/frontera/fmriprep/'
    event_dir = '/work/06873/zbretton/frontera/events/'
    results_dir = '/work/06873/zbretton/model_fitting_results/'
elif workspace == 'scratch':
    data_dir = '/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/'
    event_dir = '/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/'
    results_dir = '/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/model_fitting_results/'

sub2rsm=np.load(os.path.join(results_dir,'RSM/sub-002_pre_LSS_tmap_rsm.npy'))

sub3rsm=np.load(os.path.join(results_dir,'RSM/sub-003_pre_LSS_tmap_rsm.npy'))

sub4rsm=np.load(os.path.join(results_dir,'RSM/sub-004_pre_LSS_tmap_rsm.npy'))

plt.subplot(1,3,1, title='Sub-002 LSS RSM: Prelocalizer')
plt.imshow(
    sub2rsm, 
    vmin=-1,
    vmax=1,
    cmap='bwr', 
)
plt.vlines([59,179],0,179)
plt.hlines([59,179],0,179)
plt.subplot(1,3,2,title='Sub-003 LSS RSM: Prelocalizer')
plt.imshow(
    sub3rsm, 
    vmin=-1,
    vmax=1,     
    cmap='bwr', 
)
plt.vlines([59,179],0,179)
plt.hlines([59,179],0,179)
plt.subplot(1,3,3,title='Sub-004 LSS RSM: Prelocalizer')
plt.imshow(
    sub4rsm,
    vmin=-1,
    vmax=1,    
    cmap='bwr')
plt.vlines([59,179],0,179)
plt.hlines([59,179],0,179)
plt.tight_layout()
plt.colorbar()

