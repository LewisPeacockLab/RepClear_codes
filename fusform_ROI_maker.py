#Imports
import warnings
import sys
if not sys.warnoptions:
        warnings.simplefilter("ignore")
import numpy as np
import nibabel as nib
import scipy as scipy
import scipy.io as sio
from nilearn.input_data import NiftiMasker, MultiNiftiMasker
from scipy import stats
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import os
import fnmatch
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import PredefinedSplit, cross_validate, cross_val_predict
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest
from sklearn.preprocessing import StandardScaler

subs=['02','03','04']


brain_flag='MNI'  #or MNI
task_flag='preremoval' #preremoval, study, postremoval 

for num in range(len(subs)):
    sub_num=subs[num]
    
    #define the subject
    sub = ('sub-0%s' % sub_num)
    container_path='/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep'
    
    bold_path=os.path.join(container_path,sub,'func/')
    os.chdir(bold_path)
    
    #set up the path to the files and then moved into that directory
    
    #find the proper nii.gz files
    def find(pattern, path):
            result = []
            for root, dirs, files in os.walk(path):
                    for name in files:
                            if fnmatch.fnmatch(name, pattern):
                                    result.append(os.path.join(root, name))
            return result
        
    
    functional_files=find('*-%s_*.nii.gz' % task_flag, bold_path)
    wholebrain_mask_path=find('*-%s_*mask*.nii.gz' % task_flag, bold_path)
    
    if brain_flag=='MNI':
        pattern = '*MNI152NLin2009cAsym*'
        pattern2= '*MNI152NLin2009cAsym*aparcaseg*'
        brain_mask_path = fnmatch.filter(wholebrain_mask_path,pattern)
        functional_files = fnmatch.filter(functional_files,pattern2)
    elif brain_flag=='T1w':
        pattern = '*T1w*'
        pattern2= '*T1w*aparcaseg*'
        brain_mask_path = fnmatch.filter(wholebrain_mask_path,pattern)
        functional_files = fnmatch.filter(functional_files,pattern2)
    
    subject_path=os.path.join(container_path,sub)
    
    def new_mask(subject_path):
        srcdir = subject_path
        outdir = os.path.join(subject_path, 'new_mask')
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        aparc_aseg = functional_files[0]
    
        lh_fusiform = [os.path.join(outdir,'lh_fusiform.nii.gz'),1007]
        rh_fusiform = [os.path.join(outdir,'rh_fusiform.nii.gz'),2007]
    
        lh_inferiortemporal = [os.path.join(outdir,'lh_inferiortemporal.nii.gz'),1009]
        rh_inferiortemporal = [os.path.join(outdir,'rh_inferiortemporal.nii.gz'),2009]
    
        for roi in [lh_fusiform,rh_fusiform,lh_inferiortemporal,rh_inferiortemporal]:
                os.system('fslmaths %s -thr %s -uthr %s %s'%(aparc_aseg, roi[1], roi[1], roi[0]))
        out_mask = os.path.join(outdir, 'Fusiform_%s_%s_mask.nii.gz' % (task_flag,brain_flag))
                
        os.system('fslmaths %s -add %s -add %s -add %s -bin %s'%(lh_fusiform[0],rh_fusiform[0],lh_inferiortemporal[0],rh_inferiortemporal[0],out_mask))
    new_mask(subject_path)
    print('%s mask done' % sub)