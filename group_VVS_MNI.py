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

subs=['02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','20','23','24','25','26']

brain_flags=['MNI']  #or MNI
task_flags=['preremoval'] #preremoval, study, postremoval 

for brain_flag in brain_flags:
    for task_flag in task_flags:
        roi_paths=[]
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
            
            indir = os.path.join(subject_path, 'new_mask')
            outdir = os.path.join('/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/')
            if not os.path.exists(outdir):
                os.mkdir(outdir)
                
            mask_path=os.path.join('/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/',sub,'new_mask','VVS_%s_%s_mask.nii.gz' % (task_flag,brain_flag))
            roi_paths.append(mask_path)

        out_mask = os.path.join(outdir, 'group_MNI_VTC_mask.nii.gz')
                        
        os.system('fslmaths %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -bin %s'%(roi_paths[0], roi_paths[1], roi_paths[2], roi_paths[3], roi_paths[4], roi_paths[5], roi_paths[6], roi_paths[7], roi_paths[8], roi_paths[9], roi_paths[10], roi_paths[11], roi_paths[12] , roi_paths[13] ,roi_paths[14] ,roi_paths[15] , roi_paths[16], roi_paths[17], roi_paths[18], roi_paths[19], roi_paths[20], roi_paths[21], out_mask))
        