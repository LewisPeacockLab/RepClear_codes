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
    container_path='/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep'
    
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

        lh_parahippocampal_anterior = [os.path.join(outdir,'lh_parahippocampal_anterior'),1006]
        rh_parahippocampal_anterior = [os.path.join(outdir,'rh_parahippocampal_anterior'),2006]        

        lh_parahippocampal = [os.path.join(outdir,'lh_parahippocampal.nii.gz'),1016]
        rh_parahippocampal = [os.path.join(outdir,'rh_parahippocampal.nii.gz'),2016]
    
        lh_inferiortemporal = [os.path.join(outdir,'lh_inferiortemporal.nii.gz'),1009]
        rh_inferiortemporal = [os.path.join(outdir,'rh_inferiortemporal.nii.gz'),2009]
    
        lh_lingual = [os.path.join(outdir,'lh_lingual.nii.gz'),1013]
        rh_lingual = [os.path.join(outdir,'rh_lingual.nii.gz'),2013]
        
        lh_lateraloccipital = [os.path.join(outdir,'lh_lateraloccipital.nii.gz'),1011]
        rh_lateraloccipital = [os.path.join(outdir,'rh_lateraloccipital.nii.gz'),2011]

        lh_calcarine = [os.path.join(outdir,'lh_paracalarine.nii.gz'),1021]
        rh_calcarine = [os.path.join(outdir,'rh_paracalcarine.nii.gz'),2021]

        lh_occipital_pole = [os.path.join(outdir,'lh_occipitalpole.nii.gz'),1142]
        rh_occipital_pole = [os.path.join(outdir,'rh_occipitalpole.nii.gz'),2142]

        lh_temporal_pole = [os.path.join(outdir,'lh_temporalpole.nii.gz'),1033]
        rh_temporal_pole = [os.path.join(outdir,'rh_temporalpole.nii.gz'),2033]

        lh_middletemporal = [os.path.join(outdir,'lh_middletemporal.nii.gz'), 1015]
        rh_middletemporal = [os.path.join(outdir,'rh_middletemporal.nii.gz'), 2015]

        lh_superiortemporal = [os.path.join(outdir,'lh_superiortemporal.nii.gz'),1030]
        rh_superiortemporal = [os.path.join(outdir,'rh_superiortemporal.nii.gz'),2030]

        lh_occipitalfusiform = [os.path.join(outdir,'lh_occipitalfusiform.nii.gz'),1117]
        rh_occipitalfusiform = [os.path.join(outdir, 'rh_occipitalfusiform.nii.gz'), 2117]
    
        for roi in [lh_fusiform,rh_fusiform,lh_parahippocampal_anterior,rh_parahippocampal_anterior,lh_parahippocampal,rh_parahippocampal,lh_inferiortemporal,rh_inferiortemporal,lh_lingual,rh_lingual,lh_lateraloccipital,rh_lateraloccipital,lh_calcarine,rh_calcarine,lh_occipital_pole,rh_occipital_pole,lh_temporal_pole, rh_temporal_pole, lh_middletemporal, rh_middletemporal, lh_superiortemporal, rh_superiortemporal, lh_occipitalfusiform, rh_occipitalfusiform]:
                os.system('fslmaths %s -thr %s -uthr %s %s'%(aparc_aseg, roi[1], roi[1], roi[0]))
        out_mask = os.path.join(outdir, 'VVS_%s_%s_mask.nii.gz' % (task_flag,brain_flag))
                
        os.system('fslmaths %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -bin %s'%(lh_fusiform[0], rh_fusiform[0], lh_parahippocampal[0], rh_parahippocampal[0], lh_parahippocampal_anterior[0], rh_parahippocampal_anterior[0], lh_inferiortemporal[0], rh_inferiortemporal[0],lh_lingual[0], rh_lingual[0], lh_lateraloccipital[0], rh_lateraloccipital[0], lh_calcarine[0] ,rh_calcarine[0] ,lh_occipital_pole[0] ,rh_occipital_pole[0] ,lh_temporal_pole[0], rh_temporal_pole[0], lh_middletemporal[0], rh_middletemporal[0], lh_superiortemporal[0], rh_superiortemporal[0], lh_occipitalfusiform[0], rh_occipitalfusiform[0], out_mask))
    new_mask(subject_path)
    print('%s mask done' % sub)