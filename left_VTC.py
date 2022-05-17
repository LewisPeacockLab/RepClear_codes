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

brain_flags=['T1w','MNI']  #or MNI
task_flags=['preremoval','study','postremoval'] #preremoval, study, postremoval 

for brain_flag in brain_flags:
    for task_flag in task_flags:
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

                lh_parahippocampal_anterior = [os.path.join(outdir,'lh_parahippocampal_anterior'),1006]

                lh_parahippocampal = [os.path.join(outdir,'lh_parahippocampal.nii.gz'),1016]
            
                lh_inferiortemporal = [os.path.join(outdir,'lh_inferiortemporal.nii.gz'),1009]
            
                lh_lingual = [os.path.join(outdir,'lh_lingual.nii.gz'),1013]
                
                lh_lateraloccipital = [os.path.join(outdir,'lh_lateraloccipital.nii.gz'),1011]

                lh_calcarine = [os.path.join(outdir,'lh_paracalarine.nii.gz'),1021]

                lh_cuneus = [os.path.join(outdir,'lh_cuneus.nii.gz'),1005]

                lh_temporal_pole = [os.path.join(outdir,'lh_temporalpole.nii.gz'),1033]

                lh_middletemporal = [os.path.join(outdir,'lh_middletemporal.nii.gz'), 1015]

                lh_superiortemporal = [os.path.join(outdir,'lh_superiortemporal.nii.gz'),1030]

                lh_pericalcarine = [os.path.join(outdir,'lh_pericalcarine.nii.gz'),1021]
            
                for roi in [lh_fusiform,lh_parahippocampal_anterior,lh_parahippocampal,lh_inferiortemporal,lh_lingual,lh_lateraloccipital,lh_calcarine,lh_cuneus,lh_temporal_pole, lh_middletemporal, lh_superiortemporal, lh_pericalcarine]:
                        os.system('fslmaths %s -thr %s -uthr %s %s'%(aparc_aseg, roi[1], roi[1], roi[0]))
                out_mask = os.path.join(outdir, 'left_VVS_%s_%s_mask.nii.gz' % (task_flag,brain_flag))
                        
                os.system('fslmaths %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s  -bin %s ' % (lh_fusiform[0], lh_parahippocampal[0], lh_parahippocampal_anterior[0], lh_inferiortemporal[0] ,lh_lingual[0],  lh_lateraloccipital[0], lh_calcarine[0], lh_cuneus[0], lh_temporal_pole[0], lh_middletemporal[0], lh_superiortemporal[0], lh_pericalcarine[0], out_mask))
            new_mask(subject_path)
            print('%s mask done' % sub)