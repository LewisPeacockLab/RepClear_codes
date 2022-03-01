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
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import PredefinedSplit, cross_validate, cross_val_predict, GridSearchCV, LeaveOneGroupOut
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest, SelectFpr
from sklearn.preprocessing import StandardScaler
from nilearn.input_data import NiftiMasker,  MultiNiftiMasker
from nilearn.image import clean_img
from nilearn.signal import clean
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed


subs=['02','03','04','05','06','07','08','09','10']
brain_flag='T1w'

#code for the item level voxel activity for faces and scenes


def mkdir(path,local=False):
    if not os.path.exists(path):
        os.makedirs(path)


def pad_contrast(contrast_, n_columns):
    """A small routine to append zeros in contrast vectors"""
    return np.hstack((contrast_, np.zeros(n_columns - len(contrast_))))

def confound_cleaner(confounds):
    COI = ['a_comp_cor_00','framewise_displacement','trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']
    for _c in confounds.columns:
        if 'cosine' in _c:
            COI.append(_c)
    confounds = confounds[COI]
    confounds.loc[0,'framewise_displacement'] = confounds.loc[1:,'framewise_displacement'].mean()
    return confounds    

    #find the proper nii.gz files
def find(pattern, path): #find the pattern we're looking for
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
        return result


def item_representation_study(subID):

    print('Running sub-0%s...' %subID)
    #define the subject
    sub = ('sub-0%s' % subID)
    container_path='/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep'
  
    bold_path=os.path.join(container_path,sub,'func/')
    os.chdir(bold_path)
  
    #set up the path to the files and then moved into that directory

    bold_files=find('*study*bold*.nii.gz',bold_path)
    
    if brain_flag=='MNI':
        pattern = '*MNI*'
        pattern2= '*MNI152NLin2009cAsym*preproc*'
        bold_files = fnmatch.filter(bold_files,pattern2)
    elif brain_flag=='T1w':
        pattern = '*T1w*'
        pattern2 = '*T1w*preproc*'
        bold_files = fnmatch.filter(bold_files,pattern2)
        
    bold_files.sort()
    if brain_flag=='T1w':
        mask_path=os.path.join('/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/',sub,'new_mask','VVS_preremoval_%s_mask.nii.gz' % brain_flag)
        
        mask=nib.load(mask_path)    
    else:
        mask_path=os.path.join('/scratch/06873/zbretton/fmriprep/group_MNI_VTC_mask.nii.gz') #VTC

        mask=nib.load(mask_path) 
    mask_flag='VTC'
    
    #load in category mask that was created from the first GLM  
    #load data if it already exists, otherwise load it directly:
    if os.path.exists(os.path.join(bold_path,'sub-0%s_%s_study.nii.gz' % (subID,brain_flag))):
            
        img=nib.load(os.path.join(bold_path,'sub-0%s_%s_study.nii.gz' % (subID,brain_flag)))
        print('%s Concatenated Study BOLD Loaded...' % (brain_flag))
    else:
        img=nib.concat_images(bold_files,axis=3)

        output_name = os.path.join(bold_path,'sub-0%s_%s_study.nii.gz' % (subID,brain_flag))

        nib.save(img, output_name)  # Save the volume  

        print('%s Concatenated Study BOLD...saved' % (brain_flag))    #to be used to filter the data
    #to be used to filter the data
    #First we are removing the confounds
    #get all the folders within the bold path
    #confound_folders=[x[0] for x in os.walk(bold_path)]
    study_confounds_1=find('*study*1*confounds*.tsv',bold_path)
    study_confounds_2=find('*study*2*confounds*.tsv',bold_path)
    study_confounds_3=find('*study*3*confounds*.tsv',bold_path)


    
    confound_run1 = pd.read_csv(study_confounds_1[0],sep='\t')
    confound_run2 = pd.read_csv(study_confounds_2[0],sep='\t')
    confound_run3 = pd.read_csv(study_confounds_3[0],sep='\t')
          

    confound_run1=confound_cleaner(confound_run1)
    confound_run2=confound_cleaner(confound_run2)
    confound_run3=confound_cleaner(confound_run3)
  
    
    study_confounds=pd.concat([confound_run1,confound_run2,confound_run3], ignore_index=False)  

    #get run list so I can clean the data across each of the runs
    run1_length=int((img.get_fdata().shape[3])/3)
    run2_length=int((img.get_fdata().shape[3])/3)
    run3_length=int((img.get_fdata().shape[3])/3)


    run1=np.full(run1_length,1)
    run2=np.full(run2_length,2)
    run3=np.full(run3_length,3)  

    run_list=np.concatenate((run1,run2,run3))    
    #clean data ahead of slicing the data up
    if os.path.exists(os.path.join(bold_path,'sub-0%s_%s_study_%s_cleaned.nii.gz' % (subID,brain_flag,mask_flag))):
        
        img_clean=nib.load(os.path.join(bold_path,'sub-0%s_%s_study_%s_cleaned.nii.gz' % (subID,brain_flag,mask_flag)))

        print('%s Concatenated, Cleaned and %s Masked Study BOLD...LOADED' % (brain_flag,mask_flag))
        del img
    else:
        print('Cleaning and Masking BOLD data...')
        img_clean=clean_img(img,sessions=run_list,t_r=1,detrend=False,standardize='zscore',mask_img=mask,confounds=study_confounds)

        output_name = os.path.join(bold_path,'sub-0%s_%s_study_%s_cleaned.nii.gz' % (subID,brain_flag,mask_flag))

        nib.save(img_clean, output_name)  # Save the volume  

        print('%s Concatenated, Cleaned and %s Masked Study BOLD...saved' % (brain_flag,mask_flag))
        del img   

    '''load in the denoised bold data and events file'''
    events = pd.read_csv('/work/06873/zbretton/ls6/repclear_dataset/BIDS/task-study_events.tsv',sep='\t')
    #now will need to create a loop where I iterate over the face & scene indexes
    #I then relabel that trial of the face or scene as "face_trial#" or "scene_trial#" and then label rest and all other trials as "other"
    #I can either do this in one loop, or two consecutive

    #I want to ensure that "trial" is the # of face (e.g., first instance of "face" is trial=1, second is trial=2...)
    scene_trials=events.trial_type.value_counts().stim
    #so this will give us a sense of total trials for these two conditions
        #next step is to then get the index of these conditions, and then use the trial# to iterate through the indexes properly

    temp_events=events.copy() #copy the original events list, so that we can convert the "faces" and "scenes" to include the trial # (which corresponds to a unique image)
    scene_index=[i for i, n in enumerate(temp_events['trial_type']) if n == 'stim']#this will find the nth occurance of a desired value in the list    

    for trial in (range(len(scene_index))):    
        temp_events.loc[scene_index[trial],'trial_type']=('scene_trial%s' % (trial+1))


    print('data is loaded, and events file is sorted...')

    for trial in (range(len(scene_index))):

        print('running scene trial %s' % (trial+1))
        #get the onset of the trial so that I can average the time points:
        onset=(temp_events.loc[scene_index[trial],'onset']+5)

        affine_mat=img_clean.affine
        dimsize = img_clean.header.get_zooms()

        '''point to and if necessary create the output folder'''
        out_folder = os.path.join(container_path,sub,'item_representations_%s' % brain_flag)
        if not os.path.exists(out_folder): os.makedirs(out_folder,exist_ok=True)

        
        trial_pattern=np.mean(img_clean.get_fdata()[:,:,:,(onset):(onset+2)],axis=3) #this is getting the 2 TRs for that trial's onset and then taking the average of it across the 4th dimension (time)
        

        output_name = os.path.join(out_folder, ('Sub-0%s_study_scene_trial%s_result.nii.gz' % (subID,(trial+1))))
        trial_pattern = trial_pattern.astype('double')  # Convert the output into a precision format that can be used by other applications
        trial_pattern[np.isnan(trial_pattern)] = 0  # Exchange nans with zero to ensure compatibility with other applications
        trial_pattern_nii = nib.Nifti1Image(trial_pattern, affine_mat)  # create the volume image
        hdr = trial_pattern_nii.header  # get a handle of the .nii file's header
        hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2]))
        nib.save(trial_pattern_nii, output_name)  # Save the volume  

        del trial_pattern, trial_pattern_nii, affine_mat, onset, out_folder, output_name, hdr

Parallel(n_jobs=len(subs))(delayed(item_representation_study)(i) for i in subs)
