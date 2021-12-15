import nibabel as nib
nib.openers.Opener.default_compresslevel = 6
from nilearn.image import mean_img, get_data, threshold_img, new_img_like, clean_img, concat_imgs
from nilearn.glm.first_level import FirstLevelModel
#from nilearn import plotting
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img
import os
import fnmatch
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


subs=['61','69','77']
brain_flag='MNI'

#code for the item level weighting for faces and scenes


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


#level 1 GLM
def item_level_1(subID):
    print('Running sub-0%s...' %subID)
    #define the subject
    sub = ('sub-0%s' % subID)
    container_path='/scratch1/06873/zbretton/clearmem/'
  
    bold_path=os.path.join(container_path,sub,'func/')
    os.chdir(bold_path)
  
    #set up the path to the files and then moved into that directory

    localizer_files=find('*localizer*bold*.nii.gz',bold_path)
    wholebrain_mask_path=find('*localizer*mask*.nii.gz',bold_path)
    
    if brain_flag=='MNI':
        pattern = '*MNI*'
        pattern2= '*MNI152NLin2009cAsym*preproc*resized*'
        brain_mask_path = fnmatch.filter(wholebrain_mask_path,pattern)
        localizer_files = fnmatch.filter(localizer_files,pattern2)

    brain_mask_path.sort()
    localizer_files.sort()
    face_mask_path=os.path.join('/scratch1/06873/zbretton/clearmem/group_model/group_category_lvl2/','group_face_%s_mask.nii.gz' % brain_flag)
    scene_mask_path=os.path.join('/scratch1/06873/zbretton/clearmem/group_model/group_category_lvl2/','group_scene_%s_mask.nii.gz' % brain_flag)    
    face_mask=nib.load(face_mask_path)   
    scene_mask=nib.load(scene_mask_path)
    vtc_mask_path=os.path.join('/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/group_%s_VTC_mask.nii.gz' % brain_flag)
        
    vtc_mask=nib.load(vtc_mask_path)       
    
    #load in category mask that was created from the first GLM  

    img=concat_imgs(localizer_files,memory='/scratch1/06873/zbretton/nilearn_cache')
    #to be used to filter the data
    #First we are removing the confounds
    #get all the folders within the bold path
    #confound_folders=[x[0] for x in os.walk(bold_path)]
    localizer_confounds_1=find('*localizer*1*confounds*.tsv',bold_path)
    localizer_confounds_2=find('*localizer*2*confounds*.tsv',bold_path)
    localizer_confounds_3=find('*localizer*3*confounds*.tsv',bold_path)
    localizer_confounds_4=find('*localizer*4*confounds*.tsv',bold_path)
    localizer_confounds_5=find('*localizer*5*confounds*.tsv',bold_path)

    confound_run1 = pd.read_csv(localizer_confounds_1[0],sep='\t')
    confound_run2 = pd.read_csv(localizer_confounds_2[0],sep='\t')
    confound_run3 = pd.read_csv(localizer_confounds_3[0],sep='\t')
    confound_run4 = pd.read_csv(localizer_confounds_4[0],sep='\t')
    confound_run5 = pd.read_csv(localizer_confounds_5[0],sep='\t')

    confound_run1=confound_cleaner(confound_run1)
    confound_run2=confound_cleaner(confound_run2)
    confound_run3=confound_cleaner(confound_run3)
    confound_run4=confound_cleaner(confound_run4)
    confound_run5=confound_cleaner(confound_run5)
    
    localizer_confounds=pd.concat([confound_run1,confound_run2,confound_run3,confound_run4,confound_run5], ignore_index=False)

    #get run list so I can clean the data across each of the runs
    run1_length=int((img.get_fdata().shape[3])/5)
    run2_length=int((img.get_fdata().shape[3])/5)
    run3_length=int((img.get_fdata().shape[3])/5)
    run4_length=int((img.get_fdata().shape[3])/5)
    run5_length=int((img.get_fdata().shape[3])/5)

    run1=np.full(run1_length,1)
    run2=np.full(run2_length,2)
    run3=np.full(run3_length,3)
    run4=np.full(run4_length,4)
    run5=np.full(run5_length,5)    

    run_list=np.concatenate((run1,run2,run3,run4,run5))    
    #clean data ahead of the GLM
    img_clean=clean_img(img,sessions=run_list,t_r=1,detrend=False,standardize='zscore',mask_img=vtc_mask)
    del img
    '''load in the denoised bold data and events file'''
    events = pd.read_csv('/scratch1/06873/zbretton/clearmem/localizer_events_item_sampled.csv',sep=',')
    #now will need to create a loop where I iterate over the face & scene indexes
    #I then relabel that trial of the face or scene as "face_trial#" or "scene_trial#" and then label rest and all other trials as "other"
    #I can either do this in one loop, or two consecutive

    #this has too much info so we need to only take the important columns
    events=events[['onset','duration','trial_type']]

    '''initialize the GLM with vtc mask'''
    model = FirstLevelModel(t_r=0.46,hrf_model='glover',
                            drift_model=None,high_pass=None,mask_img=vtc_mask,signal_scaling=False,
                            smoothing_fwhm=8,noise_model='ar1',n_jobs=1,verbose=2,memory='/scratch1/06873/zbretton/nilearn_cache',memory_level=1)   

    #I want to ensure that "trial" is the # of face (e.g., first instance of "face" is trial=1, second is trial=2...)
    # face_trials=events.trial_type.value_counts().face
    # scene_trials=events.trial_type.value_counts().scene
    #so this will give us a sense of total trials for these two conditions
        #next step is to then get the index of these conditions, and then use the trial# to iterate through the indexes properly

    temp_events=events.copy() #copy the original events list, this list is set up where each "trial_type" is the image ID, so we dont need trial sorting

    # face_index=[i for i, n in enumerate(temp_events['trial_type']) if n == 'face'] #this will find the nth occurance of a desired value in the list
    # scene_index=[i for i, n in enumerate(temp_events['trial_type']) if n == 'scene']#this will find the nth occurance of a desired value in the list    
    # for trial in (range(len(face_index))):
    #     #this is a rough idea how I will create a temporary new version of the events file to use for the LSS
    #     temp_events.loc[face_index[trial],'trial_type']=('face_trial%s' % (trial+1))
    # for trial in (range(len(scene_index))):    
    #     temp_events.loc[scene_index[trial],'trial_type']=('scene_trial%s' % (trial+1))

    #using this loop to make sure that I am only modeling the first trial, since that would line up with my repclear pipeline
    for image_id in np.unique(events['trial_type'].values):
        img_indx=np.where(temp_events['trial_type']==image_id)[0]
        temp_events['trial_type'][img_indx[1:]]=0


    model.fit(run_imgs=img_clean,events=temp_events,confounds=localizer_confounds)

    '''grab the number of regressors in the model'''
    n_columns = model.design_matrices_[0].shape[-1]

    #since the columns are not sorted as expected, I will need to located the index of the current trial to place the contrast properly

    '''define the contrasts - the order of trial types is stored in model.design_matrices_[0].columns
       pad_contrast() adds 0s to the end of a vector in the case that other regressors are modeled, but not included in the primary contrasts'''
       #order is: trial1...trialN
    #bascially create an array the length of all the items
    contrasts={}
    for image_id in np.unique(events['trial_type'].values):
        if image_id==0:
            print()
        else:
            item_contrast=np.full(np.unique(events['trial_type'].values).size,-1) #start with an array of 0's
            item_contrast[0]=0
            item_contrast[model.design_matrices_[0].columns[:55]==image_id]=53 #find all the indices of image_id and set to 54

            contrasts[image_id] = pad_contrast(item_contrast,  n_columns)

    '''point to and if necessary create the output folder'''
    out_folder = os.path.join(container_path,sub,'localizer_item_level')
    if not os.path.exists(out_folder): os.makedirs(out_folder,exist_ok=True)

    #as of now it is labeling the trial estimates by the trial number, which is helpful since I can look at their individual design matricies to see which stim that is
    #but another way could be to load in the list for that sub right here, grab the number or name of that stim from the trial index and use that to save the name

    '''compute and save the contrasts'''
    for contrast in contrasts:
        z_map = model.compute_contrast(contrasts[contrast],output_type='z_score')
        nib.save(z_map,os.path.join(out_folder,f'item{contrast}_{brain_flag}_zmap.nii.gz'))
        t_map = model.compute_contrast(contrasts[contrast],stat_type='t',output_type='stat')
        nib.save(t_map,os.path.join(out_folder,f'item{contrast}_{brain_flag}_tmap.nii.gz'))  
        file_data = model.generate_report(contrasts[contrast])
        file_data.save_as_html(os.path.join(out_folder,f"item{contrast}_{brain_flag}_report.html")) 

Parallel(n_jobs=len(subs))(delayed(item_level_1)(i) for i in subs)
