import nibabel as nib
nib.openers.Opener.default_compresslevel = 6
from nilearn.image import mean_img, get_data, threshold_img, new_img_like, clean_img
from nilearn.glm.first_level import FirstLevelModel
#from nilearn import plotting
from nilearn.glm.second_level import SecondLevelModel
import os
import fnmatch
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


subs=['61','69','77']
brain_flag='MNI'

#code for the LSS of the Localizer data


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
def LSS_Category(subID):

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
    vtc_mask_path=os.path.join('/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/group_MNI_VTC_mask.nii.gz')
    vtc_mask=nib.load(vtc_mask_path)   

    
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
    '''load in the denoised bold data and events file'''
    events = pd.read_csv('/scratch1/06873/zbretton/clearmem/localizer_events_item_sampled.csv',sep=',')

    #this has too much info so we need to only take the important columns
    events=events[['onset','duration','trial_type']]

    #now will need to create a loop where I iterate over the face & scene indexes
    #I then relabel that trial of the face or scene as "face_trial#" or "scene_trial#" and then label rest and all other trials as "other"
    #I can either do this in one loop, or two consecutive

    '''initialize the GLM'''
    model = FirstLevelModel(t_r=0.46,hrf_model='glover',
                            drift_model=None,high_pass=None,mask_img=vtc_mask,signal_scaling=False,
                            smoothing_fwhm=8,noise_model='ar1',n_jobs=1,verbose=2,memory='/scratch1/06873/zbretton/nilearn_cache',memory_level=1)
    #I want to ensure that "trial" is the # of face (e.g., first instance of "face" is trial=1, second is trial=2...)
    # face_trials=events.trial_type.value_counts().face
    # scene_trials=events.trial_type.value_counts().scene
    #so this will give us a sense of total trials for these two conditions
        #next step is to then get the index of these conditions, and then use the trial# to iterate through the indexes properly

    for image_id in np.unique(events['trial_type'].values):
        #this is a rough idea how I will create a temporary new version of the events file to use for the LSS
        if image_id>0:
            temp_events=events.copy()
            img_indx=np.where(temp_events['trial_type']==image_id)[0] #find first occurance
            change_df=temp_events.index.isin(img_indx[:1])
            temp_events['trial_type'][~change_df]='other' #set the other trials to other
            temp_events['trial_type'][change_df]=('image%s' % image_id) #set our trial of interest to its image#

            model.fit(run_imgs=img_clean,events=temp_events,confounds=localizer_confounds)

            '''grab the number of regressors in the model'''
            n_columns = model.design_matrices_[0].shape[-1]

            '''define the contrasts - the order of trial types is stored in model.design_matrices_[0].columns
               pad_contrast() adds 0s to the end of a vector in the case that other regressors are modeled, but not included in the primary contrasts'''
            #set up the array to be used to feed into the pad_constrast function below
            contrast_array=[0,0]
            #these two lines then look for the column with the string that matches the input, and sets that value to 1 and -1 respectively
            #this means that regardless of where the 'scene_trial' or 'other' are in the design, it assigns the contrast properly
            contrast_array[np.where(model.design_matrices_[0].columns[:2].str.match('image'))[0][0]]=1
            contrast_array[np.where(model.design_matrices_[0].columns[:2].str.match('other'))[0][0]]=-1

            contrasts = {'image%s' % (image_id): pad_contrast(contrast_array,  n_columns)}

            '''point to and if necessary create the output folder'''
            out_folder = os.path.join(container_path,sub,'localizer_LSS_lvl1')
            if not os.path.exists(out_folder): os.makedirs(out_folder,exist_ok=True)

            #as of now it is labeling the trial estimates by the trial number, which is helpful since I can look at their individual design matricies to see which stim that is
            #but another way could be to load in the list for that sub right here, grab the number or name of that stim from the trial index and use that to save the name

            '''compute and save the contrasts'''
            for contrast in contrasts:
                t_map = model.compute_contrast(contrasts[contrast],stat_type='t',output_type='stat')
                nib.save(t_map,os.path.join(out_folder,f'{contrast}_{brain_flag}_tmap.nii.gz'))  
                file_data = model.generate_report(contrasts[contrast])
                file_data.save_as_html(os.path.join(out_folder,f"{contrast}_{brain_flag}_report.html"))  
            del temp_events, contrast_array, change_df
Parallel(n_jobs=len(subs))(delayed(LSS_Category)(i) for i in subs)