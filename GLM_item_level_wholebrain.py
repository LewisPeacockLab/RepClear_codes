import nibabel as nib
nib.openers.Opener.default_compresslevel = 6
from nilearn.image import mean_img, get_data, threshold_img, new_img_like, clean_img, load_img
from nilearn.glm.first_level import FirstLevelModel
#from nilearn import plotting
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img
import os
import fnmatch
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


subs=['02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','20','23','24','25','26']
brain_flag='T1w'

#code for the item level weighting for faces and scenes


def mkdir(path,local=False):
    if not os.path.exists(path):
        os.makedirs(path)


def pad_contrast(contrast_, n_columns):
    """A small routine to append zeros in contrast vectors"""
    return np.hstack((contrast_, np.zeros(n_columns - len(contrast_))))

def confound_cleaner(confounds):
    COI = ['a_comp_cor_00','a_comp_cor_01','a_comp_cor_02','a_comp_cor_03','a_comp_cor_05','framewise_displacement','trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']
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


#LSA
def GLM_item_level(subID):
    print('Running sub-0%s...' %subID)
    #define the subject
    sub = ('sub-0%s' % subID)
    container_path='/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep'
  
    bold_path=os.path.join(container_path,sub,'func/')
    os.chdir(bold_path)
  
    #set up the path to the files and then moved into that directory

    localizer_files=find('*preremoval*bold*.nii.gz',bold_path)
    wholebrain_mask_path=find('*preremoval*mask*.nii.gz',bold_path)
    
    if brain_flag=='MNI':
        pattern = '*MNI*'
        pattern2= '*MNI152NLin2009cAsym*preproc*'
        brain_mask_path = fnmatch.filter(wholebrain_mask_path,pattern)
        localizer_files = fnmatch.filter(localizer_files,pattern2)
    elif brain_flag=='T1w':
        pattern = '*T1w*'
        pattern2 = '*T1w*preproc*'
        brain_mask_path = fnmatch.filter(wholebrain_mask_path,pattern)
        localizer_files = fnmatch.filter(localizer_files,pattern2)
        
    brain_mask_path.sort()
    localizer_files.sort()

    if brain_flag=='MNI':

        vtc_mask_path=os.path.join('/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/group_MNI_VTC_mask.nii.gz') #VTC

    else:
        vtc_mask_path=os.path.join('/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/',sub,'new_mask','VVS_preremoval_%s_mask.nii.gz' % brain_flag)


    vtc_mask=nib.load(vtc_mask_path)   


    localizer_run1=load_img(localizer_files[0])
    localizer_run2=load_img(localizer_files[1])
    localizer_run3=load_img(localizer_files[2])
    localizer_run4=load_img(localizer_files[3])
    localizer_run5=load_img(localizer_files[4])
    localizer_run6=load_img(localizer_files[5]) 

    img=[localizer_run1,localizer_run2,localizer_run3,localizer_run4,localizer_run5,localizer_run6]
    #to be used to filter the data
    #First we are removing the confounds
    #get all the folders within the bold path
    #confound_folders=[x[0] for x in os.walk(bold_path)]
    localizer_confounds_1=find('*preremoval*1*confounds*.tsv',bold_path)
    localizer_confounds_2=find('*preremoval*2*confounds*.tsv',bold_path)
    localizer_confounds_3=find('*preremoval*3*confounds*.tsv',bold_path)
    localizer_confounds_4=find('*preremoval*4*confounds*.tsv',bold_path)
    localizer_confounds_5=find('*preremoval*5*confounds*.tsv',bold_path)
    localizer_confounds_6=find('*preremoval*6*confounds*.tsv',bold_path)

    
    confound_run1 = pd.read_csv(localizer_confounds_1[0],sep='\t')
    confound_run2 = pd.read_csv(localizer_confounds_2[0],sep='\t')
    confound_run3 = pd.read_csv(localizer_confounds_3[0],sep='\t')
    confound_run4 = pd.read_csv(localizer_confounds_4[0],sep='\t')
    confound_run5 = pd.read_csv(localizer_confounds_5[0],sep='\t')
    confound_run6 = pd.read_csv(localizer_confounds_6[0],sep='\t')            

    confound_run1=confound_cleaner(confound_run1)
    confound_run2=confound_cleaner(confound_run2)
    confound_run3=confound_cleaner(confound_run3)
    confound_run4=confound_cleaner(confound_run4)
    confound_run5=confound_cleaner(confound_run5)
    confound_run6=confound_cleaner(confound_run6)   
    
    localizer_confounds=[confound_run1,confound_run2,confound_run3,confound_run4,confound_run5,confound_run6]

    #get run list so I can clean the data across each of the runs
    run1_length=int((img[0].get_fdata().shape[3]))
    run2_length=int((img[1].get_fdata().shape[3]))
    run3_length=int((img[2].get_fdata().shape[3]))
    run4_length=int((img[3].get_fdata().shape[3]))
    run5_length=int((img[4].get_fdata().shape[3]))
    run6_length=int((img[5].get_fdata().shape[3]))

    run1=np.full(run1_length,1)
    run2=np.full(run2_length,2)
    run3=np.full(run3_length,3)
    run4=np.full(run4_length,4)
    run5=np.full(run5_length,5)    
    run6=np.full(run6_length,6)    

    run_list=np.concatenate((run1,run2,run3,run4,run5,run6))  
    #clean data ahead of the GLM
    #img_clean=clean_img(img,sessions=run_list,t_r=1,detrend=False,standardize='zscore')
    '''load in the denoised bold data and events file'''
    events = pd.read_csv('/work/06873/zbretton/ls6/repclear_dataset/BIDS/task-preremoval_events.tsv',sep='\t')     
    #this is some code that will split up this tsv into separate dicts per run   
    events_dict={g:d for g, d in events.groupby('run')}

    #this original events file is concatenated via the run times, so need to "normalize" each events file to the start of the run
    # this is done by taking the first value from the onset column and subtracting that from every other onset time
    events_1=pd.DataFrame.from_dict(events_dict[1])
    events_2=pd.DataFrame.from_dict(events_dict[2])
    events_3=pd.DataFrame.from_dict(events_dict[3])
    events_4=pd.DataFrame.from_dict(events_dict[4])
    events_5=pd.DataFrame.from_dict(events_dict[5])
    events_6=pd.DataFrame.from_dict(events_dict[6])

    #this only needs to occur to events 2+ since events 1 is already pegged to the start
    events_2['onset'] -= events_2['onset'].iat[0]  
    events_3['onset'] -= events_3['onset'].iat[0]  
    events_4['onset'] -= events_4['onset'].iat[0]  
    events_5['onset'] -= events_5['onset'].iat[0]  
    events_6['onset'] -= events_6['onset'].iat[0]  

    #removing rest as a modeled condition
    events_1=events_1[events_1['trial_type']=='face'].reset_index(drop=True)
    events_2=events_2[events_2['trial_type']=='face'].reset_index(drop=True)
    events_3=events_3[events_3['trial_type']=='scene'].reset_index(drop=True)
    events_4=events_4[events_4['trial_type']=='scene'].reset_index(drop=True)
    events_5=events_5[events_5['trial_type']=='scene'].reset_index(drop=True)
    events_6=events_6[events_6['trial_type']=='scene'].reset_index(drop=True)

    events_list=[events_1,events_2,events_3,events_4,events_5,events_6]      


    #now will need to create a loop where I iterate over the face & scene indexes
    #I then relabel that trial of the face or scene as "face_trial#" or "scene_trial#" and then label rest and all other trials as "other"
    #I can either do this in one loop, or two consecutive


    '''initialize the face GLM'''
    model_face = FirstLevelModel(t_r=1,hrf_model='glover',
                            drift_model=None,high_pass=None,mask_img=vtc_mask,signal_scaling=False,
                            smoothing_fwhm=8,noise_model='ar1',n_jobs=1,verbose=2,memory='/scratch/06873/zbretton/nilearn_cache',memory_level=1)
    
    '''initialize the scene GLM'''
    model_scene = FirstLevelModel(t_r=1,hrf_model='glover',
                            drift_model=None,high_pass=None,mask_img=vtc_mask,signal_scaling=False,
                            smoothing_fwhm=8,noise_model='ar1',n_jobs=1,verbose=2,memory='/scratch/06873/zbretton/nilearn_cache',memory_level=1)    

    #I want to ensure that "trial" is the # of face (e.g., first instance of "face" is trial=1, second is trial=2...)
    face_trials=events.trial_type.value_counts().face
    scene_trials=events.trial_type.value_counts().scene

    #create background image for reports:
    mean_img_ = mean_img(img)
    #so this will give us a sense of total trials for these two conditions
        #next step is to then get the index of these conditions, and then use the trial# to iterate through the indexes properly

    for trial in (range(face_trials)):
        if trial<30:
            events_list[0].loc[trial,'trial_type']=('face_trial%s' % (trial+1))
        elif trial>=30:
            events_list[1].loc[trial-30,'trial_type']=('face_trial%s' % (trial+1))       

    for trial in (range(scene_trials)):
        if trial<30:
            events_list[2].loc[trial,'trial_type']=('scene_trial%s' % (trial+1))
        elif (trial>=30) & (trial<60):
            events_list[3].loc[trial-30,'trial_type']=('scene_trial%s' % (trial+1))
        elif (trial>=60) & (trial<90):
            events_list[4].loc[trial-60,'trial_type']=('scene_trial%s' % (trial+1))
        elif (trial>=90):
            events_list[5].loc[trial-90,'trial_type']=('scene_trial%s' % (trial+1))            

    for trial in (range(face_trials)):
        model_face.fit(run_imgs=img[:2],events=events_list[:2],confounds=localizer_confounds[:2])

        '''grab the number of regressors in the model'''
        n_columns = model_face.design_matrices_[0].shape[-1]

        #since the columns are not sorted as expected, I will need to located the index of the current trial to place the contrast properly


        '''define the contrasts - the order of trial types is stored in model.design_matrices_[0].columns
           pad_contrast() adds 0s to the end of a vector in the case that other regressors are modeled, but not included in the primary contrasts'''
           #order is: trial1...trialN
        #bascially create an array the length of all the items
        item_contrast=[np.full(n_columns,0),np.full(n_columns,0)] #start with an array of 0's
        item_contrast[0][model_face.design_matrices_[0].columns.str.match('face_trial')]=-1 #find all the indices of face_trial and set to -1
        item_contrast[1][model_face.design_matrices_[1].columns.str.match('face_trial')]=-1 #find all the indices of face_trial and set to -1

        if trial<30:
            item_contrast[0][model_face.design_matrices_[0].columns.get_loc('face_trial%s' % (trial+1))]=(face_trials-1) #now find our trial of interest and set it equal to the sum of the rest of the contrasts
        elif trial>=30:
            item_contrast[1][model_face.design_matrices_[1].columns.get_loc('face_trial%s' % (trial+1))]=(face_trials-1) #now find our trial of interest and set it equal to the sum of the rest of the contrasts

        contrasts = {'face_trial%s' % (trial+1): item_contrast}

        '''point to and if necessary create the output folder'''
        if brain_flag=='MNI':
            out_folder = os.path.join(container_path,sub,'preremoval_item_level_MNI')
            if not os.path.exists(out_folder): os.makedirs(out_folder,exist_ok=True)
        else:
            out_folder = os.path.join(container_path,sub,'preremoval_item_level_T1w')
            if not os.path.exists(out_folder): os.makedirs(out_folder,exist_ok=True)
        #as of now it is labeling the trial estimates by the trial number, which is helpful since I can look at their individual design matricies to see which stim that is
        #but another way could be to load in the list for that sub right here, grab the number or name of that stim from the trial index and use that to save the name

        '''compute and save the contrasts'''
        for contrast in contrasts:
            z_map = model_face.compute_contrast(contrasts[contrast],output_type='z_score')
            nib.save(z_map,os.path.join(out_folder,f'{contrast}_{brain_flag}_full_zmap.nii.gz'))
            t_map = model_face.compute_contrast(contrasts[contrast],stat_type='t',output_type='stat')
            nib.save(t_map,os.path.join(out_folder,f'{contrast}_{brain_flag}_full_tmap.nii.gz'))  
            file_data = model_face.generate_report(contrasts,bg_img=mean_img_)
            file_data.save_as_html(os.path.join(out_folder,f"{contrast}_{brain_flag}_full_report.html")) 

        del item_contrast

    for trial in (range(scene_trials)):

        model_scene.fit(run_imgs=img[2:],events=events_list[2:],confounds=localizer_confounds[2:])

        '''grab the number of regressors in the model'''
        n_columns = model_scene.design_matrices_[0].shape[-1]

        '''define the contrasts - the order of trial types is stored in model.design_matrices_[0].columns
           pad_contrast() adds 0s to the end of a vector in the case that other regressors are modeled, but not included in the primary contrasts'''
           #order is: trial1...trialN

        item_contrast=[np.full(n_columns,0),np.full(n_columns,0),np.full(n_columns,0),np.full(n_columns,0)] #start with an array of 0's
        item_contrast[0][model_scene.design_matrices_[0].columns.str.match('scene_trial')]=-1 #find all the indices of scene_trial and set to -1
        item_contrast[1][model_scene.design_matrices_[1].columns.str.match('scene_trial')]=-1 #find all the indices of scene_trial and set to -1
        item_contrast[2][model_scene.design_matrices_[2].columns.str.match('scene_trial')]=-1 #find all the indices of scene_trial and set to -1
        item_contrast[3][model_scene.design_matrices_[3].columns.str.match('scene_trial')]=-1 #find all the indices of scene_trial and set to -1


        if trial<30:
            item_contrast[0][model_scene.design_matrices_[0].columns.get_loc('scene_trial%s' % (trial+1))]=(scene_trials-1) #now find our trial of interest and set it equal to the sum of the rest of the contrasts
        elif (trial>=30) & (trial<60):
            item_contrast[1][model_scene.design_matrices_[1].columns.get_loc('scene_trial%s' % (trial+1))]=(scene_trials-1) #now find our trial of interest and set it equal to the sum of the rest of the contrasts
        elif (trial>=60) & (trial<90):
            item_contrast[2][model_scene.design_matrices_[2].columns.get_loc('scene_trial%s' % (trial+1))]=(scene_trials-1) #now find our trial of interest and set it equal to the sum of the rest of the contrasts
        elif (trial>=90):
            item_contrast[3][model_scene.design_matrices_[3].columns.get_loc('scene_trial%s' % (trial+1))]=(scene_trials-1) #now find our trial of interest and set it equal to the sum of the rest of the contrasts

        contrasts = {'scene_trial%s' % (trial+1): item_contrast}

        '''point to and if necessary create the output folder'''
        if brain_flag=='MNI':
            out_folder = os.path.join(container_path,sub,'preremoval_item_level_MNI')
            if not os.path.exists(out_folder): os.makedirs(out_folder,exist_ok=True)
        else:
            out_folder = os.path.join(container_path,sub,'preremoval_item_level_T1w')
            if not os.path.exists(out_folder): os.makedirs(out_folder,exist_ok=True)

        #as of now it is labeling the trial estimates by the trial number, which is helpful since I can look at their individual design matricies to see which stim that is
        #but another way could be to load in the list for that sub right here, grab the number or name of that stim from the trial index and use that to save the name

        '''compute and save the contrasts'''
        for contrast in contrasts:
            z_map = model_scene.compute_contrast(contrasts[contrast],output_type='z_score')
            nib.save(z_map,os.path.join(out_folder,f'{contrast}_{brain_flag}_full_zmap.nii.gz'))
            t_map = model_scene.compute_contrast(contrasts[contrast],stat_type='t',output_type='stat')
            nib.save(t_map,os.path.join(out_folder,f'{contrast}_{brain_flag}_full_tmap.nii.gz'))  
            file_data = model_scene.generate_report(contrasts,bg_img=mean_img_)
            file_data.save_as_html(os.path.join(out_folder,f"{contrast}_{brain_flag}_full_report.html"))
        #make sure to clear the item constrast to make sure we dont carry it over in to the next trial     
        del item_contrast

Parallel(n_jobs=4)(delayed(GLM_item_level)(i) for i in subs)


