import nibabel as nib
nib.openers.Opener.default_compresslevel = 6
from nilearn.image import mean_img, get_data, threshold_img, new_img_like, clean_img, load_img
from nilearn.glm.first_level import FirstLevelModel
#from nilearn import plotting
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img
from nilearn import plotting, surface, datasets
import os
import fnmatch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



subs=['02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','20','23','24','25','26']
brain_flag='MNI'


def mkdir(path,local=False):
    if not os.path.exists(path):
        os.makedirs(path)


def pad_contrast(contrast_, n_columns):
    """A small routine to append zeros in contrast vectors"""
    return np.hstack((contrast_, np.zeros(n_columns - len(contrast_))))

def confound_cleaner(confounds):
    COI = ['a_comp_cor_00','a_comp_cor_01','a_comp_cor_02','a_comp_cor_03','a_comp_cor_04','a_comp_cor_05','framewise_displacement','trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']
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
    

container_path='/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep'


#level 1 GLM
for num in range(len(subs)):

    sub_num=subs[num]

    print('Running sub-0%s...' %sub_num)
    #define the subject
    sub = ('sub-0%s' % sub_num)
  
    bold_path=os.path.join(container_path,sub,'func/')
    os.chdir(bold_path)
  
    #set up the path to the files and then moved into that directory

    study_files=find('*study*bold*.nii.gz',bold_path)
    wholebrain_mask_path=find('*study*mask*.nii.gz',bold_path)
    
    if brain_flag=='MNI':
        pattern = '*MNI*'
        pattern2= '*MNI152NLin2009cAsym*preproc*'
        brain_mask_path = fnmatch.filter(wholebrain_mask_path,pattern)
        study_files = fnmatch.filter(study_files,pattern2)
    elif brain_flag=='T1w':
        pattern = '*T1w*'
        pattern2 = '*T1w*preproc*'
        brain_mask_path = fnmatch.filter(wholebrain_mask_path,pattern)
        study_files = fnmatch.filter(study_files,pattern2)
        
    brain_mask_path.sort()
    study_files.sort()
    if brain_flag=='MNI':

        gm_mask_path=os.path.join('/scratch/06873/zbretton/fmriprep/group_%s_GM_mask.nii.gz' % brain_flag)


    gm_mask=nib.load(gm_mask_path)  

    #img=nib.concat_images(study_files,axis=3)

    study_run1=load_img(study_files[0])
    study_run2=load_img(study_files[1])
    study_run3=load_img(study_files[2])
    

    #create list of all nifti files
    img=[study_run1,study_run2,study_run3]


    study_confounds_1=find('*study*1*confounds*.tsv',bold_path)
    study_confounds_2=find('*study*2*confounds*.tsv',bold_path)
    study_confounds_3=find('*study*3*confounds*.tsv',bold_path)


    
    confound_run1 = pd.read_csv(study_confounds_1[0],sep='\t')
    confound_run2 = pd.read_csv(study_confounds_2[0],sep='\t')
    confound_run3 = pd.read_csv(study_confounds_3[0],sep='\t')
        

    confound_run1=confound_cleaner(confound_run1)
    confound_run2=confound_cleaner(confound_run2)
    confound_run3=confound_cleaner(confound_run3)
 
    
    #confounds are all cleaned and organized

    study_confounds=[confound_run1,confound_run2,confound_run3]

    #get run list so I can clean the data across each of the runs
    run1_length=int((img[0].get_fdata().shape[3]))
    run2_length=int((img[1].get_fdata().shape[3]))
    run3_length=int((img[2].get_fdata().shape[3]))


    run1=np.full(run1_length,1)
    run2=np.full(run2_length,2)
    run3=np.full(run3_length,3)


    run_list=np.concatenate((run1,run2,run3))    
    
    #load in the events file
    events = pd.read_csv('/work/06873/zbretton/ls6/repclear_dataset/BIDS/task-study_events.tsv',sep='\t')     
    #this is some code that will split up this tsv into separate dicts per run   
    events_dict={g:d for g, d in events.groupby('run')}

    #this original events file is concatenated via the run times, so need to "normalize" each events file to the start of the run
    # this is done by taking the first value from the onset column and subtracting that from every other onset time
    events_1=pd.DataFrame.from_dict(events_dict[1])
    events_2=pd.DataFrame.from_dict(events_dict[2])
    events_3=pd.DataFrame.from_dict(events_dict[3])


    #this only needs to occur to events 2+ since events 1 is already pegged to the start
    events_2['onset'] -= events_2['onset'].iat[0]  
    events_3['onset'] -= events_3['onset'].iat[0]  


    #removing rest as a modeled condition

    events_list=[events_1,events_2,events_3]      

    '''initialize and fit the GLMs'''
    #the first levels need to be split by stimulus type since they are basically two unique situations 
    model_operations = FirstLevelModel(t_r=1,hrf_model='glover',
                            drift_model=None,high_pass=None,mask_img=gm_mask,signal_scaling=False,
                            smoothing_fwhm=8,noise_model='ar1',n_jobs=1,verbose=2,memory='/scratch/06873/zbretton/nilearn_cache',memory_level=1)

    model_operations.fit(run_imgs=img,events=events_list,confounds=study_confounds)  

    '''grab the number of regressors in the model'''
    n_columns = model_operations.design_matrices_[0].shape[-1]
    #this is the same for both conditions
    #create background image for reports:
    mean_img_ = mean_img(img)

    '''define the contrasts - the order of trial types is stored in model.design_matrices_[0].columns
       pad_contrast() adds 0s to the end of a vector in the case that other regressors are modeled, but not included in the primary contrasts'''

    contrasts = {'maintain':             pad_contrast([2,-1,0,-1],  n_columns),
                 'replace':               pad_contrast([-1,2,0,-1],  n_columns),
                 'suppress':               pad_contrast([-1,-1,0,2],  n_columns),
                 'item_manipulation':       pad_contrast([-2,1,0,1],  n_columns),
                 'suppress_v_maintain':     pad_contrast([-1,0,0,1],  n_columns)}

    '''point to and if necessary create the output folder'''
    out_folder = os.path.join(container_path,sub,'study_lvl1_%s' % brain_flag)
    if not os.path.exists(out_folder): os.makedirs(out_folder,exist_ok=True)

    '''compute and save the contrasts'''
    for contrast in contrasts:   
        z_map_s = model_operations.compute_contrast(contrasts[contrast],output_type='z_score')

        nib.save(z_map_s,os.path.join(out_folder,f'{contrast}_{brain_flag}_zmap.nii.gz'))

        t_map_s = model_operations.compute_contrast(contrasts[contrast],stat_type='t',output_type='stat')

        nib.save(t_map_s,os.path.join(out_folder,f'{contrast}_{brain_flag}_tmap.nii.gz'))  

        file_data_s = model_operations.generate_report(contrasts[contrast],bg_img=mean_img_)

        file_data_s.save_as_html(os.path.join(out_folder,f"{contrast}_{brain_flag}_report.html"))         

####################################
#level 2 GLM
subs=['sub-002','sub-003','sub-004','sub-005','sub-006','sub-007','sub-008','sub-009','sub-010','sub-011','sub-012','sub-013','sub-014','sub-015','sub-016','sub-017','sub-018','sub-020','sub-023','sub-024','sub-025','sub-026']
contrasts = ['maintain','replace','suppress','item_manipulation','suppress_v_maintain']

'''point to the save directory'''
out_dir = os.path.join(container_path,'group_model','group_operation_lvl2')
if not os.path.exists(out_dir):os.makedirs(out_dir,exist_ok=True)

for contrast in contrasts:
    '''load in the subject maps'''
    maps = [nib.load(os.path.join(container_path,sub,'study_lvl1_%s' % brain_flag,f'{contrast}_{brain_flag}_tmap.nii.gz')) for sub in subs]

    '''a simple group mean design'''
    design_matrix = pd.DataFrame([1] * len(maps), columns=['intercept'])
    
    '''initialize and fit the GLM'''
    second_level_model = SecondLevelModel(smoothing_fwhm=None,
                                          mask_img='/scratch/06873/zbretton/fmriprep/group_MNI_GM_mask.nii.gz',
                                          verbose=2,n_jobs=-1)
    second_level_model = second_level_model.fit(maps, design_matrix=design_matrix)
    t_map = second_level_model.compute_contrast(second_level_stat_type='t',output_type='stat')

    '''save the group map'''
    nib.save(t_map, os.path.join(out_dir,f'group_{contrast}_{brain_flag}_tmap.nii.gz'))
    #now I want to treshold this to focus on the important clusters:
    thresholded_map, _ = threshold_stats_img(
        t_map,
        alpha=0.05,
        threshold=3.02,
        height_control=None,
        cluster_threshold=73
        )
    file_data = second_level_model.generate_report(contrasts='intercept',alpha=0.05,threshold=3.02,height_control=None,cluster_threshold=73)
    file_data.save_as_html(os.path.join(out_dir,f"group+{contrast}_{brain_flag}_report.html"))     
    #use this threshold to look at the second-level results
    nib.save(thresholded_map, os.path.join(out_dir,f'group+{contrast}_{brain_flag}_thresholded_tmap.nii.gz'))
    del thresholded_map, t_map, second_level_model, maps