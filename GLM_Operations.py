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

subs=['02','03','04']
brain_flag='MNI'


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
for num in range(len(subs)):

    sub_num=subs[num]

    print('Running sub-0%s...' %sub_num)
    #define the subject
    sub = ('sub-0%s' % sub_num)
    container_path='/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep'
  
    bold_path=os.path.join(container_path,sub,'func/')
    os.chdir(bold_path)
  
    #set up the path to the files and then moved into that directory

    localizer_files=find('*study*bold*.nii.gz',bold_path)
    wholebrain_mask_path=find('*study*mask*.nii.gz',bold_path)
    
    if brain_flag=='MNI':
        pattern = '*MNI*'
        pattern2= '*MNI152NLin2009cAsym*preproc*'
        brain_mask_path = fnmatch.filter(wholebrain_mask_path,pattern)
        study_files = fnmatch.filter(localizer_files,pattern2)
    elif brain_flag=='T1w':
        pattern = '*T1w*'
        pattern2 = '*T1w*preproc*'
        brain_mask_path = fnmatch.filter(wholebrain_mask_path,pattern)
        study_files = fnmatch.filter(localizer_files,pattern2)
        
    brain_mask_path.sort()
    localizer_files.sort()

    gm_mask_path=find('*GM_MNI_mask*',container_path)
    mask=nib.load(gm_mask_path[0])  

    img=nib.concat_images(study_files,axis=3)
      
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
    #clean data ahead of the GLM
    img_clean=clean_img(img,sessions=run_list,t_r=1,detrend=False,standardize='zscore',confounds=study_confounds)
    '''load in the denoised bold data and events file'''
    events = pd.read_csv('/scratch1/06873/zbretton/repclear_dataset/BIDS/task-study_events.tsv',sep='\t')        

    '''initialize and fit the GLM'''
    model = FirstLevelModel(t_r=1,hrf_model='glover',
                            drift_model=None,high_pass=None,mask_img=mask,signal_scaling=False,
                            smoothing_fwhm=6,noise_model='ar1',n_jobs=1,verbose=2,memory='/scratch1/06873/zbretton/nilearn_cache',memory_level=1)

    model.fit(run_imgs=img_clean,events=events,confounds=None)

    '''grab the number of regressors in the model'''
    n_columns = model.design_matrices_[0].shape[-1]

    '''define the contrasts - the order of trial types is stored in model.design_matrices_[0].columns
       pad_contrast() adds 0s to the end of a vector in the case that other regressors are modeled, but not included in the primary contrasts'''
       #order is: maintain, replace, stim, suppress (may need to add in ITI)
    contrasts = {'maintain':             pad_contrast([2,-1,0,-1],  n_columns),
                 'replace':               pad_contrast([-1,2,0,-1],  n_columns),
                 'suppress':               pad_contrast([-1,-1,0,2],  n_columns)}

    '''point to and if necessary create the output folder'''
    out_folder = os.path.join(container_path,sub,'study_lvl1')
    if not os.path.exists(out_folder): os.makedirs(out_folder,exist_ok=True)

    '''compute and save the contrasts'''
    for contrast in contrasts:
        z_map = model.compute_contrast(contrasts[contrast],output_type='z_score')
        nib.save(z_map,os.path.join(out_folder,f'{contrast}_{brain_flag}_zmap.nii.gz'))

####################################
#level 2 GLM
subs=['sub-002','sub-003','sub-004']
contrasts = ['maintain','replace','suppress']

'''point to the save directory'''
out_dir = os.path.join(container_path,'group_model','group_category_lvl2')
if not os.path.exists(out_dir):os.makedirs(out_dir,exist_ok=True)

for contrast in contrasts:
    '''load in the subject maps'''
    maps = [nib.load(os.path.join(container_path,sub,'study_lvl1',f'{contrast}_{brain_flag}_zmap.nii.gz')) for sub in subs]

    '''a simple group mean design'''
    design_matrix = pd.DataFrame([1] * len(maps), columns=['intercept'])
    
    '''initialize and fit the GLM'''
    second_level_model = SecondLevelModel(smoothing_fwhm=None,
                                          mask_img=gm_mask_path[0],
                                          verbose=2,n_jobs=-1)
    second_level_model = second_level_model.fit(maps, design_matrix=design_matrix)
    z_map = second_level_model.compute_contrast(output_type='z_score')

    '''save the group map'''
    nib.save(z_map, os.path.join(out_dir,f'group_{contrast}_{brain_flag}_zmap.nii.gz'))