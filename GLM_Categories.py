import nibabel as nib
nib.openers.Opener.default_compresslevel = 6
from nilearn.image import mean_img, get_data, threshold_img, new_img_like, clean_img
from nilearn.glm.first_level import FirstLevelModel
#from nilearn import plotting
from nilearn.glm.second_level import SecondLevelModel
import os
import fnmatch
import numpy as np

subs=['02','03','04']
brain_flag='T1w'


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


for num in range(len(subs)):

    sub_num=subs[num]

    print('Running sub-0%s...' %sub_num)
    #define the subject
    sub = ('sub-0%s' % sub_num)
    container_path='/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep'
  
    bold_path=os.path.join(container_path,sub,'func/')
    os.chdir(bold_path)
  
    #set up the path to the files and then moved into that directory
    
    #find the proper nii.gz files
    def find(pattern, path): #find the pattern we're looking for
        result = []
        for root, dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root, name))
            return result

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
    vtc_mask_path=os.path.join('/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/',sub,'new_mask','VVS_preremoval_%s_mask.nii.gz' % brain_flag)
    
        
    vtc_mask=nib.load(vtc_mask_path)   

    img=nib.concat_images(localizer_files,axis=3)

    # localizer_run1=nib.load(localizer_files[0])
    # localizer_run2=nib.load(localizer_files[1])
    # localizer_run3=nib.load(localizer_files[2])
    # localizer_run4=nib.load(localizer_files[3])
    # localizer_run5=nib.load(localizer_files[4])
    # localizer_run6=nib.load(localizer_files[5]) 

      
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
    
    localizer_confounds=pd.concat([confound_run1,confound_run2,confound_run3,confound_run4,confound_run5,confound_run6], ignore_index=False)  

    #get run list so I can clean the data across each of the runs
    run1_length=int((img.get_data().shape[3])/6)
    run2_length=int((img.get_data().shape[3])/6)
    run3_length=int((img.get_data().shape[3])/6)
    run4_length=int((img.get_data().shape[3])/6)
    run5_length=int((img.get_data().shape[3])/6)
    run6_length=int((img.get_data().shape[3])/6)

    run1=np.full(run1_length,1)
    run2=np.full(run2_length,2)
    run3=np.full(run3_length,3)
    run4=np.full(run4_length,4)
    run5=np.full(run5_length,5)    
    run6=np.full(run6_length,6)    

    run_list=np.concatenate((run1,run2,run3,run4,run5,run6))    
    #clean data ahead of the GLM
    img_clean=clean_img(img,sessions=run_list,t_r=1,detrend=False,standardize='zscore',confounds=localizer_confounds)
    '''load in the denoised bold data and events file'''
    events = pd.read_csv(os.path.join(container_path,'task-preremoval_events.tsv'),sep='\t')        

    '''initialize and fit the GLM'''
    model = FirstLevelModel(t_r=1,slice_time_ref=.5,hrf_model='glover',
                            drift_model=None,high_pass=None,mask_img=vtc_mask,signal_scaling=False,
                            smoothing_fwhm=6,noise_model='ar1',n_jobs=1,verbose=2,memory='/scratch1/06873/zbretton/nilearn_cache',memory_level=1)

    model.fit(run_imgs=img_clean,events=events,confounds=None)

    '''grab the number of regressors in the model'''
    n_columns = model.design_matrices_[0].shape[-1]

    '''define the contrasts - the order of trial types is stored in model.design_matrices_[0].columns
       pad_contrast() adds 0s to the end of a vector in the case that other regressors are modeled, but not included in the primary contrasts'''

    contrasts = {'faces':             pad_contrast([1,-1],  n_columns),
                 'scenes':               pad_contrast([-1,1],  n_columns)}

    '''point to and if necessary create the output folder'''
    out_folder = os.path,join(container_path,sub,'preremoval_lvl1')
    if not os.path.exists(out_folder): os.makedirs(out_folder,exist_ok=True)

    '''compute and save the contrasts'''
    for contrast in contrasts:
        z_map = model.compute_contrast(contrasts[contrast],output_type='z_score')
        nib.save(z_map,os.path.join(out_folder,f'{contrast}_zmap.nii.gz'))


def lvl2(subs=[],ses=None,task=None):

    contrasts = ['faces','scenes']

    '''point to the save directory'''
    out_dir = path(bids_dir,'derivatives','group_model',f'ses-{ses}_task-{task}_lvl2')
    if not os.path.exists(out_dir):os.makedirs(out_dir,exist_ok=True)

    for contrast in contrasts:
        '''load in the subject maps'''
        maps = [nib.load(path(bids_meta(sub).model,f'ses-{ses}_task-{task}_lvl1',f'{contrast}_zmap.nii.gz')) for sub in subs]

        '''a simple group mean design'''
        design_matrix = pd.DataFrame([1] * len(maps), columns=['intercept'])
        
        '''initialize and fit the GLM'''
        second_level_model = SecondLevelModel(smoothing_fwhm=None,
                                              mask_img='/scratch1/05426/ach3377/standard/MNI152NLin2009cAsym_T1_2mm_brain_mask.nii.gz',
                                              verbose=2,n_jobs=-1)
        second_level_model = second_level_model.fit(maps, design_matrix=design_matrix)
        z_map = second_level_model.compute_contrast(output_type='z_score')

        '''save the group map'''
        nib.save(z_map, path(out_dir,f'group_{contrast}_zmap.nii.gz'))