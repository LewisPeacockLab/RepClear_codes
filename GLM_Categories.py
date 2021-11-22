import nibabel as nib
nib.openers.Opener.default_compresslevel = 6
from nilearn.image import mean_img, get_data, threshold_img, new_img_like, clean_img
from nilearn.glm.first_level import FirstLevelModel
#from nilearn import plotting
from nilearn.glm.second_level import SecondLevelModel


def mkdir(path,local=False):
    if not os.path.exists(path):
        os.makedirs(path)


def pad_contrast(contrast_, n_columns):
    """A small routine to append zeros in contrast vectors"""
    return np.hstack((contrast_, np.zeros(n_columns - len(contrast_))))

def run_lvl1(subs=[],ses_list=[]):
    for sub in bar(subs):
        for task in tasks:
            for ses in tasks[task]['ses']:
                if ses in ses_list:
                    lvl1(sub,ses,task)

def lvl1(sub,ses=None,task=None):
    '''object that returns useful relative paths'''
    subj = bids_meta(sub)

    '''load in the denoised bold data and events file'''
    if task == 'localizer':
        img = [nib.load(path(subj.denoised,f'{subj.fsub}_ses-1_task-localizer_run-{run}_space-MNI152NLin2009cAsym_res-2_desc-preproc_denoised_bold.nii.gz')) for run in [1,2]]
        events = [pd.read_csv(path(subj.timing,'ses-1','func',f'{subj.fsub}_ses-1_task-localizer_run-{run}_events.tsv'),sep='\t') for run in [1,2]]
        events = [df[df.trial_type!='rest'].reset_index(drop=True) for df in events]        
    else:
        img = nib.load(path(subj.denoised,f'{subj.fsub}_ses-{ses}_task-{task}_space-MNI152NLin2009cAsym_res-2_desc-preproc_denoised_bold.nii.gz'))
        events = pd.read_csv(path(subj.timing,f'ses-{ses}','func',f'{subj.fsub}_ses-{ses}_task-{task}_events.tsv'),sep='\t')

    if ses > 1:
        events.trial_type = events.trial_type + '_' + events.half

    if task == 'acquisition':
        events.shock = events.shock.astype(bool)

        for i in range(events.shape[0]):
            onset, duration, trial_type, shock = events.loc[i,['onset','duration','trial_type','shock']]
 
            if shock: 
                new_row = {'onset': (onset+duration), 'duration': 0, 'trial_type':'US'}
                events = events.append(new_row, ignore_index=True)

        events = events.sort_values(by='onset').reset_index(drop=True)

    '''initialize and fit the GLM'''
    model = FirstLevelModel(t_r=1,slice_time_ref=.5,hrf_model='glover',
                            drift_model=None,high_pass=None,mask_img=subj.refvol_mask,signal_scaling=False,
                            smoothing_fwhm=6,noise_model='ar1',n_jobs=1,verbose=2,memory='/scratch1/05426/ach3377/nilearn_cache',memory_level=1)
    
    model.fit(run_imgs=img,events=events,confounds=None)

    '''grab the number of regressors in the model'''
    n_columns = model.design_matrices_[0].shape[-1]
    
    '''define the contrasts - the order of trial types is stored in model.design_matrices_[0].columns
       pad_contrast() adds 0s to the end of a vector in the case that other regressors are modeled, but not included in the primary contrasts'''
    if ses > 1:

        contrasts = {'early_CS+_vs_CS-':   pad_contrast([1,0,1,0,-2,0],  n_columns),
                     'early_CS-_vs_CS+':   pad_contrast([-1,0,-1,0,2,0], n_columns),
                     'early_CS+E_vs_CS-':  pad_contrast([1,0,0,0,-1,0],  n_columns),
                     'early_CS+U_vs_CS-':  pad_contrast([0,0,1,0,-1,0],  n_columns),
                     'early_CS+E_vs_CS+U': pad_contrast([1,0,-1,0,0,0],  n_columns),
                     'early_CS+E':         pad_contrast([1,0,0,0,0,0],   n_columns),
                     'early_CS+U':         pad_contrast([0,0,1,0,0,0],   n_columns),
                     'early_CS-':          pad_contrast([0,0,0,0,1,0],   n_columns),
                     'late_CS+_vs_CS-':    pad_contrast([0,1,0,1,0,-2],  n_columns),
                     'late_CS-_vs_CS+':    pad_contrast([0,-1,0,-1,0,2], n_columns),
                     'late_CS+E_vs_CS-':   pad_contrast([0,1,0,0,0,-1],  n_columns),
                     'late_CS+U_vs_CS-':   pad_contrast([0,0,0,1,0,-1],  n_columns),
                     'late_CS+E_vs_CS+U':  pad_contrast([0,1,0,1,0,0],   n_columns),
                     'late_CS+E':          pad_contrast([0,-1,0,0,0,0],  n_columns),
                     'late_CS+U':          pad_contrast([0,0,0,-1,0,0],  n_columns),
                     'late_CS-':           pad_contrast([0,0,0,0,0,-1],  n_columns)
                    }
    
    elif task == 'acquisition':  

        contrasts = {'CS+_vs_CS-':   pad_contrast([1,1,-2],  n_columns),
                     'CS-_vs_CS+':   pad_contrast([-1,-1,2], n_columns),
                     'CS+E_vs_CS-':  pad_contrast([1,0,-1],  n_columns),
                     'CS+U_vs_CS-':  pad_contrast([0,1,-1],  n_columns),
                     'CS+E_vs_CS+U': pad_contrast([1,-1,0],  n_columns),
                     'CS+E_acq':     pad_contrast([1,0,0],   n_columns),
                     'CS+U_acq':     pad_contrast([0,1,0],   n_columns),
                     'CS-_acq':      pad_contrast([0,0,1],   n_columns)
                    }
    
    elif task == 'extinction':  

        contrasts = {'CS+E_vs_CS-':  pad_contrast([1,-1], n_columns),
                     'CS-_vs_CS+E':  pad_contrast([-1,1], n_columns),
                     'CS+E_ext':     pad_contrast([1,0],  n_columns),
                     'CS-_ext':      pad_contrast([0,1],  n_columns)
                    }

    elif task == 'localizer':
    
        contrasts = {'animals':             pad_contrast([3,-1,-1,0,-1],  n_columns),
                     'tools':               pad_contrast([-1,-1,-1,0,3],  n_columns),
                     'scenes':              pad_contrast([-1,3,-1,0,-1],  n_columns),
                     'sound':               pad_contrast([-1,-1,-1,4,-1], n_columns),
                     'visual_vs_sound':     pad_contrast([1,1,1,-4,1],    n_columns),
                     'intact_vs_scrambled': pad_contrast([1,1,-3,0,1],    n_columns),
                     'tags':                pad_contrast([1,-1,-1,0,1],   n_columns),
                     'animals_vs_tools':    pad_contrast([1,0,0,0,-1],    n_columns)}

    '''point to and if necessary create the output folder'''
    out_folder = path(subj.model,f'ses-{ses}_task-{task}_lvl1')
    if not os.path.exists(out_folder): os.makedirs(out_folder,exist_ok=True)

    '''compute and save the contrasts'''
    for contrast in contrasts:
        z_map = model.compute_contrast(contrasts[contrast],output_type='z_score')
        nib.save(z_map,path(out_folder,f'{contrast}_zmap.nii.gz'))


def lvl2(subs=[],ses=None,task=None):

    '''define the first level contrasts to use'''
    if ses > 1:
        contrasts = ['early_CS+_vs_CS-','early_CS-_vs_CS+','early_CS+E_vs_CS-','early_CS+U_vs_CS-','early_CS+E_vs_CS+U','early_CS+E','early_CS+U','early_CS-',
                     'late_CS+_vs_CS-','late_CS-_vs_CS+','late_CS+E_vs_CS-','late_CS+U_vs_CS-','late_CS+E_vs_CS+U','late_CS+E','late_CS+U','late_CS-']

    elif task == 'localizer':
        contrasts = ['animals','tools','scenes','sound','visual_vs_sound','intact_vs_scrambled','tags','animals_vs_tools']

    elif task == 'acquisition':
        contrasts = ['CS+_vs_CS-','CS-_vs_CS+','CS+E_vs_CS-','CS+U_vs_CS-','CS+E_vs_CS+U','CS+E_acq','CS+U_acq','CS-_acq']

    elif task == 'extinction':
        contrasts = ['CS+E_vs_CS-','CS-_vs_CS+E','CS+E_ext','CS-_ext']

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