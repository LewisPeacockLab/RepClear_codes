OUTDATED_IGNORE=1
import os
import glob
import numpy as np
import pandas as pd
import pickle
import time
from datetime import datetime
import matplotlib

from nilearn.signal import clean
from nilearn.maskers import NiftiMasker
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import PredefinedSplit
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

import psutil
import nibabel as nib
from scipy.signal import resample

import fnmatch

xstep='ph5_classification'
xsep=1
xsubject_id = '001'

steps = {'ph1_loading':False,
         'ph2_feature_selection':False,
         'ph3_merging_data':False,
         'ph4_PCA':False,
         'ph5_classification':False}

steps[xstep] = 'True'
print(xstep)

###########################################
# classification arguments
###########################################
# subj (df): regressor, train/test index
# pats (np): feature-selected patterns per each iteration
# align: anatomical
# classifier: LinearSVM, L2Logistic

xargs = {'subject_id': xsubject_id,
         'n_subj': 50,
         'align':'anatomical',
         'phase':'operation',
         'bold':'bold_mcf_brain_hpass_dt_mni_2mm',
         'patterns':'zpats_mni',
         'mask':'harvardoxford_gm_mask',
         'shift_tr':10,
         'operation':'4',
         'n_seps':20,
         'feat_top':10000,
         'n_conds':4,
         'condition':['maintain','replace_category','supress','clear'],
         'PCA': True,
         'classifier':'L2Logistic',
         'penalty':50,
         'max_iter':100,
         'patterns':'zpats_mni',
         'n_comps':70}

# optimal n_comps from Clearmem: 70

######### directories
xdirs = {'home': '/pl/active/banich/studies/wmem/fmri/mvpa/utaustin/'}
xdirs['data']   = os.path.join(xdirs['home'], 'data')
xdirs['mvpa']   = os.path.join(xdirs['data'], 'group_mvpa_operation_cs')
xdirs['pats']   = os.path.join(xdirs['mvpa'], 'pats')
xdirs['fmap']   = os.path.join(xdirs['mvpa'], 'fmap')
xdirs['out']    = os.path.join(xdirs['mvpa'], 'out')
xdirs['test']   = os.path.join(xdirs['mvpa'], 'test')

xdirs['script'] = os.path.join(xdirs['home'], 'cu_src/clearmem-master/clearmem_cs')
xdirs['regs']   = os.path.join(xdirs['script'], 'cs_regs')
xdirs['log']    = os.path.join(xdirs['script'], 'log')
xdirs['fsl']    = '/projects/ics/software/fsl/6.0.3'

for i in xdirs.keys():
    if not os.path.isdir(xdirs[i]):
        os.mkdir(xdirs[i])

######### subject 
f = open(os.path.join(xdirs['script'], "subj50.lst"), "r")
subject_lists = f.read().split('\n')
f.close()

subject_lists = subject_lists[:xargs['n_subj']]
print('* Number of subjects: %s' % str(len(subject_lists)))
  
######### feature mask
xargs['feat_mask'] =  '/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/MVPA_cross_subjects/masks/feature_mask.nii.gz'

def save_model(classifier, pca, filename):
    # Create a dictionary to store the models
    models = {'classifier': classifier, 'pca': pca}

    # Pickle the models and save to disk
    with open(filename, 'wb') as f:
        pickle.dump(models, f)

def load_model(filename):
    # Load the pickled models from disk
    with open(filename, 'rb') as f:
        models = pickle.load(f)
    return models['classifier'], models['pca']

def confound_cleaner(confounds):
    COI = ['a_comp_cor_00','a_comp_cor_01','a_comp_cor_02','a_comp_cor_03','a_comp_cor_04','a_comp_cor_05','framewise_displacement','trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']
    for _c in confounds.columns:
        if 'cosine' in _c:
            COI.append(_c)
    confounds = confounds[COI]
    confounds.loc[0,'framewise_displacement'] = confounds.loc[1:,'framewise_displacement'].mean()
    return confounds  

#this function takes the mask data and applies it to the bold data
def apply_mask(mask=None,target=None):
    coor = np.where(mask == 1)
    values = target[coor]
    if values.ndim > 1:
        values = np.transpose(values) #swap axes to get feature X sample
    return values

def find(pattern, path): #find the pattern we're looking for
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
        return result 

def sample_for_training(full_data, label_df, include_rest=False):
    """
    sample data by runs. 
    Return: sampled labels and bold data
    """ 

    # operation_list: 1 - Maintain, 2 - Replace, 3 - Suppress
    # stim_on labels: 1 actual stim; 2 operation; 3 ITI; 0 rest between runs
    print("\n***** Subsampling data points by runs...")

    category_list = label_df['condition']
    stim_on = label_df['stim_present']
    run_list = label_df['run']
    image_list = label_df['image_id']

    # get faces
    oper_inds = np.where((stim_on == 2) | (stim_on == 3))[0]
    rest_inds = []

    runs = run_list.unique()[1:]

    if include_rest:
        print("Including resting category...")
        # get TR intervals for rest TRs between stims (stim_on == 2)
        rest_bools = ((run_list == runi) | (run_list == runj)) & (stim_on == 2)
        padded_bools = np.r_[False, rest_bools, False]  # pad the rest_bools 1 TR before and after to separate it from trial information
        rest_diff = np.diff(padded_bools)  # get the pairwise diff in the array --> True for start and end indices of rest periods
        rest_intervals = rest_diff.nonzero()[0].reshape((-1,2))  # each pair is the interval of rest periods

        # get desired time points in the middle of rest periods for rest samples; if 0.5, round up
        rest_intervals[:,-1] -= 1
        rest_inds = [np.ceil(np.average(interval)).astype(int) for interval in rest_intervals] + \
                    [np.ceil(np.average(interval)).astype(int)+1 for interval in rest_intervals]

    operation_reg=category_list.values[oper_inds]
    run_reg=run_list.values[oper_inds]
    image_reg = image_list.values[oper_inds]

    # === get sample_bold & sample_regressor
    sample_bold = []
    sample_regressor = operation_reg
    sample_runs = run_reg

    sample_bold = full_data[oper_inds]

    return sample_bold, sample_regressor, sample_runs, image_reg


def load_process_data(subID, task, space, mask_paths):
    save=True
    runs=np.arange(3) + 1  
    mask_ROIS=[mask_paths]

    # ======= generate file names to load
    # get list of data names
    fname_template = f"sub-{subID}_task-{task}_run-{{}}_resampled.nii.gz"
    bold_fnames = [fname_template.format(i)for i in runs]
    bold_paths = [os.path.join('/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/', f"sub-{subID}", "func_resampled", fname) for fname in bold_fnames]    

    # get mask names
    mask_paths = xargs['feat_mask']

    runs=np.arange(3) + 1  

    # get confound filenames
    confound_fnames = [f"*{task}*{run}*confounds*.tsv" for run in runs]
    confound_paths = [os.path.join('/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/', f"sub-{subID}", "func", f) for f in confound_fnames]  # template for each run 
    confound_paths = [glob.glob(p)[0] for p in confound_paths]  # actual file names

    # ======= load bold data & preprocess

    # ===== load data files 
    print("\n*** Loading & cleaning data...")
    print("Loading bold data...")
    # loaded bold shape: (x/y/z x time))
    bolds = [nib.load(p) for p in bold_paths]

    print("Loading masks...")
    masks = nib.load(mask_paths)

    print("Loading confound files...")
    confounds = [pd.read_csv(p,sep='\t') for p in confound_paths]
    confounds_cleaned = [confound_cleaner(c) for c in confounds]

    # ===== for each run & ROI, mask & clean
    print("\n*** Masking & cleaing bold data...")
    cleaned_bolds = [[None for _ in range(len(runs))] for _ in range(len(mask_ROIS))]

    for rowi, mask in enumerate([masks]):
        print(f"Processing mask {rowi}...")
        for coli, (bold, confound) in enumerate(zip(bolds, confounds_cleaned)):
            print(f"Processing run {coli}...")

            # masked: time x vox
            masked = apply_mask(mask=mask.get_fdata(), target=bold.get_fdata())

            # *** clean: confound rows are time; 
            cleaned_bolds[rowi][coli] = clean(masked, confounds=confound, t_r=1, detrend=False, standardize='zscore')
            print(f"ROI {rowi}, run {coli}")
            print(f"shape: {cleaned_bolds[rowi][coli].shape}")

        # {ROI: time x vox}
        preproc_data = {ROI: np.vstack(run_data) for ROI, run_data in zip(mask_ROIS, cleaned_bolds)}

    print("processed data shape: ", [d.shape for d in preproc_data.values()])
    print("*** Done with preprocessing!")

    # save for future use
    if save: 
        for ROI, run_data in preproc_data.items():
            if ROI=='VVS': ROI='VTC'
            bold_dir = os.path.join('/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/', f"sub-{subID}", "func_resampled")            
            out_fname_template = f"sub-{subID}_{space}_{task}_{{}}_masked_cleaned.npy"            
            out_fname = out_fname_template.format('feature_mask')
            print(f"Saving to file {bold_dir}/{out_fname}...")
            np.save(f"{bold_dir}/{out_fname}", run_data)

    #this will handle mutliple ROIs and then if needed save them into 1 dictionary
    try:
        full_dict = {**ready_data, **preproc_data}
    except:
        full_dict = {**preproc_data}
    # array: all_runs_time x all_ROI_vox 
    full_data = np.hstack(list(full_dict.values()))
    return full_data    

def get_shifted_labels(task, shift_size_TR, rest_tag=0):
    # load labels, & add hemodynamic shift to all vars

    def shift_timing(label_df, TR_shift_size, tag=0):
        # Shift 2D df labels by given TRs with paddings of tag
        # Input label_df must be time x nvars
        nvars = len(label_df.loc[0])
        shift = pd.DataFrame(np.zeros((TR_shift_size, nvars))+tag, columns=label_df.columns)
        shifted = pd.concat([shift, label_df])
        return shifted[:len(label_df)]  # trim time points outside of scanning time     

    print("\n***** Loading labels...")

    subject_design_dir='/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/subject_designs/'

    #using the task tag, we want to get the proper tag to pull the subject and phase specific dataframes
    if task=='preremoval': temp_task='pre-localizer'
    if task=='postremoval': temp_task='post-localizer'
    if task=='study': temp_task='study'

    sub_design=(f"*{xsubject_id}*{temp_task}*tr*")
    sub_design_file=find(sub_design,subject_design_dir)
    sub_design_matrix = pd.read_csv(sub_design_file[0]) #this is the correct, TR by TR list of what happened during this subject's study phase

    shifted_df = shift_timing(sub_design_matrix, shift_size_TR, rest_tag)

    return shifted_df 

def remove_label(X, Y, label):
    # Find the indices of the time points with the given label
    idx_remove = np.where(Y == label)[0]
    
    # Remove the time points with the given label from the data and regressors
    X_new = np.delete(X, idx_remove, axis=0)
    Y_new = np.delete(Y, idx_remove, axis=0)
    
    return X_new, Y_new

def rename_labels(Y, flag):
    # Create a new array to hold the renamed labels
    Y_new = np.zeros_like(Y, dtype='str')
    
    # Map the old label values to new label values based on the flag
    if flag == "clearmem":
        label_map = {1: 'maintain', 2: 'replace', 4: 'suppress'}
    elif flag == "repclear":
        label_map = {1: 'maintain', 2: 'replace', 3: 'suppress'}
    else:
        print("Invalid flag")
        return None
    
    # Rename the labels
    for old_label, new_label in label_map.items():
        idx = np.where(Y == old_label)[0]
        Y_new[idx] = new_label
    
    return Y_new

###########################################
# PH1: load patterns/ zscoring/ per each subject
#      removed rest TRs from shifted regs/sels/patterns
###########################################
if steps['ph1_loading']:
    start = time.time()
    
    logs('======================================', xlog=xlog)
    logs('PH1: ph1_loading: sub%s' % xsubject_id, xlog=xlog)
    logs('======================================', xlog=xlog)

    ######### Open log file
    xlog_fname = os.path.join(xdirs['log'], 'logs_%s_%s.txt' %  (xstep, xsubject_id))
    xlog = open(xlog_fname, 'w')

    ######### 0. setup directories
    xdirs['subject'] = os.path.join(xdirs['data'], 'clearmem_v1_sub%s' % xsubject_id)
    xdirs['study']   = np.sort(glob.glob('%s/bold/clearmem*study*' % xdirs['subject']))
    
    ###########################################
    ######### 1. subject regressor
    # load regs: selectors=run, regressors=reg_sh
    # shift TRs -> remove rest TRs from regressors
    
    # classification regressors: operation
    xdf_reg = pd.read_csv(os.path.join(xdirs['regs'], 'operation_regs_%s.csv' % xsubject_id))
    norest_trs = np.squeeze(np.nonzero(np.array(xdf_reg['reg_sh'])))

    logs('(+) Loading regressors for sub%s:' % xsubject_id, xlog=xlog)
    logs(' | %d runs, %d volumes, norest: %d volumes' % 
         (np.unique(xdf_reg['run']).shape[0], xdf_reg.shape[0], norest_trs.shape[0]), xlog=xlog)

    # volume id
    xdata = pd.DataFrame()
    #xdata['volume'] = xdf_reg['volume'][norest_trs].values
    #xdata['regressor'] = xdf_reg['reg_sh'][norest_trs].values
    #xdata['runs'] = xdf_reg['run'][norest_trs].values
    
    ###########################################
    ######### 2. load patterns + zscore within run
    # load masked patterns for classification
    logs('(+) Loading fMRI patterns for sub%s' % xsubject_id, xlog=xlog)

    # create a masker and give it the fMRI run data 
    xmask = os.path.join(xdirs['script'], '%s.nii.gz' % xargs['mask'])
    masker = NiftiMasker(mask_img=xmask, standardize=False, detrend=True, t_r=2)

    allmaskedData = []
    for xrun in xdirs['study']:
        print(' ... loading: ', xrun)
        fmri = os.path.join(xrun, "%s.nii.gz" % xargs['bold'])
        maskedData = masker.fit_transform(fmri)

        # voxel-wise zscore within run
        n_trs, n_vox = maskedData.shape
        maskedData_zscore = StandardScaler().fit_transform(maskedData)
        allmaskedData.append(maskedData_zscore)

    # set any nan values to 0
    z_patterns = np.nan_to_num(np.vstack(allmaskedData))

    print(' | concatenated patterns (tr, vox): ', z_patterns.shape)
    
    # remove rest (+spike movement) trs
    xpats = z_patterns[norest_trs,:]
    logs(' | masked voxels: %s, n_trs: %s' % (str(xpats.shape[1]), str(xpats.shape[0])), xlog=xlog)
    
    ######### subject structure
    xdata['patterns']=pd.DataFrame(xpats).values.tolist()

    ######### Save each subj.dataFrame
    logs('(+) Saving fMRI patterns for %s' % xsubject_id, xlog=xlog)
    xf = os.path.join(xdirs['pats'],'%s_%s.pkl' % (xargs['patterns'], xsubject_id))
    xdata.to_pickle(xf)

    end = time.time()
    elapsed = end - start
    m, s = divmod(elapsed, 60)
    logs('Time elapsed: %s minutes %s seconds' % (round(m), round(s)), xlog=xlog) 
    xlog.close() 

###########################################
# save regressors
###########################################
xreg_save = False
xf = os.path.join(xdirs['out'], 'cat_regs_n%d.pkl' % xargs['n_subj'])

if xreg_save:
    ######### Concatenate all subjects
    xcat_reg = {}
    for i in ['subject','volume','run','regressor']:
        xcat_reg[i] = []
    
    for xsubject_id in subject_lists:
        ######### classification regressors: operation
        xdf_reg = pd.read_csv(os.path.join(xdirs['regs'], 'operation_regs_%s.csv' % xsubject_id))
        norest_trs = np.squeeze(np.nonzero(np.array(xdf_reg['reg_sh'])))

        print('(+) Loading regressors for sub%s:' % xsubject_id)
        print(' | %d runs, %d volumes, norest: %d volumes' % 
             (np.unique(xdf_reg['run']).shape[0], xdf_reg.shape[0], norest_trs.shape[0]))

        # volume id
        xcat_reg['subject'].append(xsubject_id)
        xcat_reg['volume'].append(xdf_reg['volume'][norest_trs].values)
        xcat_reg['regressor'].append(xdf_reg['reg_sh'][norest_trs].values)
        xcat_reg['run'].append(xdf_reg['run'][norest_trs].values)

    print(' ... saving the cat regs')
    xcat_reg = pd.DataFrame(xcat_reg)
    xcat_reg.to_pickle(xf)
    
else:
    print(' ... loading the cat regs')
    xcat_reg = pd.read_pickle(xf)

# display(xcat_reg)

###########################################
# PH2: feature selection: ANOVA 
###########################################
if steps['ph2_feature_selection']:
    start = time.time()

    ######### Open log file
    xlog_fname = os.path.join(xdirs['log'], 'logs_%s_%s.txt' %  (xstep, xsep))
    xlog = open(xlog_fname, 'w')
    
    logs('======================================', xlog=xlog)
    logs('PH2: feature-selection', xlog=xlog)
    logs('======================================', xlog=xlog)

    # 1. concatenate 1/n_sep voxels across all subjects
    # 2. save fmap from feature-selection for the 1/n patterns for n_sub iterations
    # 3. concatenate the separated fmaps

    logs('... separation: %d' % xsep, xlog=xlog)

    ######### Concatenate 1/n voxels for all subjects
    for xsubject_id in subject_lists:
        print('#########################')
        print('separation %d / %d, subject: %s' 
            % (xsep, xargs['n_seps'], xsubject_id))
        print(' * memory available: %0.2f percent' % psutil.virtual_memory().percent)

        ######### Load each subject dataFrame
        xf = os.path.join(xdirs['pats'],'%s_%s.pkl' % (xargs['patterns'], xsubject_id))
        xdata = pd.read_pickle(xf)

        # xpat[tr][vox]: timecourse=xpat[:][vox]
        xpat = xdata['patterns'].to_numpy()
        xdata = xdata.drop(columns=['patterns'])    
        del xdata

        # define subset of voxels
        sep_vox = round(len(xpat[0])/xargs['n_seps'])
        if xsep!=xargs['n_seps']:
            it_voxs = range(sep_vox*(xsep-1), sep_vox*xsep)
        else:
            it_voxs = range(sep_vox*(xsep-1), len(xpat[0]))

        t_pat = np.zeros([len(xpat), len(it_voxs)])
        for xtr in range(len(xpat)):
            t_pat[xtr, :] = xpat[xtr][it_voxs[0]:(it_voxs[-1]+1)]

        # cat all subjects pat
        if xsubject_id=='001':
            cat_pat = t_pat
        else:
            cat_pat = np.concatenate((cat_pat, t_pat), axis=0)

        del xpat, t_pat

    n_trs, n_voxs = cat_pat.shape
    logs('TRs: %d, voxels: %d' % (n_trs, n_voxs), xlog=xlog)

    ######### load regressor
    xcat_reg = pd.read_pickle(os.path.join(xdirs['out'], 'cat_regs_n%d.pkl' % xargs['n_subj']))
    xreg_sh = np.hstack(xcat_reg['regressor'])

    ######### ANOVA
    logs('ANOVA in progress...', xlog)
    xF, xP = f_classif(cat_pat, xreg_sh)

    xF = np.nan_to_num(xF)
    xP[np.isnan(xP)] = 1

    del cat_pat

    ######### Save separated tmp_fmap
    temp_f = pd.DataFrame(np.vstack(xF))
    temp_p = pd.DataFrame(np.vstack(xP))

    feat_sels = pd.DataFrame()
    feat_sels['fmap'] = temp_f.values.tolist()
    feat_sels['pmap'] = temp_p.values.tolist()

    xf = os.path.join(xdirs['fmap'], 'tmp_feat_fmap_sep%d_n%d.pkl' % 
                      (xsep, xargs['n_subj']))
    logs('Saving fmap ...: %s' % xf, xlog)
    feat_sels.to_pickle(xf)

    del feat_sels

    end = time.time()
    elapsed = end - start
    m, s = divmod(elapsed, 60)
    logs('Time elapsed: %s minutes %s seconds' % (round(m), round(s)), xlog=xlog) 
    xlog.close() 

###########################################
# PH3: Merging data
###########################################
if steps['ph3_merging_data']:
    xmerge_fmap = False
    
    ######### Open log file
    xlog_fname = os.path.join(xdirs['log'], 'logs_%s.txt' %  xstep)
    xlog = open(xlog_fname, 'w')

    logs('======================================', xlog=xlog)
    logs('PH3: merging data', xlog=xlog)
    logs('======================================', xlog=xlog)
    
    # 1. load fmap for all xiterations
    # 2. concatenate pattern of each subjects from the selected voxels
    
    ######### Merge fmaps
    xf_fmap = os.path.join(xdirs['out'], 'feat_fmap_n%d.pkl' % xargs['n_subj'])
    
    if xmerge_fmap:
        feat_sels = pd.DataFrame()
        for xsep in range(xargs['n_seps']):
            xf = os.path.join(xdirs['fmap'], 'tmp_feat_fmap_sep%d_n%d.pkl' % 
                              (xsep+1, xargs['n_subj']))
            tfeat = pd.read_pickle(xf)
            print(' ... %d: %d voxels' % (xsep+1, tfeat.shape[0]))

            if xsep==0:
                feat_sels = tfeat
            else:
                feat_sels = pd.concat([feat_sels, tfeat], axis=0)
                
        logs(' ... saving the merging fmaps', xlog=xlog)
        feat_sels = pd.DataFrame(feat_sels.values, columns=['fmap', 'pmap'])    
        feat_sels.to_pickle(xf_fmap)
    else:
        ######### load fmaps
        logs(' ... loading the merging fmaps', xlog=xlog)
        feat_sels = pd.read_pickle(xf_fmap)
        
    ######### thresholding
    xP = np.squeeze(feat_sels['pmap'].values.tolist())
    xthres_p = np.sort(xP)[xargs['feat_top']]
    xthres_F = feat_sels['fmap'][np.where(feat_sels['pmap']==xthres_p)[0]].values[0][0]
    
    xfeat = xP < xthres_p
    logs(' | top %d voxels were selected P < %0.4f' % 
         (np.where(xfeat==True)[0].shape[0], xthres_p), xlog)

    ###########################################
    ######### Merge pats
    xf_pat = os.path.join(xdirs['out'], 'cat_pat_n%d.npy' % xargs['n_subj'])
    
    ######### Load feature-selected pats
    xcat_pat = []
    for xsubject_id in subject_lists:

        logs('#########################', xlog=xlog)
        logs('| subject: %s' % xsubject_id, xlog=xlog)
        print('available: %0.2f' % psutil.virtual_memory().percent)

        xf = os.path.join(xdirs['pats'],'%s_%s.pkl' % (xargs['patterns'], xsubject_id))
        xdata = pd.read_pickle(xf)

        n_trs = xdata.patterns.shape[0]
        n_voxs = len(xP)
        n_selvoxs = np.sum(xfeat)

        logs('TRs = %d, features = %d out of %d' % (n_trs, n_selvoxs, n_voxs), xlog=xlog)

        xpats = np.hstack(xdata.patterns.values).reshape(n_trs, n_voxs)
        xfeat_pats = xpats[:, xfeat]

        if xsubject_id=='001':
            xcat_pat = xfeat_pats
        else:
            xcat_pat = np.concatenate((xcat_pat, xfeat_pats), axis=0)

        del xdata

    ######### save cat_pats
    logs('... saving cat patterns: %s' % xf_pat, xlog)
    np.save(xf_pat, xcat_pat)

    ###########################################
    ######### save feature mask nii.gz
    xf = os.path.join(xdirs['out'], 'feature_mask.nii.gz')
    
    xmask = os.path.join(xdirs['script'], '%s.nii.gz' % xargs['mask'])
    masker = NiftiMasker(mask_img=xmask, standardize=False, detrend=True, t_r=2)
    masker.fit_transform(xmask)
    masker.inverse_transform((1*xfeat)).to_filename(xf)
    
    ### same as inverse_transform
    # xmask_img = nib.load(xmask)
    # xmask_mx = xmask_img.get_fdata()
    # xmx = np.zeros(xmask_mx.shape)
    # xmx[np.where(xmask_mx>0)] = (1*xfeat)
    # xmx = xmx.astype(xmask_img.get_data_dtype())
    # x = nib.Nifti1Image(xmx, xmask_img.affine, xmask_img.header)
    # nib.save(x, xf)
    
    logs(' | whole brain mask: %d voxels' % 
         np.where(nib.load(xmask).get_fdata()>0)[0].shape[0], xlog=xlog)
    logs(' | fmap: %d voxels' % feat_sels.shape[0], xlog=xlog)

########
### preparing testing data
xprepare_test = True

if xprepare_test:
    xsubject_id = '002'
    xtest = f'/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/sub-{xsubject_id}/func_resampled/sub-{xsubject_id}_task-study_run-1_resampled.nii.gz'

    # regressor + selector:
    task='study'
    shift_size_TR=5
    rest_tag=0
    label_df = get_shifted_labels(task, shift_size_TR, rest_tag)

    full_data = load_process_data(xsubject_id, task, 'MNI', xargs['feat_mask'])
    print(f"Full_data shape: {full_data.shape}")

    X, Y, runs, imgs = sample_for_training(full_data, label_df)

    #X is the processes and prepared BOLD data
    # Y is the regressors for operation (no rest)


    ######################################################
    # TRAIN DATA
    ######### load train pats: feature selected
    xtrain_pat = np.load('/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/MVPA_cross_subjects/out/cat_pat_n50.npy')
    
    xtrain_reg = pd.read_pickle('/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/MVPA_cross_subjects/out/cat_regs_n50.pkl')



#before we combine for PCA, we need to relabel and remove regessors (drop clear and rename to operation):
xcat_regs = np.hstack(xtrain_reg['regressor'])

xtrain_trim, xcat_regs_trim = remove_label(xtrain_pat, xcat_regs, 5) #this will drop the "clear" labels

#use a renaming of label functions to organize all the data:

xcat_regs_named=rename_labels(xcat_regs_trim,'clearmem')
regressors_named = rename_labels(Y, 'repclear')

combined_data=np.concatenate([xtrain_trim,X])

# Create an array of 1's for training data
train_labels = np.ones(xtrain_trim.shape[0])

# Create an array of 2's for testing data
test_labels = np.ones(X.shape[0]) * 2

# Concatenate the two arrays
all_labels = np.concatenate((train_labels, test_labels))

## we will need to add back in PCA because of training issues:
pca = PCA(n_components=xargs['n_comps'])
xpats_pca_train = pca.fit_transform(xtrain_trim)

xpats_pca_test = pca.transform(X)

# # Split the combined_data array into separate training and testing arrays based on the labels
# train_data = xpats_pca_combined[all_labels == 1]
# test_data = xpats_pca_combined[all_labels == 2]

# xf_pca = os.path.join(xdirs['out'], 'PCA_%dcomp_pat_%d.npy' % (xargs['n_comps'], xfold))
# np.save(xf_pca, xpats_pca)

###########################################
# PH5: Classification
###########################################
start = time.time()

######################################################
# Classification
######################################################


classifier = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000,
                                multi_class='ovr',n_jobs=-1,verbose=1)

clf = classifier.fit(xpats_pca_train, xcat_regs_named)

save_model(clf,pca,'trained_models.pkl')

y_score = clf.decision_function(xpats_pca_test)
n_classes=np.unique(regressors_460ms_named).size
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    temp_y=np.zeros(regressors_460ms_named.size)
    label_ind=np.where(Y==(i+1))
    temp_y[label_ind]=1

    fpr[i], tpr[i], _ = roc_curve(temp_y, y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

pred = clf.predict(xpats_pca_test)
evi = clf.predict_proba(xpats_pca_test)
xscore = accuracy_score(regressors_460ms_named, pred)



######### Confusion matrix (actual, predicted)
# cm = confusion_matrix(xregs_test, pred)
# cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
xconds = np.unique(xcat_regs)
xconfusion = np.zeros([len(xconds), len(xconds)])
xevidence = np.zeros([len(xconds), len(xconds)])
xaccuracy = np.zeros(len(xconds))

for xx in range(0, len(xconds)):
    xdesire = xconds[xx]
    xpred = pred[xregs_test==xdesire]
    xevi  = evi[xregs_test==xdesire, :]

    for yy in range(0, len(xconds)):
        ycond = xconds[yy]
        xconfusion[xx, yy] = len(xpred[xpred==ycond])/len(xpred)
        xevidence[xx, yy] = np.mean(xevi[:,yy])

        if yy==xx:
            xaccuracy[xx] = len(xpred[xpred==ycond])/len(xpred)

    del xdesire, xpred

xout = {}
xout['confusion_%d' % xfold] = list(xconfusion)
xout['accuracy_%d' % xfold] = list(xaccuracy)
xout['evidence_%d' % xfold] = list(xevidence)
xout['score_%d' % xfold] = xscore
xout = pd.DataFrame(xout, index=xargs['condition'])
display(xout)

print(' ... saving MVPA output')
xf = os.path.join(xdirs['out'], 'mvpaout_%d.pkl' % xfold)
xout.to_pickle(xf)

######### time
end = time.time()
elapsed = end - start
m, s = divmod(elapsed, 60)
logs('Time elapsed: %s minutes %s seconds' % (round(m), round(s)), xlog=xlog) 

