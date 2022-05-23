#code to train on localizer data and then test on the study data

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
from sklearn.model_selection import PredefinedSplit, cross_validate, cross_val_predict, GridSearchCV, LeaveOneGroupOut, KFold
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest, SelectFpr
from sklearn.preprocessing import StandardScaler
from nilearn.input_data import NiftiMasker,  MultiNiftiMasker
from nilearn.image import clean_img
from nilearn.signal import clean
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score

subs=['02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','20','23','24','25','26']
TR_shift=5
brain_flag='T1w'

#masks=['wholebrain','vtc'] #wholebrain/vtc
mask_flag='vtc'

clear_data=1 #0 off / 1 on
force_clean=1

workspace = 'scratch'
data_dir = '/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/'
param_dir = '/scratch/06873/zbretton/repclear_dataset/BIDS/params/'
results_dir = '/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/model_fitting_results/'


def get_preprocessed_data(subID, task, space, mask_ROIS, runs=np.arange(6)+1, save=False):
    '''
    Data preprocessing for required sub, task, space, & ROI. 
    Load preprocessed data if saved; 
    Else generate paths for bold, mask, & confound files, then mask & clean data

    Input:
    subID: str of 3 digit number (e.g. "002")
    task: 'rest', 'preremoval', 'study', or 'postremoval'
    space: 'T1w' or 'MNI'
    mask_ROIS: could be (1) "wholebrain", str;
                        (2) be a list of ROIs

    Output:
    full_data: array of (all_runs_time x all_ROI_vox)
    '''

    space_long = spaces[space]
    def load_existing_data(): 
        print("\n*** Attempting to load existing data if there is any...")
        preproc_data = {}
        todo_ROIs = []
        
        if type(mask_ROIS) == str and mask_ROIS == 'wholebrain':
            if os.path.exists(os.path.join(bold_dir, out_fname_template.format('wholebrain'))):
                print("Loading saved preprocessed data", out_fname_template, '...')
                preproc_data["wholebrain"] = np.load(os.path.join(bold_dir, out_fname_template.format('wholebrain')))
            else:
                print("Wholebrain data to be processed.")
                todo_ROIs = "wholebrain"

        elif type(mask_ROIS) == list:
            for ROI in mask_ROIS:
                if ROI == 'VVS': ROI = 'vtc'   # only change when loading saved data
                if os.path.exists(os.path.join(bold_dir, out_fname_template.format(ROI))):
                    print("Loading saved preprocessed data", out_fname_template.format(ROI), '...')
                    preproc_data[ROI] = np.load(os.path.join(bold_dir, out_fname_template.format(ROI)))
                else: 
                    if ROI == 'vtc': ROI = 'VVS'  # change back to laod masks...
                    print(f"ROI {ROI} data to be processed.")
                    todo_ROIs.append(ROI)
        else: 
            raise ValueError(f"Man! Incorrect ROI value! (Entered: {mask_ROIS})")

        return preproc_data, todo_ROIs

    def confound_cleaner(confounds):
        COI = ['a_comp_cor_00','a_comp_cor_01','a_comp_cor_02','a_comp_cor_03','a_comp_cor_05','framewise_displacement','trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']
        for _c in confounds.columns:
            if 'cosine' in _c:
                COI.append(_c)
        confounds = confounds[COI]
        # *** pd future warning: "A value is trying to be set on a copy of a slice from a DataFrame" ***
        confounds.loc[0,'framewise_displacement'] = confounds.loc[1:,'framewise_displacement'].mean()
        return confounds

    def apply_mask(mask=None,target=None):
        coor = np.where(mask == 1)
        values = target[coor]
        # *** data is already vox x time when loaded ***
        # print("before transpose:", values.shape)
        # if values.ndim > 1:
        #     values = np.transpose(values) #swap axes to get feature X sample
        # print("after transpose:", values.shape)
        return values


    # ====================================================
    print(f"\n***** Data preprocessing for sub {subID} {task} {space} with ROIs {mask_ROIS}...")

    # whole brain mask: includeing white matter. should be changed to grey matter mask later
    if mask_ROIS == 'wholebrain':
        raise NotImplementedError("function doesn't support wholebrain mask!")

    # FIXME: regenerate masked & cleaned data & save to new dir
    bold_dir = os.path.join(data_dir, f"sub-{subID}", "func")
    out_fname_template = f"sub-{subID}_{space}_{task}_{{}}_masked_cleaned.npy"

    # ========== check & load existing files
    ready_data, mask_ROIS = load_existing_data()
    if type(mask_ROIS) == list and len(mask_ROIS) == 0: 
        return np.vstack(list(ready_data.values()))
    else: 
        print("Preprocessing ROIs", mask_ROIS)

    # ========== start from scratch for todo_ROIs
    # ======= generate file names to load
    # get list of data names
    fname_template = f"sub-{subID}_task-{task}_run-{{}}_space-{space_long}_desc-{{}}.nii.gz"
    bold_fnames = [fname_template.format(i, "preproc_bold") for i in runs]
    bold_paths = [os.path.join(data_dir, f"sub-{subID}", "func", fname) for fname in bold_fnames]

    # get mask name
    if type(mask_ROIS) == str:  # 'wholebrain'
        # whole brain masks: 1 for each run
        mask_fnames = [fname_template.format(i, "brain_mask") for i in runs]
        mask_paths = [os.path.join(data_dir, f"sub-{subID}", "func", fname) for fname in mask_fnames]
    else:
        # ROI masks: 1 for each ROI
        mask_fnames = [f"{ROI}_{task}_{space}_mask.nii.gz" for ROI in mask_ROIS]
        mask_paths = [os.path.join(data_dir, f"sub-{subID}", "new_mask", fname) for fname in mask_fnames]

    # get confound filenames
    confound_fnames = [f"*{task}*{run}*confounds*.tsv" for run in runs]
    confound_paths = [os.path.join(data_dir, f"sub-{subID}", "func", f) for f in confound_fnames]  # template for each run 
    confound_paths = [glob.glob(p)[0] for p in confound_paths]  # actual file names

    # ======= load data & preprocessing

    # ===== load data files 
    print("\n*** Loading & cleaning data...")
    print("Loading bold data...")
    # loaded bold shape: (x/y/z x time))
    bolds = [nib.load(p) for p in bold_paths]

    print("Loading masks...")
    masks = [nib.load(p) for p in mask_paths]

    print("Loading confound files...")
    confounds = [pd.read_csv(p,sep='\t') for p in confound_paths]
    confounds_cleaned = [confound_cleaner(c) for c in confounds]

    # ===== for each run & ROI, mask & clean
    print("\n*** Masking & cleaing bold data...")
    if type(mask_ROIS) == str:  # 'wholebrain'
        cleaned_bolds = [None for _ in range(len(runs))]
        # all files are by nruns
        for runi, (bold, mask, confound) in enumerate(zip(bolds, masks, confounds_cleaned)):
            print(f"Processing run {runi}...")
            masked = apply_mask(mask=mask.get_data(), target=bold.get_data())
            # *** clean: confound length in time; transpose signal to (time x vox) for cleaning & model fitting ***
            cleaned_bolds[runi] = clean(masked.T, confounds=confound, t_r=1, detrend=False, standardize='zscore')
            print("claened shape: ", cleaned_bolds[runi].shape)

        # {ROI: time x vox}
        preproc_data = {'wholebrain': np.hstack(cleaned_bolds)}

    else:  # list of specific ROIs
        cleaned_bolds = [[None for _ in range(len(runs))] for _ in range(len(mask_ROIS))]

        for rowi, mask in enumerate(masks):
            print(f"Processing mask {rowi}...")
            for coli, (bold, confound) in enumerate(zip(bolds, confounds_cleaned)):
                print(f"Processing run {coli}...")
                # *** nib deprecation warning: 
                #       "get_data() is deprecated in favor of get_fdata(), which has a more predictable return type. 
                #        To obtain get_data() behavior going forward, use numpy.asanyarray(img.dataobj)."
                # masked: vox x time
                masked = apply_mask(mask=mask.get_data(), target=bold.get_data())
                # *** clean: confound length in time; transpose signal to (time x vox) for cleaning & model fitting ***
                cleaned_bolds[rowi][coli] = clean(masked.T, confounds=confound, t_r=1, detrend=False, standardize='zscore')
                print(f"ROI {rowi}, run {coli}")
                print(f"shape: {cleaned_bolds[rowi][coli].shape}")

        # {ROI: time x vox}
        preproc_data = {ROI: np.vstack(run_data) for ROI, run_data in zip(mask_ROIS, cleaned_bolds)}

    print("processed data shape: ", [d.shape for d in preproc_data.values()])
    print("*** Done with preprocessing!")

    # save for future use
    if save: 
        for ROI, run_data in preproc_data.items():
            out_fname = out_fname_template.format(ROI)
            print(f"Saving to file {bold_dir}/{out_fname}...")
            np.save(f"{bold_dir}/{out_fname}", run_data)

    full_dict = {**ready_data, **preproc_data}
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

    event_path = os.path.join(param_dir, f'{task}_events.csv')
    events_df = pd.read_csv(event_path)
    # === commented out: getting only three rows
    # # categories: 1 Scenes, 2 Faces / 0 is rest
    # TR_category = events_df["category"].values
    # # stim_on labels: 1 actual stim; 2 rest between stims; 0 actual rest
    # TR_stim_on = events_df["stim_present"].values
    # # run
    # TR_run_list = events_df["run"].values
    # # shifted
    # sTR_category = shift_timing(TR_category, shift_size_TR, rest_tag)
    # sTR_stim_on = shift_timing(TR_stim_on, shift_size_TR, rest_tag)
    # sTR_run_list = shift_timing(TR_run_list, shift_size_TR, rest_tag)

    shifted_df = shift_timing(events_df, shift_size_TR, rest_tag)

    return shifted_df


def random_subsample(full_data, label_df, include_rest=True):
    """
    Subsample data by random sampling, only based on target labels but not runs
    """
    # stim_list: 1 Scenes, 2 Faces / 0 is rest
    # stim_on labels: 1 actual stim; 2 rest between stims; 0 rest between runs
    print("\n***** Randomly subsampling data points...")

    stim_list = label_df['category']
    stim_on = label_df['stim_present']

    labels = set(stim_list)
    labels.remove(0)  # rest will be done separately afterwards

    # indices for each category
    stim_inds = {lab: np.where((stim_on == 1) & (stim_list == lab))[0] for lab in labels}
    min_n = min([len(inds) for inds in stim_inds.values()])  # min sample size to get from each category

    sampled_inds = {}
    # ===== get indices of samples to choose
    # subsample min_n samples from each catefgory
    for lab, inds in stim_inds.items():
        chosen_inds = np.random.choice(inds, min_n, replace=False)
        sampled_inds[int(lab)] = sorted(chosen_inds)

    # === if including rest: 
    if include_rest: 
        print("Including resting category...")
        # get TR intervals for rest between stims (stim_on == 2)
        rest_bools = stim_on == 2
        padded_bools = np.r_[False, rest_bools, False]  # pad the bools at beginning and end for diff to operate
        rest_diff = np.diff(padded_bools)  # get the pairwise diff in the array --> True for start and end indices of rest periods
        rest_intervals = rest_diff.nonzero()[0].reshape((-1,2))  # each pair is the interval of rest periods
        print("random sample rest_intervals: ", rest_intervals)
        exit()

        # get desired time points: can be changed to be middle/end of resting periods, or just random subsample
        # current choice: get time points in the middle of rest periods for rest samples; if 0.5, round up
        rest_inds = [np.ceil(np.average(interval)).astype(int) for interval in rest_intervals]  

        # subsample to min_n
        chosen_rest_inds = np.random.choice(rest_inds, min_n, replace=False)
        sampled_inds[0] = sorted(chosen_rest_inds)

    # ===== stack indices
    X = []
    Y = []
    for lab, inds in sampled_inds.items():
        X.append(full_data[inds, :])
        Y.append(np.zeros(len(inds)) + lab)

    X = np.vstack(X)
    Y = np.concatenate(Y)
    return X, Y, _


def subsample_by_runs(full_data, label_df, include_rest=True):
    """
    Subsample data by runs. Yield splits or all combination of 2 runs.
    Return: stacked X & Y for train/test split & model fitting
    """ 

    # stim_list: 1 Scenes, 2 Faces / 0 is rest
    # stim_on labels: 1 actual stim; 2 rest between stims; 0 rest between runs
    print("\n***** Subsampling data points by runs...")

    stim_list = label_df['category']
    stim_on = label_df['stim_present']
    run_list = label_df['run']

    # get faces
    face_inds = np.where((stim_on == 1) & (stim_list == 2))[0]
    rest_inds = []
    groups = np.concatenate([np.full(int(len(face_inds)/2), 1), np.full(int(len(face_inds)/2), 2)])

    scenes_runs = [3,4,5,6]
    for i in range(len(scenes_runs)):
        runi = scenes_runs[i]
        for j in range(i+1, len(scenes_runs)):
            runj = scenes_runs[j]
            print(f"\nSubsampling scenes with runs {runi} & {runj}...")
            
            # choose scene samples based on run
            scene_inds = np.where((stim_on == 1) & (stim_list == 1) & 
                                    ((run_list == runi) | (run_list == runj)))[0] # actual stim; stim is scene; stim in the two runs
            
            if include_rest:
                print("Including resting category...")
                # get TR intervals for rest between stims (stim_on == 2)
                rest_bools = ((run_list == runi) | (run_list == runj)) & (stim_on == 2)
                padded_bools = np.r_[False, rest_bools, False]  # pad the bools at beginning and end for diff to operate
                rest_diff = np.diff(padded_bools)  # get the pairwise diff in the array --> True for start and end indices of rest periods
                rest_intervals = rest_diff.nonzero()[0].reshape((-1,2))  # each pair is the interval of rest periods

                # get desired time points: can be changed to be middle/end of resting periods, or just random subsample
                # current choice: get time points in the middle of rest periods for rest samples; if 0.5, round up
                rest_intervals[:,-1] -= 1
                rest_inds = [np.ceil(np.average(interval)).astype(int) for interval in rest_intervals] + \
                            [np.ceil(np.average(interval)).astype(int)+1 for interval in rest_intervals]

                # should give same number of rest samples; if not, do random sample
                # rest_inds = np.random.choice(rest_inds, len(face_inds), replace=False)

            # === get X & Y
            X = []
            Y = []
            print(f"rest_inds: {len(rest_inds)}, scene_inds: {len(scene_inds)}, face_inds: {len(face_inds)}")
            for lab, inds in zip([0,1,2], [rest_inds, scene_inds, face_inds]):
                print("label counts:", lab, len(inds))
                X.append(full_data[inds, :])
                Y.append(np.zeros(len(inds)) + lab)

            X = np.vstack(X)
            Y = np.concatenate(Y)
            all_groups = np.concatenate([groups, groups, groups])
            yield X, Y, all_groups

            # flip groups so even & odd groups can be paired
            all_groups = np.concatenate([groups, list(reversed(groups)), list(reversed(groups))])
            yield X, Y, all_groups

def confound_cleaner(confounds):
    COI = ['a_comp_cor_00','a_comp_cor_01','a_comp_cor_02','a_comp_cor_03','a_comp_cor_05','framewise_displacement','trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']
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
    container_path='/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep'
  
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

    if mask_flag=='vtc':
        vtc_mask_path=os.path.join('/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/',sub,'new_mask','VVS_preremoval_%s_mask.nii.gz' % brain_flag)
        
        vtc_mask=nib.load(vtc_mask_path)
    


    if clear_data==0:

        if os.path.exists(os.path.join(bold_path,'sub-0%s_%s_preremoval_%s_masked.npy' % (sub_num,brain_flag,mask_flag))):
            
            localizer_bold=np.load(os.path.join(bold_path,'sub-0%s_%s_preremoval_%s_masked.npy' % (sub_num,brain_flag,mask_flag)))
            print('%s %s Localizer Loaded...' % (brain_flag,mask_flag))
            run1_length=int((len(localizer_bold)/6))
            run2_length=int((len(localizer_bold)/6))
            run3_length=int((len(localizer_bold)/6))
            run4_length=int((len(localizer_bold)/6))
            run5_length=int((len(localizer_bold)/6))
            run6_length=int((len(localizer_bold)/6))            
        else:
            #select the specific file
            localizer_run1=nib.load(localizer_files[0])
            localizer_run2=nib.load(localizer_files[1])
            localizer_run3=nib.load(localizer_files[2])
            localizer_run4=nib.load(localizer_files[3])
            localizer_run5=nib.load(localizer_files[4])
            localizer_run6=nib.load(localizer_files[5])          
              
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

            confound_run1=confound_run1.fillna(confound_run1.mean())
            confound_run2=confound_run2.fillna(confound_run2.mean())
            confound_run3=confound_run3.fillna(confound_run3.mean())
            confound_run4=confound_run4.fillna(confound_run4.mean())
            confound_run5=confound_run5.fillna(confound_run5.mean())
            confound_run6=confound_run6.fillna(confound_run6.mean())                                    

            wholebrain_mask1=nib.load(brain_mask_path[0])
            wholebrain_mask2=nib.load(brain_mask_path[1])
            wholebrain_mask3=nib.load(brain_mask_path[2])
            wholebrain_mask4=nib.load(brain_mask_path[3])
            wholebrain_mask5=nib.load(brain_mask_path[4])
            wholebrain_mask6=nib.load(brain_mask_path[5])

            
            def apply_mask(mask=None,target=None):
                coor = np.where(mask == 1)
                values = target[coor]
                if values.ndim > 1:
                    values = np.transpose(values) #swap axes to get feature X sample
                return values

            if mask_flag=='wholebrain':
                localizer_run1=apply_mask(mask=(wholebrain_mask1.get_data()),target=(localizer_run1.get_data()))
                localizer_run2=apply_mask(mask=(wholebrain_mask1.get_data()),target=(localizer_run2.get_data()))
                localizer_run3=apply_mask(mask=(wholebrain_mask1.get_data()),target=(localizer_run3.get_data()))
                localizer_run4=apply_mask(mask=(wholebrain_mask1.get_data()),target=(localizer_run4.get_data()))
                localizer_run5=apply_mask(mask=(wholebrain_mask1.get_data()),target=(localizer_run5.get_data()))
                localizer_run6=apply_mask(mask=(wholebrain_mask1.get_data()),target=(localizer_run6.get_data()))                        

            elif mask_flag=='vtc':
                localizer_run1=apply_mask(mask=(vtc_mask.get_data()),target=(localizer_run1.get_data()))
                localizer_run2=apply_mask(mask=(vtc_mask.get_data()),target=(localizer_run2.get_data()))
                localizer_run3=apply_mask(mask=(vtc_mask.get_data()),target=(localizer_run3.get_data()))
                localizer_run4=apply_mask(mask=(vtc_mask.get_data()),target=(localizer_run4.get_data()))
                localizer_run5=apply_mask(mask=(vtc_mask.get_data()),target=(localizer_run5.get_data()))   
                localizer_run6=apply_mask(mask=(vtc_mask.get_data()),target=(localizer_run6.get_data()))   

            preproc_1 = clean(localizer_run1,t_r=1,detrend=False,standardize='zscore')
            preproc_2 = clean(localizer_run2,t_r=1,detrend=False,standardize='zscore')
            preproc_3 = clean(localizer_run3,t_r=1,detrend=False,standardize='zscore')
            preproc_4 = clean(localizer_run4,t_r=1,detrend=False,standardize='zscore')
            preproc_5 = clean(localizer_run5,t_r=1,detrend=False,standardize='zscore')
            preproc_6 = clean(localizer_run6,t_r=1,detrend=False,standardize='zscore')                       


            localizer_bold=np.concatenate((preproc_1,preproc_2,preproc_3,preproc_4,preproc_5,preproc_6))
            #save this data if we didn't have it saved before
            os.chdir(bold_path)
            np.save('sub-0%s_%s_preremoval_%s_masked' % (sub_num,brain_flag,mask_flag), localizer_bold)
            print('%s %s masked data...saved' % (mask_flag,brain_flag))

    else:
        if ((os.path.exists(os.path.join(bold_path,'sub-0%s_%s_preremoval_%s_masked_cleaned.npy' % (sub_num,brain_flag,mask_flag)))) & (force_clean==0)):
            
            localizer_bold=np.load(os.path.join(bold_path,'sub-0%s_%s_preremoval_%s_masked_cleaned.npy' % (sub_num,brain_flag,mask_flag)))
            print('%s %s Cleaned Localizer Loaded...' % (brain_flag,mask_flag))
            run1_length=int((len(localizer_bold)/6))
            run2_length=int((len(localizer_bold)/6))
            run3_length=int((len(localizer_bold)/6))
            run4_length=int((len(localizer_bold)/6))
            run5_length=int((len(localizer_bold)/6))
            run6_length=int((len(localizer_bold)/6))            
        else:
            #select the specific file
            localizer_run1=nib.load(localizer_files[0])
            localizer_run2=nib.load(localizer_files[1])
            localizer_run3=nib.load(localizer_files[2])
            localizer_run4=nib.load(localizer_files[3])
            localizer_run5=nib.load(localizer_files[4])
            localizer_run6=nib.load(localizer_files[5])          
              
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

            wholebrain_mask1=nib.load(brain_mask_path[0])
            wholebrain_mask2=nib.load(brain_mask_path[1])
            wholebrain_mask3=nib.load(brain_mask_path[2])
            wholebrain_mask4=nib.load(brain_mask_path[3])
            wholebrain_mask5=nib.load(brain_mask_path[4])
            wholebrain_mask6=nib.load(brain_mask_path[5])

            
            def apply_mask(mask=None,target=None):
                coor = np.where(mask == 1)
                values = target[coor]
                if values.ndim > 1:
                    values = np.transpose(values) #swap axes to get feature X sample
                return values

            if mask_flag=='wholebrain':
                localizer_run1=apply_mask(mask=(wholebrain_mask1.get_data()),target=(localizer_run1.get_data()))
                localizer_run2=apply_mask(mask=(wholebrain_mask1.get_data()),target=(localizer_run2.get_data()))
                localizer_run3=apply_mask(mask=(wholebrain_mask1.get_data()),target=(localizer_run3.get_data()))
                localizer_run4=apply_mask(mask=(wholebrain_mask1.get_data()),target=(localizer_run4.get_data()))
                localizer_run5=apply_mask(mask=(wholebrain_mask1.get_data()),target=(localizer_run5.get_data()))
                localizer_run6=apply_mask(mask=(wholebrain_mask1.get_data()),target=(localizer_run6.get_data()))                        

            elif mask_flag=='vtc':
                localizer_run1=apply_mask(mask=(vtc_mask.get_data()),target=(localizer_run1.get_data()))
                localizer_run2=apply_mask(mask=(vtc_mask.get_data()),target=(localizer_run2.get_data()))
                localizer_run3=apply_mask(mask=(vtc_mask.get_data()),target=(localizer_run3.get_data()))
                localizer_run4=apply_mask(mask=(vtc_mask.get_data()),target=(localizer_run4.get_data()))
                localizer_run5=apply_mask(mask=(vtc_mask.get_data()),target=(localizer_run5.get_data()))   
                localizer_run6=apply_mask(mask=(vtc_mask.get_data()),target=(localizer_run6.get_data()))   

            elif mask_flag=='PHG':
                localizer_run1=apply_mask(mask=(PHG_mask.get_data()),target=(localizer_run1.get_data()))
                localizer_run2=apply_mask(mask=(PHG_mask.get_data()),target=(localizer_run2.get_data()))
                localizer_run3=apply_mask(mask=(PHG_mask.get_data()),target=(localizer_run3.get_data()))
                localizer_run4=apply_mask(mask=(PHG_mask.get_data()),target=(localizer_run4.get_data()))
                localizer_run5=apply_mask(mask=(PHG_mask.get_data()),target=(localizer_run5.get_data()))   
                localizer_run6=apply_mask(mask=(PHG_mask.get_data()),target=(localizer_run6.get_data()))                         
            
            elif mask_flag=='FG':    
                localizer_run1=apply_mask(mask=(FG_mask.get_data()),target=(localizer_run1.get_data()))
                localizer_run2=apply_mask(mask=(FG_mask.get_data()),target=(localizer_run2.get_data()))
                localizer_run3=apply_mask(mask=(FG_mask.get_data()),target=(localizer_run3.get_data()))
                localizer_run4=apply_mask(mask=(FG_mask.get_data()),target=(localizer_run4.get_data()))
                localizer_run5=apply_mask(mask=(FG_mask.get_data()),target=(localizer_run5.get_data()))   
                localizer_run6=apply_mask(mask=(FG_mask.get_data()),target=(localizer_run6.get_data()))                         

            preproc_1 = clean(localizer_run1,confounds=(confound_run1),t_r=1,detrend=False,standardize='zscore')
            preproc_2 = clean(localizer_run2,confounds=(confound_run2),t_r=1,detrend=False,standardize='zscore')
            preproc_3 = clean(localizer_run3,confounds=(confound_run3),t_r=1,detrend=False,standardize='zscore')
            preproc_4 = clean(localizer_run4,confounds=(confound_run4),t_r=1,detrend=False,standardize='zscore')
            preproc_5 = clean(localizer_run5,confounds=(confound_run5),t_r=1,detrend=False,standardize='zscore')
            preproc_6 = clean(localizer_run6,confounds=(confound_run6),t_r=1,detrend=False,standardize='zscore')                       


            run1_length=int((len(localizer_run1)))
            run2_length=int((len(localizer_run2)))
            run3_length=int((len(localizer_run3)))
            run4_length=int((len(localizer_run4)))
            run5_length=int((len(localizer_run5)))
            run6_length=int((len(localizer_run6)))


            localizer_bold=np.concatenate((preproc_1,preproc_2,preproc_3,preproc_4,preproc_5,preproc_6))
            #save this data if we didn't have it saved before
            os.chdir(bold_path)
            np.save('sub-0%s_%s_preremoval_%s_masked_cleaned' % (sub_num,brain_flag,mask_flag), localizer_bold)
            print('%s %s masked and cleaned localizer data...saved' % (mask_flag,brain_flag))
           
    #fill in the run array with run number
    run1=np.full(run1_length,1)
    run2=np.full(run2_length,2)
    run3=np.full(run3_length,3)
    run4=np.full(run4_length,4)
    run5=np.full(run5_length,5)    
    run6=np.full(run6_length,6)    

    run_list=np.concatenate((run1,run2,run3,run4,run5,run6)) #now combine

    #load regs / labels

    #Categories: 1 Scenes, 2 Faces 

    params_dir='/scratch/06873/zbretton/repclear_dataset/BIDS/params'
    #find the mat file, want to change this to fit "sub"
    param_search='preremoval*events*.csv'
    param_file=find(param_search,params_dir)

    reg_matrix = pd.read_csv(param_file[0])
    reg_category=reg_matrix["category"].values
    reg_stim_on=reg_matrix["stim_present"].values
    reg_run=reg_matrix["run"].values

    run1_index=np.where(reg_run==1)
    run2_index=np.where(reg_run==2)
    run3_index=np.where(reg_run==3)
    run4_index=np.where(reg_run==4)
    run5_index=np.where(reg_run==5)
    run6_index=np.where(reg_run==6)            

    stim1_index=len(run1)
    stim2_index=(stim1_index+len(run2))
    stim3_index=stim2_index+len(run3)
    stim4_index=stim3_index+len(run4)
    stim5_index=stim4_index+len(run5)

#extract times where stimuli is on for both categories:
    #stim_on=np.where((reg_stim_on==1) & ((reg_category==1) | (reg_category==2)))
    stim_on=reg_stim_on
#need to convert this list to 1-d
    stim_list=np.empty(len(localizer_bold))
    stim_list=reg_category
#this list is now 1d list, need to add a dimentsionality to it
    stim_list=stim_list[:,None]
    stim_on=stim_on[:,None]
        

        # Create a function to shift the size, and will do the rest tag
    def shift_timing(label_TR, TR_shift_size, tag):
        # Create a short vector of extra zeros or whatever the rest label is
        zero_shift = np.full(TR_shift_size, tag)
        # Zero pad the column from the top
        zero_shift = np.vstack(zero_shift)
        label_TR_shifted = np.vstack((zero_shift, label_TR))
        # Don't include the last rows that have been shifted out of the time line
        label_TR_shifted = label_TR_shifted[0:label_TR.shape[0],0]
      
        return label_TR_shifted

# Apply the function
    # Apply the function
    shift_size = TR_shift #this is shifting by 5TR
    tag = 0 #rest label is 0
    stim_list_shift = shift_timing(stim_list, shift_size, tag) #rest is label 0
    stim_on_shift= shift_timing(stim_on, shift_size, tag)
    import random
    
    #Here I need to balance the trials of the categories / rest. I will be removing rest, but Scenes have twice the data of faces, so need to decide how to handle
    rest_times=np.where(stim_list_shift==0)
    rest_times_on=np.where(stim_on_shift==0)
    #these should be the same, but doing it just in case

    rest_btwn_stims=np.where((stim_on_shift==2) & ((stim_list_shift==1) | (stim_list_shift==2)))
    rest_btwn_stims_filt=rest_btwn_stims[0][2::6] # start at the 3rd index and then take each 6th item from this list of indicies
    #this above line results in 180 samples, we need to bring it down to 120 to match the samples of faces (minimum samples)
    #this works well with the current situation since we are not using the last 2 runs in the x-validation because of uneven samples

    # this condition is going to be rebuild to include the resting times, so that the two versions being tested are stimuli on alone and then stimuli on + balanced amount of rest
    stims_on=np.where((stim_on_shift==1) & ((stim_list_shift==1) | (stim_list_shift==2))) #get times where stims are on
    temp_shift=stim_list_shift
    temp_shift[rest_btwn_stims_filt]=0 #using the targeted rest times, I am setting them to 0 (rest's label) since for now they are still labeled as the category of that trial
    stims_and_rest=np.concatenate((rest_btwn_stims_filt,stims_on[0])) #these are all the events we need, faces(120), scenes(240) and rest(180)
    stims_and_rest.sort() #put this in order so we can sample properly
    stim_on_rest=temp_shift[stims_and_rest]
    localizer_bold_stims_and_rest=localizer_bold[stims_and_rest]
    run_list_stims_and_rest=run_list[stims_and_rest]

    #sorted only for stimuli being on
    stim_on_filt=stim_list_shift[np.where((stim_on_shift==1) & ((stim_list_shift==1) | (stim_list_shift==2)))]
    stim_on_filt=stim_on_filt.flatten()
    #stim_on_filt=stim_on_filt[:,None] #this seems unneeded at the moment
    localizer_bold_on_filt=localizer_bold[np.where((stim_on_shift==1) & ((stim_list_shift==1) | (stim_list_shift==2)))]
    run_list_on_filt=run_list[np.where((stim_on_shift==1) & ((stim_list_shift==1) | (stim_list_shift==2)))]


    bold_files=find('*study*bold*.nii.gz',bold_path)
    wholebrain_mask_path=find('*study*mask*.nii.gz',bold_path)
    pattern = '*MNI*'
    pattern2 = '*MNI*preproc*'
    brain_mask_path = fnmatch.filter(wholebrain_mask_path,pattern)
    study_files = fnmatch.filter(bold_files,pattern2)

    brain_mask_path.sort()
    study_files.sort()


    if clear_data==0:


        if os.path.exists(os.path.join(bold_path,'sub-0%s_%s_study_%s_masked.npy' % (sub_num,brain_flag,mask_flag))):
            
            study_bold=np.load(os.path.join(bold_path,'sub-0%s_%s_study_%s_masked.npy' % (sub_num,brain_flag,mask_flag)))
            print('%s %s study Loaded...' % (brain_flag,mask_flag))
            run1_length=int((len(study_bold)/3))
            run2_length=int((len(study_bold)/3))
            run3_length=int((len(study_bold)/3))           
        else:
            #select the specific file
            study_run1=nib.load(study_files[0])
            study_run2=nib.load(study_files[1])
            study_run3=nib.load(study_files[2])

            #to be used to filter the data
            #First we are removing the confounds
            #get all the folders within the bold path
            study_confounds_1=find('*study*1*confounds*.tsv',bold_path)
            study_confounds_2=find('*study*2*confounds*.tsv',bold_path)
            study_confounds_3=find('*study*3*confounds*.tsv',bold_path)
            
            confound_run1 = pd.read_csv(study_confounds_1[0],sep='\t')
            confound_run2 = pd.read_csv(study_confounds_2[0],sep='\t')
            confound_run3 = pd.read_csv(study_confounds_3[0],sep='\t')

            wholebrain_mask1=nib.load(brain_mask_path[0])
            
            def apply_mask(mask=None,target=None):
                coor = np.where(mask == 1)
                values = target[coor]
                if values.ndim > 1:
                    values = np.transpose(values) #swap axes to get feature X sample
                return values
            if mask_flag=='wholebrain':
                study_run1=apply_mask(mask=(wholebrain_mask1.get_data()),target=(study_run1.get_data()))
                study_run2=apply_mask(mask=(wholebrain_mask1.get_data()),target=(study_run2.get_data()))
                study_run3=apply_mask(mask=(wholebrain_mask1.get_data()),target=(study_run3.get_data()))                    

            elif mask_flag=='vtc':
                whole_study_run1=apply_mask(mask=(wholebrain_mask1.get_data()),target=(study_run1.get_data()))
                whole_study_run2=apply_mask(mask=(wholebrain_mask1.get_data()),target=(study_run2.get_data()))
                whole_study_run3=apply_mask(mask=(wholebrain_mask1.get_data()),target=(study_run3.get_data()))

                study_run1=apply_mask(mask=(vtc_mask.get_data()),target=(study_run1.get_data()))
                study_run2=apply_mask(mask=(vtc_mask.get_data()),target=(study_run2.get_data()))
                study_run3=apply_mask(mask=(vtc_mask.get_data()),target=(study_run3.get_data()))

            elif mask_flag=='GM':
                study_run1=apply_mask(mask=(gm_mask.get_data()),target=(study_run1.get_data()))
                study_run2=apply_mask(mask=(gm_mask.get_data()),target=(study_run2.get_data()))
                study_run3=apply_mask(mask=(gm_mask.get_data()),target=(study_run3.get_data()))

            preproc_1 = clean(study_run1,t_r=1,detrend=False,standardize='zscore')
            preproc_2 = clean(study_run2,t_r=1,detrend=False,standardize='zscore')
            preproc_3 = clean(study_run3,t_r=1,detrend=False,standardize='zscore')


            study_bold=np.concatenate((preproc_1,preproc_2,preproc_3))
            #save this data if we didn't have it saved before
            os.chdir(bold_path)
            np.save('sub-0%s_%s_study_%s_masked' % (sub_num,brain_flag,mask_flag), study_bold)
            print('%s %s masked data...saved' % (mask_flag,brain_flag))

        #create run array
            run1_length=int((len(study_run1)))
            run2_length=int((len(study_run2)))
            run3_length=int((len(study_run3)))
    else: 
        if (os.path.exists(os.path.join(bold_path,'sub-0%s_%s_study_%s_masked_cleaned.npy' % (sub_num,brain_flag,mask_flag))) & (force_clean==0)):
            
            study_bold=np.load(os.path.join(bold_path,'sub-0%s_%s_study_%s_masked_cleaned.npy' % (sub_num,brain_flag,mask_flag)))
            print('%s %s study Loaded...' % (brain_flag,mask_flag))
            run1_length=int((len(study_bold)/3))
            run2_length=int((len(study_bold)/3))
            run3_length=int((len(study_bold)/3))           
        else:
            #select the specific file
            study_run1=nib.load(study_files[0])
            study_run2=nib.load(study_files[1])
            study_run3=nib.load(study_files[2])

            #to be used to filter the data
            #First we are removing the confounds
            #get all the folders within the bold path
            study_confounds_1=find('*study*1*confounds*.tsv',bold_path)
            study_confounds_2=find('*study*2*confounds*.tsv',bold_path)
            study_confounds_3=find('*study*3*confounds*.tsv',bold_path)
            
            confound_run1 = pd.read_csv(study_confounds_1[0],sep='\t')
            confound_run2 = pd.read_csv(study_confounds_2[0],sep='\t')
            confound_run3 = pd.read_csv(study_confounds_3[0],sep='\t')

            confound_run1=confound_cleaner(confound_run1)
            confound_run2=confound_cleaner(confound_run2)
            confound_run3=confound_cleaner(confound_run3)

            wholebrain_mask1=nib.load(brain_mask_path[0])
            
            def apply_mask(mask=None,target=None):
                coor = np.where(mask == 1)
                values = target[coor]
                if values.ndim > 1:
                    values = np.transpose(values) #swap axes to get feature X sample
                return values
            if mask_flag=='wholebrain':
                study_run1=apply_mask(mask=(wholebrain_mask1.get_data()),target=(study_run1.get_data()))
                study_run2=apply_mask(mask=(wholebrain_mask1.get_data()),target=(study_run2.get_data()))
                study_run3=apply_mask(mask=(wholebrain_mask1.get_data()),target=(study_run3.get_data()))                    

            elif mask_flag=='vtc':
                whole_study_run1=apply_mask(mask=(wholebrain_mask1.get_data()),target=(study_run1.get_data()))
                whole_study_run2=apply_mask(mask=(wholebrain_mask1.get_data()),target=(study_run2.get_data()))
                whole_study_run3=apply_mask(mask=(wholebrain_mask1.get_data()),target=(study_run3.get_data()))

                study_run1=apply_mask(mask=(vtc_mask.get_data()),target=(study_run1.get_data()))
                study_run2=apply_mask(mask=(vtc_mask.get_data()),target=(study_run2.get_data()))
                study_run3=apply_mask(mask=(vtc_mask.get_data()),target=(study_run3.get_data()))

            elif mask_flag=='GM':
                study_run1=apply_mask(mask=(gm_mask.get_data()),target=(study_run1.get_data()))
                study_run2=apply_mask(mask=(gm_mask.get_data()),target=(study_run2.get_data()))
                study_run3=apply_mask(mask=(gm_mask.get_data()),target=(study_run3.get_data()))                        

            preproc_1 = clean(study_run1,confounds=(confound_run1),t_r=1,detrend=False,standardize='zscore')
            preproc_2 = clean(study_run2,confounds=(confound_run2),t_r=1,detrend=False,standardize='zscore')
            preproc_3 = clean(study_run3,confounds=(confound_run3),t_r=1,detrend=False,standardize='zscore')


            study_bold=np.concatenate((preproc_1,preproc_2,preproc_3))
            #save this data if we didn't have it saved before
            os.chdir(bold_path)
            np.save('sub-0%s_%s_study_%s_masked_cleaned' % (sub_num,brain_flag,mask_flag), study_bold)
            print('%s %s masked and cleaned Study data...saved' % (mask_flag,brain_flag))

    params_dir='/scratch/06873/zbretton/repclear_dataset/BIDS/params'
    #find the mat file, want to change this to fit "sub"
    param_search='study*events*.csv'
    param_file=find(param_search,params_dir)

    study_matrix = pd.read_csv(param_file[0])
    study_operation=study_matrix["condition"].values
    study_run=study_matrix["run"].values
    study_present=study_matrix["stim_present"].values
    study_operation_trial=study_matrix["cond_trial"].values

    run1_index=np.where(study_run==1)
    run2_index=np.where(study_run==2)
    run3_index=np.where(study_run==3)
          
#need to convert this list to 1-d                           
#this list is now 1d list, need to add a dimentsionality to it
#Condition:
#1. maintain
#2. replace_category
#3. suppress
    study_stim_list=np.full(len(study_bold),0)
    maintain_list=np.where((study_operation==1) & ((study_present==1) |(study_present==2) | (study_present==3)))
    suppress_list=np.where((study_operation==3) & ((study_present==1) |(study_present==2) | (study_present==3)))
    replace_list=np.where((study_operation==2) & ((study_present==1) |(study_present==2) | (study_present==3)))
    study_stim_list[maintain_list]=1
    study_stim_list[suppress_list]=3
    study_stim_list[replace_list]=2

    oper_list=study_operation
    oper_list=oper_list[:,None]

    study_stim_list=study_stim_list[:,None]
    study_operation_trial=study_operation_trial[:,None]          

    #trim the training to the first 4 localizer runs, to match samples and with rest
    run1_4=np.where((run_list_stims_and_rest<=4))
    localizer_bold_14=localizer_bold_stims_and_rest[run1_4]
    stim_on_14=stim_on_rest[run1_4]  

    task = 'preremoval'
    space = 'T1w'
    ROIs = ['VVS'] #'LatOcc'
    shift_size_TR = shift_sizes_TR[0]
    rest_tag = 0

    full_data = get_preprocessed_data(subID, task, space, ROIs, save=True)
    print(f"Full_data shape: {full_data.shape}")

    # get labels
    label_df = get_shifted_labels(task, shift_size_TR, rest_tag)
    print(f"Category label shape: {label_df.shape}")  

    for X, Y, groups in subsample_by_runs(full_data, label_df):
        X=X
        Y=Y
        groups=groups

    #do a L2 estimator
    def CLF(train_data, train_labels, test_data, test_labels, k_best):
        scores = []
        predicts = []
        trues = []
        decisions=[]
        evidences=[]
        predict_probs=[] #adding this to test against the decision function
        sig_scores=[]
        C_best=[]

        X_train, X_test = train_data, test_data
        y_train, y_test = train_labels, test_labels

        #selectedvoxels=SelectKBest(f_classif,k=1500).fit(X_train,y_train)
        selectedvoxels=SelectFpr(f_classif,alpha=0.01).fit(X_train,y_train) #I compared this method to taking ALL k items in the F-test and filtering by p-value, so I assume this is a better feature selection method

        X_train=selectedvoxels.transform(X_train)
        X_test=selectedvoxels.transform(X_test)

        parameters ={'C':[0.01,0.1,1,10,100,1000]}
        inner_clf = GridSearchCV(
            LogisticRegression(penalty='l2',solver='liblinear'),
            parameters,
            cv=4,
            return_train_score=True)
        inner_clf.fit(X_train,y_train)
        C_best_i = inner_clf.best_params_['C']
        C_best.append(C_best_i)


        clf=LogisticRegression(penalty='l2',solver='liblinear',C=C_best_i,multi_class='ovr')

        # fit the model
        clf.fit(X_train, y_train)
        
        #output decision values
        decisions=clf.decision_function(X_test)

        evidence=(1. / (1. + np.exp(-clf.decision_function(X_test))))
        evidences.append(evidence)
        predict_prob=clf.predict_proba(X_test)
        predict_probs.append(predict_prob)
     
        
        # score the model, but we care more about values
        score=clf.score(X_test, y_test)
        predict = clf.predict(X_test)
        predicts.append(predict)
        true = y_test

        #sig_score=roc_auc_score(true,clf.predict_proba(X_test), multi_class='ovr')
        #sig_scores.append(sig_score)           
        scores.append(score)
        trues.append(true)
        return clf, scores, predicts, trues, decisions, evidence, predict_probs

#    L2_models_nr, L2_scores_nr, L2_predicts_nr, L2_trues_nr, L2_decisions_nr, L2_evidence_nr = CLF(localizer_bold_on_filt, stim_on_filt, study_bold, study_stim_list, 1500)
#    L2_subject_score_nr_mean = np.mean(L2_scores_nr)                                        
    L2_models, L2_scores, L2_predicts, L2_trues, L2_decisions, L2_evidence, L2_predict_probs = CLF(X, Y, study_bold, study_stim_list, 3000)
    L2_subject_score_mean = np.mean(L2_scores) 

    #train on ALL data
    # L2_models, L2_scores, L2_predicts, L2_trues, L2_decisions, L2_evidence, L2_predict_probs = CLF(localizer_bold_stims_and_rest, stim_on_rest, study_bold, study_stim_list, 3000)
    # L2_subject_score_mean = np.mean(L2_scores) 


    np.savetxt("%s_train_localizer_test_study_category_evidence.csv" % brain_flag,L2_evidence, delimiter=",")
    np.savetxt("%s_train_localizer_test_study_category_predictprob.csv"% brain_flag,L2_predict_probs[0], delimiter=",")    
    np.savetxt("%s_Operation_labels.csv"% brain_flag,study_stim_list, delimiter=",")
    np.savetxt("%s_Operation_trials.csv"% brain_flag,study_operation_trial, delimiter=",")

    output_table = {
        "subject" : sub,

        "CLF Average Scores from Testing" : L2_subject_score_mean,
        "CLF Model Testing" : L2_models,
        "CLF Model Decisions" : L2_decisions,

        "CLF Category Evidece" : L2_evidence,

        "CLF Operation Trues" : L2_trues,
        "CLF Predict Probs" : L2_predict_probs,
        "CLF Score" : L2_scores,
        "CLF Predictions" : L2_predicts,
        "CLF True" : L2_trues,

        "Category List Shifted w/ Rest" : stim_on_filt,
        "Operation List": study_stim_list,
        
        "Localizer Shifted w/ Rest": localizer_bold_on_filt,
        
        }
    
    import pickle
    os.chdir(os.path.join(container_path,sub))
    f = open("%s-train_localizer_test_study_%s_%sTR lag_data.pkl" % (sub,brain_flag,TR_shift),"wb")
    pickle.dump(output_table,f)
    f.close()    