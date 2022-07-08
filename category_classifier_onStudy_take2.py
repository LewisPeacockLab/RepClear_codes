#recoding my category classifier / decoding of study phase - pipeline
import warnings
import sys
# if not sys.warnoptions:
#     warnings.simplefilter("ignore")
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nibabel as nib
from nilearn.image import clean_img
from nilearn.signal import clean
import scipy as scipy
import scipy.io as sio
import seaborn as sns
cmap = sns.color_palette("crest", as_cmap=True)
import fnmatch
import pickle
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut  #train_test_split, PredefinedSplit, cross_validate, cross_val_predict, 
from sklearn.feature_selection import SelectFpr, f_classif  #VarianceThreshold, SelectKBest, 
from sklearn.preprocessing import StandardScaler
from nilearn.input_data import NiftiMasker,  MultiNiftiMasker
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from joblib import Parallel, delayed


# global consts
subIDs = ['002','003','004','005','006','007','008','009','010','011','012','013','014','015','016','017','018','020','023','024','025','026']


phases = ['rest', 'preremoval', 'study', 'postremoval']
runs = np.arange(6) + 1  
spaces = {'T1w': 'T1w', 
            'MNI': 'MNI152NLin2009cAsym'}
descs = ['brain_mask', 'preproc_bold']
ROIs = ['vtc']
shift_sizes_TR = [5]

save=1
shift_size_TR = shift_sizes_TR[0]


stim_labels = {0: "Rest",
                1: "Scenes",
                2: "Faces"}

workspace = 'scratch'
data_dir = '/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/'
param_dir = '/scratch/06873/zbretton/repclear_dataset/BIDS/params/'

#function to load in the confounds file for each run and then select the columns we want for cleaning
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

def load_existing_data(): #try to find if the data we want has already been cleaned and saved...
    print("\n*** Attempting to load existing data if there is any...")
    preproc_data = {}
    todo_ROIs = []

    bold_dir = os.path.join(data_dir, f"sub-{subID}", "func")
    out_fname_template = f"sub-{subID}_{space}_{task}_{{}}_masked_cleaned.npy"
    for ROI in mask_ROIS:
        if os.path.exists(os.path.join(bold_dir, out_fname_template.format(ROI))):
            print("\nLoading saved preprocessed data", out_fname_template.format(ROI), '...')
            preproc_data[ROI] = np.load(os.path.join(bold_dir, out_fname_template.format(ROI)))
        else: 
            if ROI == 'vtc': ROI = 'VVS'  # change to load masks...
            print(f"\nROI {ROI} data to be processed.")
            todo_ROIs.append(ROI)    
    return preproc_data, todo_ROIs

def load_process_data(): #this wraps the above function, with a process to load and save the data if there isnt an already saved .npy file
    # ========== check & load existing files
    ready_data, mask_ROIS = load_existing_data()
    if type(mask_ROIS) == list and len(mask_ROIS) == 0: 
        return np.vstack(list(ready_data.values()))
    else: 
        print("\nPreprocessing ROIs", mask_ROIS)


    print(f"\n***** Data preprocessing for sub {subID} {task} {space} with ROIs {todo_ROIs}...")

    space_long = spaces[space]

    # ========== start from scratch for todo_ROIs
    # ======= generate file names to load
    # get list of data names
    fname_template = f"sub-{subID}_task-{task}_run-{{}}_space-{space_long}_desc-{{}}.nii.gz"
    bold_fnames = [fname_template.format(i, "preproc_bold") for i in runs]
    bold_paths = [os.path.join(data_dir, f"sub-{subID}", "func", fname) for fname in bold_fnames]    

    # get mask names
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

    # ======= load bold data & preprocess

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
            bold_dir = os.path.join(data_dir, f"sub-{subID}", "func")            
            out_fname_template = f"sub-{subID}_{space}_{task}_{{}}_masked_cleaned.npy"            
            out_fname = out_fname_template.format(ROI)
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

    sub_design=(f"*{subID}*{temp_task}*tr*")
    sub_design_file=find(sub_design,subject_design_dir)
    sub_design_matrix = pd.read_csv(sub_design_file[0]) #this is the correct, TR by TR list of what happened during this subject's study phase

    shifted_df = shift_timing(sub_design_matrix, shift_size_TR, rest_tag)

    return shifted_df

def subsample_by_runs(full_data, label_df, include_rest=True):
    """
    Subsample data by runs. Yield all combinations of 2 runs.
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
            
            # choose scene samples based on runs in this subsample
            scene_inds = np.where((stim_on == 1) & (stim_list == 1) & 
                                    ((run_list == runi) | (run_list == runj)))[0] # actual stim; stim is scene; stim in the two runs
            
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

def fit_model(X, Y, groups, save=False, out_fname=None, v=False):
    if v: print("\n***** Fitting model...")

    scores = []
    auc_scores = []
    cms = []
    best_Cs = []
    trained_models=[]
    fprs=[]

    logo = LeaveOneGroupOut()
    for train_inds, test_inds in logo.split(X, Y, groups):
        X_train, X_test, y_train, y_test = X[train_inds], X[test_inds], Y[train_inds], Y[test_inds]
        
        # feature selection and transformation
        fpr = SelectFpr(f_classif, alpha=0.01).fit(X_train, y_train)
        X_train_sub = fpr.transform(X_train)
        X_test_sub = fpr.transform(X_test)

        # train & hyperparam tuning
        parameters ={'C':[.001, .01, .1, 1, 10, 100, 1000]}
        gscv = GridSearchCV(
            LogisticRegression(penalty='l2', solver='lbfgs',max_iter=1000),
            parameters,
            return_train_score=True)
        gscv.fit(X_train_sub, y_train)
        best_Cs.append(gscv.best_params_['C'])
        
        # refit with full data and optimal penalty value
        lr = LogisticRegression(penalty='l2', solver='lbfgs', C=best_Cs[-1],max_iter=1000)
        lr.fit(X_train_sub, y_train)
        # test on held out data
        score = lr.score(X_test_sub, y_test)
        auc_score = roc_auc_score(y_test, lr.predict_proba(X_test_sub), multi_class='ovr')
        preds = lr.predict(X_test_sub)

        # confusion matrix
        true_counts = np.asarray([np.sum(y_test == i) for i in stim_labels.keys()])
        cm = confusion_matrix(y_test, preds, labels=list(stim_labels.keys())) / true_counts[:,None] * 100

        scores.append(score)
        auc_scores.append(auc_score)
        cms.append(cm)
        trained_models.append(lr) #this alllows me to extract the trained model from this function, so I can then use this trained classifier to decode another phase
        fprs.append(fpr) #this allows me to to extract the feature selection, since if I want to carry this trained classifier into another phase, I will need to feature select that data in the same way
    scores = np.asarray(scores)
    auc_scores = np.asarray(auc_scores)
    cms = np.stack(cms)
    best_Cs = np.asarray(best_Cs)

    if v: print(f"\nClassifier score: \n"
        f"scores: {scores.mean()} +/- {scores.std()}\n"
        f"auc scores: {auc_scores.mean()} +/- {auc_scores.std()}\n"
        f"best Cs: {best_Cs}\n"
        f"average confution matrix:\n"
        f"{cms.mean(axis=0)}")

    return scores, auc_scores, cms, trained_models, fprs

def decode(trained_model,fpr,data_decode):
    print(f"\n ***** Decoding the category content in {task2}")

    featsel_data=fpr.transform(data_decode) #using the feature selection method from the trained model to select the proper voxels for decoding

    predictions=trained_model.predict(featsel_data) #get the predicted categories from running the model
    evidence=(1. / (1. + np.exp(-trained_model.decision_function(featsel_data)))) #MAIN: collect the evidence values of all timepoints (3 columns: 0-rest, 1-faces, 2-scenes)

    return predictions, evidence


def classification(subID):
    task = 'preremoval'
    task2 = 'study'
    space = 'T1w'
    ROIs = ['VVS']
    n_iters = 1


    shift_size_TR = shift_sizes_TR[0]
    rest_tag = 0

    print(f"\n***** Running category level classification for sub {subID} {task} {space} with ROIs {ROIs}...")

    # get data: all_ROI_vox x all_runs_time
    full_data = get_preprocessed_data(subID, task, space, ROIs, save=True)
    print(f"Full_data shape: {full_data.shape}")

    # get labels
    label_df = get_shifted_labels(task, shift_size_TR, rest_tag)
    print(f"Category label shape: {label_df.shape}")

    assert len(full_data) == len(label_df), \
        f"Length of data ({len(full_data)}) does not match Length of labels ({len(label_df)})!"
    
    # cross run xval
    scores = []
    auc_scores = []
    cms = []
    trained_models=[]
    fprs=[]
    
    for X, Y, groups in subsample_by_runs(full_data, label_df): 
        print(f"Running model fitting...")
        print("shape of X & Y:", X.shape, Y.shape)
        assert len(X) == len(Y), f"Length of X ({len(X)}) doesn't match length of Y({len(Y)})"

        # model fitting 
        results_fname = os.path.join(results_dir, f"sub-{subID}_task-{task}_space-{space}_{ROIs[0]}_lrxval.npz")
        score, auc_score, cm, trained_model, fpr = fit_model(X, Y, groups, save=False, v=True)
        
        scores.append(score)
        auc_scores.append(auc_score)
        cms.append(cm)
        trained_models.append(trained_model)
        fprs.append(fpr)

    scores = np.concatenate(scores)
    auc_scores = np.concatenate(auc_scores)
    cms = np.stack(cms)

    print(f"\n***** Running category level classification for sub {subID} {task2} {space} with ROIs {ROIs}...")

    # get data: all_ROI_vox x all_runs_time
    full_data2 = get_preprocessed_data(subID, task2, space, ROIs, save=True)
    print(f"Full_data shape: {full_data.shape}")

    # get labels
    label_df2 = get_shifted_labels(task2, shift_size_TR, rest_tag)
    print(f"Category label shape: {label_df2.shape}") 

    #will need to build in a real selection method, but for now can take the first versions
    predicts, evidence = decode(trained_models[0],fprs[0],full_data2) #this will decode the data using the already trained model and feature selection method

    #need to then save the evidence:
    #CODE HERE

    #after saving the evidence, we need to use the label_df2 to sort out the conditions so this can be plotted properly (and will also wanna sort based on remember vs. forgotten )

    # save results
    np.savez_compressed(results_fname, scores=scores, auc_scores=auc_scores, cms=cms,)
