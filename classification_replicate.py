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
# import scipy as scipy
# import scipy.io as sio
import seaborn as sns
cmap = sns.color_palette("crest", as_cmap=True)
# import fnmatch
# import pickle
# from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split  #PredefinedSplit, cross_validate, cross_val_predict, GridSearchCV, LeaveOneGroupOut
from sklearn.feature_selection import SelectFpr, f_classif  #VarianceThreshold, f_classif, SelectKBest, 
# from sklearn.preprocessing import StandardScaler
# from nilearn.input_data import NiftiMasker,  MultiNiftiMasker
# from scipy import stats
# from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# global consts
subIDs = ['002', '003', '004']
tasks = ['rest', 'preremoval', 'study', 'postremoval']
runs = np.arange(6) + 1  
spaces = {'T1w': 'T1w', 
            'MNI': 'MNI152NLin2009cAsym'}
descs = ['brain_mask', 'preproc_bold']
ROIs = ['VVS', 'PHG', 'FG']
shift_sizes_TR = [5, 6]

stim_labels = {0: "Rest",
        1: "Scenes",
        2: "Faces"}

workspace = 'scratch'
if workspace == 'work':
    data_dir = '/work/07365/sguo19/frontera/fmriprep/'
    event_dir = '/work/07365/sguo19/frontera/events/'
    results_dir = '/work/07365/sguo19/model_fitting_results/'
elif workspace == 'scratch':
    data_dir = '/scratch1/07365/sguo19/fmriprep/'
    event_dir = '/scratch1/07365/sguo19/events/'
    results_dir = '/scratch1/07365/sguo19/model_fitting_results/'




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

    def load_existing_data(): 
        print("\n*** Attempting to load existing data if there is any...")
        preproc_data = {}
        todo_ROIs = []
        
        if type(mask_ROIS) == str and mask_ROIS == 'wholebrain':
            if os.path.exists(os.path.join(bold_dir, out_fname_template.format('wholebrain'))):
                print("Loading saved preprocessed data", out_fname_template, '...')
                preproc_data["wholebrain"] = np.load(os.path.join(bold_dir, out_fname_template.format('wholebrain')))
            else:
                print(f"wholebrain data to be processed.")
                todo_ROIs = "wholebrain"

        elif type(mask_ROIS) == list:
            for ROI in mask_ROIS:
                if ROI == 'VVS': ROI = 'vtc'   # only change when loading saved data
                if os.path.exists(os.path.join(bold_dir, out_fname_template.format(ROI))):
                    print("Loading saved preprocessed data", out_fname_template.format(ROI), '...')
                    preproc_data[ROI] = np.load(os.path.join(bold_dir, out_fname_template.format(ROI)))
                else: 
                    print(f"ROI {ROI} data to be processed.")
                    todo_ROIs.append(ROI)
        else: 
            raise ValueError(f"Man! Incorrect ROI value! (Entered: {mask_ROIS})")

        return preproc_data, todo_ROIs

    def confound_cleaner(confounds):
        COI = ['a_comp_cor_00','framewise_displacement','trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']
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
    fname_template = f"sub-{subID}_task-{task}_run-{{}}_space-{space}_desc-{{}}.nii.gz"
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

        # {ROI: time x vox}
        preproc_data = {ROI: np.hstack(run_data) for ROI, run_data in zip(mask_ROIS, cleaned_bolds)}

    print("processed data shape: ", [d.shape for d in preproc_data.values()])
    print("*** Done with preprocessing!")

    # save for future use
    if save: 
        for ROI, run_data in preproc_data.items():
            out_fname = out_fname_template.format(ROI)
            print(f"Saving to file {out_fname}...")
            np.save(out_fname, run_data)

    full_dict = {**ready_data, **preproc_data}
    # array: all_runs_time x all_ROI_vox 
    full_data = np.vstack(list(full_dict.values()))
    return full_data


def get_labels(task, shift_size_TR, rest_tag=0):
    # load labels, & add hemodynamic shift

    def shift_timing(label_TR, TR_shift_size, tag=0):
        # Shift 1D label vectors by given TRs with paddings of tag
        shifted = np.concatenate([np.zeros(TR_shift_size)+tag, label_TR])
        return shifted[:len(label_TR)]  # trim time points outside of scanning time

    print("\n***** Loading labels...")

    event_path = os.path.join(event_dir, f'{task}_events.csv')
    events_df = pd.read_csv(event_path)
    # categories: 1 Scenes, 2 Faces / 0 is rest
    TR_category = events_df["category"].values
    # stim_on labels: 1 actual stim; 2 rest between stims; 0 actual rest
    TR_stim_on = events_df["stim_present"].values

    # shifted
    sTR_category = shift_timing(TR_category, shift_size_TR, rest_tag)
    sTR_stim_on = shift_timing(TR_stim_on, shift_size_TR, rest_tag)

    return sTR_category, sTR_stim_on


def subsample(full_data, stim_list, stim_on, include_rest=True):
    # stim_list: 1 Scenes, 2 Faces / 0 is rest
    # stim_on labels: 1 actual stim; 2 rest between stims; 0 rest between runs
    print("\n***** Subsampling data points...")

    # ===== get indices of samples to choose
    labels = set(stim_list)
    labels.remove(0)  # rest will be done separately afterwards

    # indices for each category
    stim_inds = {lab: np.where((stim_on == 1) & (stim_list == lab))[0] for lab in labels}
    min_n = min([len(inds) for inds in stim_inds.values()])  # min sample size to get from each category

    # subsample min_n samples from each catefgory
    sampled_inds = {}
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

        # get desired time points: can be changed to be middle/end of resting periods, or just random subsample
        # current choice: get time points in the middle of rest periods for rest samples; if 0.5, round up
        rest_inds = [np.ceil(np.average(interval)).astype(int) for interval in rest_intervals]  

        # subsample to min_n
        chosen_rest_inds = np.random.choice(rest_inds, min_n, replace=False)
        sampled_inds[0] = sorted(chosen_rest_inds)

    # ===== stack samples
    X = []
    Y = []
    for lab, inds in sampled_inds.items():
        X.append(full_data[inds, :])
        Y.append(np.zeros(len(inds)) + lab)

    X = np.vstack(X)
    Y = np.concatenate(Y)
    return X, Y


def fit_model(X, Y, save=False, out_fname=None, v=False):
    if v: print("\n***** Fitting model...")

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1/5, stratify=Y)

    # feature selection
    fpr = SelectFpr(f_classif, alpha=0.01).fit(X_train, y_train)
    X_train_sub = fpr.transform(X_train)
    X_test_sub = fpr.transform(X_test)

    # train
    lr = LogisticRegressionCV(penalty='l2', solver='liblinear', cv=5)
    lr = lr.fit(X_train_sub, y_train)

    # test & save
    score = lr.score(X_test_sub, y_test)
    auc_score = roc_auc_score(y_test, lr.predict_proba(X_test_sub), multi_class='ovr')
    preds = lr.predict(X_test_sub)

    if v: print(f"Classifier score: {score}, AUC score: {auc_score}")

    true_counts = np.asarray([np.sum(y_test == i) for i in stim_labels.keys()])
    print(f"True counts: {true_counts}")
    cm = confusion_matrix(y_test, preds, labels=list(stim_labels.keys())) / true_counts[:,None] * 100

    if save:
        if out_fname is None:
            out_fname = os.path.join(results_dir, "lr_fitting.npz")
            print(f"*** Warning: saving without specified path. Saving to {out_fname}")

        np.savez_compressed(out_fname, model=lr, X_train=X_train_sub, X_test=X_test_sub, y_train=y_train, y_test=y_test, preds=preds, cm=cm)

    return lr, score, auc_score, cm


def permutation_test(X, Y, n_iters=1):
    print("\n***** Running {n_iters} permutation tests...")

    scores = []
    auc_scores = []
    cms = []

    for iteri in range(n_iters):
        permuted_Y = np.random.permutation(Y)
        _, score, auc_score, cm = fit_model(X, permuted_Y)

        scores.append(score)
        auc_scores.append(auc_score)
        cms.append(cm)

    scores = np.asarray(scores)
    auc_scores = np.asarray(auc_scores)
    cms = np.asarray(cms)

    print("Permutation results: \n"
        f"scores: {scores.mean()} +/- {scores.std()}\n"
        f"auc scores: {auc_scores.mean()} +/- {auc_scores.std()}\n"
        f"average confution matrix:\n"
        f"{cms.mean(axis=0)}"
    )

    return scores, auc_scores, cms


def classification():
    subID = '004'
    task = 'preremoval'
    space = 'T1w'
    ROIs = ['VVS']
    # ROIs = 'wholebrain'
    n_iters = 10

    # TODO: add random seed

    shift_size_TR = shift_sizes_TR[0]
    rest_tag = 0

    # get data: all_ROI_vox x all_runs_time
    full_data = get_preprocessed_data(subID, task, space, ROIs)
    print(f"Full_data shape: {full_data.shape}")

    # get labels
    sTR_category, sTR_stim_on = get_labels(task, shift_size_TR, rest_tag)
    print(f"Category label shape: {sTR_category.shape}, stim_on shape: {sTR_stim_on.shape}")

    assert len(full_data) == len(sTR_category) == len(sTR_stim_on), \
        f"Length of data ({len(full_data)}) does not match Length of labels ({len(sTR_category)})!"
    
    scores = []
    auc_scores = []
    cms = []
    for iteri in range(n_iters):
        print(f"Running model fitting iteration {iteri}/{n_iters}...")
        # subsample category data & bold time points based on labels
        X, Y = subsample(full_data, sTR_category, sTR_stim_on)
        print("shape of X & Y:", X.shape, Y.shape)

        assert len(X) == len(Y), f"Length of X ({len(X)}) doesn't match length of Y({len(Y)})"

        # model fitting 
        results_fname = os.path.join(results_dir, f"sub-{subID}_task-{task}_space-{space}_{ROIs[0]}_lr.npz")
        _, score, auc_score, cm = fit_model(X, Y, save=False, v=True)
        
        scores.append(score)
        auc_scores.append(auc_score)
        cms.append(cm)
    scores = np.asarray(scores)
    auc_scores = np.asarray(auc_scores)
    cms = np.asarray(cms)

    print("Model fitting results: \n"
        f"scores: {scores.mean()} +/- {scores.std()}\n"
        f"auc scores: {auc_scores.mean()} +/- {auc_scores.std()}\n"
        f"average confution matrix:\n"
        f"{cms.mean(axis=0)}"
    )
    
    # permutation test
    perm_scores, perm_auc_scores, perm_cms = permutation_test(X, Y, n_iters=n_iters)

    # save results
    np.savez_compressed(results_fname, scores=scores, auc_scores=auc_scores, cms=cms,
                        perm_scores=perm_scores, perm_auc_scores=perm_auc_scores, perm_cms=perm_cms)

def visualization():
    subID = '002'
    task = 'preremoval'
    space = 'T1w'
    ROIs = ['VVS']
    model_fname = os.path.join(results_dir, f"sub-{subID}_task-{task}_space-{space}_{ROIs[0]}_lr.npz")
    out_fname = os.path.join(results_dir, f"sub-{subID}_task-{task}_space-{space}_{ROIs[0]}_lr.png")

    f = np.load(model_fname)
    y_test = f['y_test']
    preds = f['preds']

    # for num, lab in labels.items():
    #     y_test[y_test == num] = lab
    #     preds[preds == num] = lab

    true_counts = np.asarray([np.sum(y_test == i) for i in stim_labels.keys()])
    cm = confusion_matrix(y_test, preds, labels=list(stim_labels.keys()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm/true_counts[:,None]*100,
                                  display_labels=list(stim_labels.values()))
    disp.plot(cmap=cmap)
    plt.savefig(out_fname)

    



if __name__ == '__main__':
    # add arguments
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-s", type=str, help="subject ID in 3 digits")
    # parser.add_argument("-t", type=str, help="task", choice=tasks)
    # parser.add_argument("-b", type=str, help="brain space", choice=spaces.keys())
    

    classification()
    # visualization()


