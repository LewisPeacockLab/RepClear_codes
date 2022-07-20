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
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests


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

def load_existing_data(subID, task, space, mask_ROIS,load=False): #try to find if the data we want has already been cleaned and saved...
    print("\n*** Attempting to load existing data if there is any...")
    preproc_data = {}
    todo_ROIs = []

    bold_dir = os.path.join(data_dir, f"sub-{subID}", "func")
    out_fname_template = f"sub-{subID}_{space}_{task}_{{}}_masked_cleaned.npy"
    for ROI in mask_ROIS:
        if ROI=='VVS': ROI='VTC'
        if load:
            if os.path.exists(os.path.join(bold_dir, out_fname_template.format(ROI))):
                print("\nLoading saved preprocessed data", out_fname_template.format(ROI), '...')
                preproc_data[ROI] = np.load(os.path.join(bold_dir, out_fname_template.format(ROI)))
            else: 
                if ROI == 'VTC': ROI = 'VVS'  # change to load masks...
                print(f"\nROI {ROI} data to be processed.")
                todo_ROIs.append(ROI)
        else:
            if ROI == 'VTC': ROI = 'VVS'  # change to load masks...
            print(f"\nROI {ROI} data to be processed.")
            todo_ROIs.append(ROI)
    return preproc_data, todo_ROIs

def load_process_data(subID, task, space, mask_ROIS): #this wraps the above function, with a process to load and save the data if there isnt an already saved .npy file
    # ========== check & load existing files
    ready_data, mask_ROIS = load_existing_data(subID, task, space, mask_ROIS)
    if type(mask_ROIS) == list and len(mask_ROIS) == 0: 
        return np.vstack(list(ready_data.values()))
    else: 
        print("\nPreprocessing ROIs", mask_ROIS)


    print(f"\n***** Data preprocessing for sub {subID} {task} {space} with ROIs {mask_ROIS}...")

    space_long = spaces[space]

    if task=='study':
        runs=np.arange(3) + 1  
    else:
        runs=np.arange(6) + 1

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
            run_pair=[runi,runj]
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
            yield X, Y, all_groups, run_pair

            # flip groups so even & odd groups can be paired
            all_groups = np.concatenate([groups, list(reversed(groups)), list(reversed(groups))])
            yield X, Y, all_groups, run_pair

def subsample_for_training(full_data, label_df, train_pairs, include_rest=True):
    """
    Subsample data by runs. 
    Return: subsampled labels and bold data
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

    scenes_runs = train_pairs
    runi = scenes_runs[0]
    runj = scenes_runs[1]
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
    return X, Y

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

def decode(training_runs, train_data, train_labels, test_data, test_labels):
    print(f"Running model fitting...")

    X_train, y_train = subsample_for_training(train_data, train_labels, training_runs, include_rest=True)

    X_test, y_test = test_data, test_labels['category'].values
    
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
    best_C=(gscv.best_params_['C'])
    
    # refit with full data and optimal penalty value
    lr = LogisticRegression(penalty='l2', solver='lbfgs', C=best_C,max_iter=1000)
    lr.fit(X_train_sub, y_train)
    
    # test on held out data
    predictions=lr.predict(X_test_sub) #get the predicted categories from running the model
    evidence=(1. / (1. + np.exp(-lr.decision_function(X_test_sub)))) #MAIN: collect the evidence values of all timepoints (3 columns: 0-rest, 1-faces, 2-scenes)
    true=y_test

    return predictions, evidence, true


def classification(subID):
    task = 'preremoval'
    task2 = 'study'
    space = 'T1w'
    ROIs = ['VVS']
    n_iters = 1


    shift_size_TR = shift_sizes_TR[0]
    rest_tag = 0

    print(f"\n***** Running category level classification for sub {subID} {task} {space} with ROIs {ROIs}...")

    # get data:
    full_data = load_process_data(subID, task, space, ROIs)
    print(f"Full_data shape: {full_data.shape}")

    # get labels:
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
    run_pairs=[]
    
    for X, Y, groups, run_pair in subsample_by_runs(full_data, label_df): 
        print(f"Running model fitting...")
        print("shape of X & Y:", X.shape, Y.shape)
        assert len(X) == len(Y), f"Length of X ({len(X)}) doesn't match length of Y({len(Y)})"

        # model fitting 
        score, auc_score, cm, trained_model, fpr = fit_model(X, Y, groups, save=False, v=True)
        
        scores.append(score)
        auc_scores.append(auc_score)
        cms.append(cm)
        trained_models.append(trained_model)
        fprs.append(fpr)
        run_pairs.append(run_pair)

    scores = np.mean(scores,axis=1) #get the mean of each iteration (since theyre in pairs for the train/test pair)
    auc_scores = np.mean(auc_scores,axis=1) #get the auc mean of each iteration (since theyre in pairs for the train/test pair)
    cms = np.stack(cms)

    #need to find the highest score/auc_score to maximize decoding. So we want to select the best combo of train/test (since I get scores for each set)
    best_model=np.where(auc_scores==max(auc_scores))[0][0] #finding the max via auc_score
    train_pairs=run_pairs[best_model]

    print(f"\n***** Running category level classification for sub {subID} {task2} {space} with ROIs {ROIs}...")

    # get data: all_ROI_vox x all_runs_time
    full_data2 = load_process_data(subID, task2, space, ROIs)
    print(f"Full_data shape: {full_data2.shape}")

    # get labels
    label_df2 = get_shifted_labels(task2, shift_size_TR, rest_tag)
    print(f"Category label shape: {label_df2.shape}") 

    #will need to build in a real selection method, but for now can take the first versions
    predicts, evidence, true = decode(train_pairs, full_data, label_df, full_data2, label_df2) #this will decode the data using the already trained model and feature selection method

    #need to then save the evidence:
    evidence_df=pd.DataFrame(data=label_df2) #take the study DF for this subject
    evidence_df.drop(columns=evidence_df.columns[0], axis=1, inplace=True) #now drop the extra index column
    evidence_df['rest_evi']=evidence[:,0] #add in the evidence values for rest
    evidence_df['scene_evi']=evidence[:,1] #add in the evidence values for scenes
    evidence_df['face_evi']=evidence[:,2] #add in the evidence values for faces

    #this will export the subject level evidence to the subject's folder
    sub_dir = os.path.join(data_dir, f"sub-{subID}")
    out_fname_template = f"sub-{subID}_{space}_trained-{task}_tested-{task2}_evidence.csv"            
    print("\n *** Saving evidence values with subject dataframe ***")
    evidence_df.to_csv(os.path.join(sub_dir,out_fname_template))


#now I need to create a function to take each subject's evidence DF, and then sort, organize and then visualize:
def organize_evidence(subID,save=True):
    task = 'preremoval'
    task2 = 'study'
    space = 'T1w'
    ROIs = ['VVS']

    print("\n *** loading evidence values from subject dataframe ***")

    sub_dir = os.path.join(data_dir, f"sub-{subID}")
    in_fname_template = f"sub-{subID}_{space}_trained-{task}_tested-{task2}_evidence.csv"   

    sub_df=pd.read_csv(os.path.join(sub_dir,in_fname_template))  
    sub_df.drop(columns=sub_df.columns[0], axis=1, inplace=True) #now drop the extra index column

    sub_images,sub_index=np.unique(sub_df['image_id'], return_index=True) #this searches through the dataframe to find each occurance of the image_id. This allows me to find the start of each trial, and linked to the image_ID #

    #now to sort the trials, we need to figure out what the operation performed is:
    sub_condition_list=sub_df['condition'][sub_index].values.astype(int) #so using the above indices, we will now grab what the condition is of each image

    counter=0
    maintain_trials={}
    replace_trials={}
    suppress_trials={}

    for i in sub_condition_list:
        if i==0: #this is the first rest period (because we had to shift of hemodynamics. So this "0" condition is nothing)
            print('i==0')
            counter+=1                         
            continue
        elif i==1:
            temp_image=sub_images[counter]
            maintain_trials[temp_image]=sub_df[['rest_evi','scene_evi','face_evi']][sub_index[counter]-5:sub_index[counter]+9].values
            counter+=1

        elif i==2:
            temp_image=sub_images[counter]            
            replace_trials[temp_image]=sub_df[['rest_evi','scene_evi','face_evi']][sub_index[counter]-5:sub_index[counter]+9].values
            counter+=1

        elif i==3:
            temp_image=sub_images[counter]            
            suppress_trials[temp_image]=sub_df[['rest_evi','scene_evi','face_evi']][sub_index[counter]-5:sub_index[counter]+9].values
            counter+=1

    #now that the trials are sorted, we need to get the subject average for each condition:
    avg_maintain=pd.DataFrame(data=np.dstack(maintain_trials.values()).mean(axis=2))
    avg_replace=pd.DataFrame(data=np.dstack(replace_trials.values()).mean(axis=2))
    avg_suppress=pd.DataFrame(data=np.dstack(suppress_trials.values()).mean(axis=2))

    #now I will have to change the structure to be able to plot in seaborn:
    avg_maintain=avg_maintain.T.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_maintain['sub']=np.repeat(subID,len(avg_maintain)) #input the subject so I can stack melted dfs
    avg_maintain['evidence_class']=np.tile(['rest','scenes','faces'],14) #add in the labels so we know what each data point is refering to
    avg_maintain.rename(columns={'variable':'TR','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_maintain['condition']='maintain' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_replace=avg_replace.T.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_replace['sub']=np.repeat(subID,len(avg_replace)) #input the subject so I can stack melted dfs
    avg_replace['evidence_class']=np.tile(['rest','scenes','faces'],14) #add in the labels so we know what each data point is refering to
    avg_replace.rename(columns={'variable':'TR','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_replace['condition']='replace' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_suppress=avg_suppress.T.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_suppress['sub']=np.repeat(subID,len(avg_suppress)) #input the subject so I can stack melted dfs
    avg_suppress['evidence_class']=np.tile(['rest','scenes','faces'],14) #add in the labels so we know what each data point is refering to
    avg_suppress.rename(columns={'variable':'TR','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_suppress['condition']='suppress' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_subject_df= pd.concat([avg_maintain,avg_replace,avg_suppress], ignore_index=True, sort=False)

    # save for future use
    if save: 
        sub_dir = os.path.join(data_dir, f"sub-{subID}")
        out_fname_template = f"sub-{subID}_{space}_{task2}_evidence_dataframe.csv"  
        print(f"\n Saving the sorted evidence dataframe for {subID} - phase: {task2} - as {out_fname_template}")
        avg_subject_df.to_csv(os.path.join(sub_dir,out_fname_template))
    return avg_subject_df

def organize_memory_evidence(subID,save=True):
    task = 'preremoval'
    task2 = 'study'
    space = 'T1w'
    ROIs = ['VVS']

    print("\n *** loading evidence values from subject dataframe ***")

    sub_dir = os.path.join(data_dir, f"sub-{subID}")
    in_fname_template = f"sub-{subID}_{space}_trained-{task}_tested-{task2}_evidence.csv"   

    sub_df=pd.read_csv(os.path.join(sub_dir,in_fname_template))  
    sub_df.drop(columns=sub_df.columns[0], axis=1, inplace=True) #now drop the extra index column

    sub_images,sub_index=np.unique(sub_df['image_id'], return_index=True) #this searches through the dataframe to find each occurance of the image_id. This allows me to find the start of each trial, and linked to the image_ID #

    #now to sort the trials, we need to figure out what the operation performed is:
    sub_condition_list=sub_df['condition'][sub_index].values.astype(int) #so using the above indices, we will now grab what the condition is of each image

    subject_design_dir='/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/subject_designs/'

    memory_file_path=os.path.join(subject_design_dir,f"memory_and_familiar_sub-{subID}.csv")
    memory_df=pd.read_csv(memory_file_path)

    for i in np.unique(sub_df['image_id'].values):
        if i==0:
            continue
        else:
            response=memory_df[memory_df['image_num']==i].resp.values[0]

            if response==4:
                sub_df.loc[sub_df['image_id']==i,'memory']=1
            else:
                sub_df.loc[sub_df['image_id']==i,'memory']=0


    counter=0
    maintain_remember_trials={}
    replace_remember_trials={}
    suppress_remember_trials={}

    maintain_forgot_trials={}
    replace_forgot_trials={}
    suppress_forgot_trials={}

    for i in sub_condition_list:
        if i==0: #this is the first rest period (because we had to shift of hemodynamics. So this "0" condition is nothing)
            print('i==0')
            counter+=1                         
            continue
        elif i==1:
            temp_image=sub_images[counter]
            temp_memory=sub_df[sub_df['image_id']==temp_image]['memory'].values[0] #this grabs the first index of the images, and checks the memory outcome

            if temp_memory==1:
                maintain_remember_trials[temp_image]=sub_df[['rest_evi','scene_evi','face_evi']][sub_index[counter]-5:sub_index[counter]+9].values
            elif temp_memory==0:
                maintain_forgot_trials[temp_image]=sub_df[['rest_evi','scene_evi','face_evi']][sub_index[counter]-5:sub_index[counter]+9].values                
            counter+=1

        elif i==2:
            temp_image=sub_images[counter]   
            temp_memory=sub_df[sub_df['image_id']==temp_image]['memory'].values[0] #this grabs the first index of the images, and checks the memory outcome

            if temp_memory==1:
                replace_remember_trials[temp_image]=sub_df[['rest_evi','scene_evi','face_evi']][sub_index[counter]-5:sub_index[counter]+9].values
            elif temp_memory==0:
                replace_forgot_trials[temp_image]=sub_df[['rest_evi','scene_evi','face_evi']][sub_index[counter]-5:sub_index[counter]+9].values
            counter+=1

        elif i==3:
            temp_image=sub_images[counter]
            temp_memory=sub_df[sub_df['image_id']==temp_image]['memory'].values[0] #this grabs the first index of the images, and checks the memory outcome

            if temp_memory==1:
                suppress_remember_trials[temp_image]=sub_df[['rest_evi','scene_evi','face_evi']][sub_index[counter]-5:sub_index[counter]+9].values
            elif temp_memory==0:
                suppress_forgot_trials[temp_image]=sub_df[['rest_evi','scene_evi','face_evi']][sub_index[counter]-5:sub_index[counter]+9].values                            
            counter+=1

    #now that the trials are sorted, we need to get the subject average for each condition and memory:
    avg_remember_maintain=pd.DataFrame(data=np.dstack(maintain_remember_trials.values()).mean(axis=2))
    avg_remember_replace=pd.DataFrame(data=np.dstack(replace_remember_trials.values()).mean(axis=2))
    avg_remember_suppress=pd.DataFrame(data=np.dstack(suppress_remember_trials.values()).mean(axis=2))

    if maintain_forgot_trials:
        avg_forgot_maintain=pd.DataFrame(data=np.dstack(maintain_forgot_trials.values()).mean(axis=2))
    else:
        avg_forgot_maintain=pd.DataFrame()

    if replace_forgot_trials:
        avg_forgot_replace=pd.DataFrame(data=np.dstack(replace_forgot_trials.values()).mean(axis=2))
    else:
        avg_forgot_replace=pd.DataFrame()    

    if suppress_forgot_trials:
        avg_forgot_suppress=pd.DataFrame(data=np.dstack(suppress_forgot_trials.values()).mean(axis=2))
    else: 
        avg_forgot_suppress=pd.DataFrame()

    #now I will have to change the structure to be able to plot in seaborn:
    avg_remember_maintain=avg_remember_maintain.T.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_remember_maintain['sub']=np.repeat(subID,len(avg_remember_maintain)) #input the subject so I can stack melted dfs
    avg_remember_maintain['evidence_class']=np.tile(['rest','scenes','faces'],14) #add in the labels so we know what each data point is refering to
    avg_remember_maintain.rename(columns={'variable':'TR','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_remember_maintain['condition']='maintain' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_forgot_maintain=avg_forgot_maintain.T.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_forgot_maintain['sub']=np.repeat(subID,len(avg_forgot_maintain)) #input the subject so I can stack melted dfs
    avg_forgot_maintain['evidence_class']=np.tile(['rest','scenes','faces'],14) #add in the labels so we know what each data point is refering to
    avg_forgot_maintain.rename(columns={'variable':'TR','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_forgot_maintain['condition']='maintain' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    ####

    avg_remember_replace=avg_remember_replace.T.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_remember_replace['sub']=np.repeat(subID,len(avg_remember_replace)) #input the subject so I can stack melted dfs
    avg_remember_replace['evidence_class']=np.tile(['rest','scenes','faces'],14) #add in the labels so we know what each data point is refering to
    avg_remember_replace.rename(columns={'variable':'TR','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_remember_replace['condition']='replace' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_forgot_replace=avg_forgot_replace.T.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_forgot_replace['sub']=np.repeat(subID,len(avg_forgot_replace)) #input the subject so I can stack melted dfs
    avg_forgot_replace['evidence_class']=np.tile(['rest','scenes','faces'],14) #add in the labels so we know what each data point is refering to
    avg_forgot_replace.rename(columns={'variable':'TR','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_forgot_replace['condition']='replace' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    ####

    avg_remember_suppress=avg_remember_suppress.T.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_remember_suppress['sub']=np.repeat(subID,len(avg_remember_suppress)) #input the subject so I can stack melted dfs
    avg_remember_suppress['evidence_class']=np.tile(['rest','scenes','faces'],14) #add in the labels so we know what each data point is refering to
    avg_remember_suppress.rename(columns={'variable':'TR','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_remember_suppress['condition']='suppress' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_forgot_suppress=avg_forgot_suppress.T.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_forgot_suppress['sub']=np.repeat(subID,len(avg_forgot_suppress)) #input the subject so I can stack melted dfs
    avg_forgot_suppress['evidence_class']=np.tile(['rest','scenes','faces'],14) #add in the labels so we know what each data point is refering to
    avg_forgot_suppress.rename(columns={'variable':'TR','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_forgot_suppress['condition']='suppress' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_remember_subject_df= pd.concat([avg_remember_maintain,avg_remember_replace,avg_remember_suppress], ignore_index=True, sort=False)

    avg_forgot_subject_df= pd.concat([avg_forgot_maintain,avg_forgot_replace,avg_forgot_suppress], ignore_index=True, sort=False)


    # save for future use
    if save: 
        sub_dir = os.path.join(data_dir, f"sub-{subID}")
        out_fname_template = f"sub-{subID}_{space}_{task2}_remember_evidence_dataframe.csv"  
        print(f"\n Saving the sorted remebered evidence dataframe for {subID} - phase: {task2} - as {out_fname_template}")
        avg_remember_subject_df.to_csv(os.path.join(sub_dir,out_fname_template))

        out_fname_template2 = f"sub-{subID}_{space}_{task2}_forgot_evidence_dataframe.csv"  
        print(f"\n Saving the sorted forgot evidence dataframe for {subID} - phase: {task2} - as {out_fname_template}")
        avg_forgot_subject_df.to_csv(os.path.join(sub_dir,out_fname_template2))        

    return avg_remember_subject_df, avg_forgot_subject_df


def visualize_evidence():
    group_evidence_df=pd.DataFrame()
    for subID in subIDs:
        temp_subject_df=organize_evidence(subID)
        group_evidence_df=pd.concat([group_evidence_df,temp_subject_df],ignore_index=True, sort=False)

    ax=sns.lineplot(data=group_evidence_df.loc[(group_evidence_df['condition']=='maintain') & (group_evidence_df['evidence_class']=='scenes')], x='TR',y='evidence',color='green',label='maintain', ci=68)
    ax=sns.lineplot(data=group_evidence_df.loc[(group_evidence_df['condition']=='replace') & (group_evidence_df['evidence_class']=='scenes')], x='TR',y='evidence',color='blue',label='replace-old', ci=68)
    ax=sns.lineplot(data=group_evidence_df.loc[(group_evidence_df['condition']=='replace') & (group_evidence_df['evidence_class']=='faces')], x='TR',y='evidence',color='skyblue',label='replace-new',ci=68)
    ax=sns.lineplot(data=group_evidence_df.loc[(group_evidence_df['condition']=='suppress') & (group_evidence_df['evidence_class']=='scenes')], x='TR',y='evidence',color='red',label='suppress',ci=68)

    plt.legend()
    ax.set(xlabel='TR', ylabel='Category Classifier Evidence', title='T1w - Category Classifier (group average)')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs','group_level_category_decoding_during_removal.png'))
    plt.clf()

    #now I want to sort the data based on the outcome of that item:
    group_remember_evidence_df=pd.DataFrame()
    group_forgot_evidence_df=pd.DataFrame()

    for subID in subIDs:
        temp_remember_subject_df, temp_forgot_subject_df=organize_memory_evidence(subID)

        group_remember_evidence_df=pd.concat([group_remember_evidence_df,temp_remember_subject_df],ignore_index=True, sort=False)    
        group_forgot_evidence_df=pd.concat([group_forgot_evidence_df,temp_forgot_subject_df],ignore_index=True, sort=False)    

    ax=sns.lineplot(data=group_remember_evidence_df.loc[(group_remember_evidence_df['condition']=='maintain') & (group_remember_evidence_df['evidence_class']=='scenes')], x='TR',y='evidence',color='green',label='maintain', ci=68)
    ax=sns.lineplot(data=group_remember_evidence_df.loc[(group_remember_evidence_df['condition']=='replace') & (group_remember_evidence_df['evidence_class']=='scenes')], x='TR',y='evidence',color='blue',label='replace-old', ci=68)
    ax=sns.lineplot(data=group_remember_evidence_df.loc[(group_remember_evidence_df['condition']=='replace') & (group_remember_evidence_df['evidence_class']=='faces')], x='TR',y='evidence',color='skyblue',label='replace-new',ci=68)
    ax=sns.lineplot(data=group_remember_evidence_df.loc[(group_remember_evidence_df['condition']=='suppress') & (group_remember_evidence_df['evidence_class']=='scenes')], x='TR',y='evidence',color='red',label='suppress',ci=68)

    plt.legend()
    ax.set(xlabel='TR', ylabel='Category Classifier Evidence', title='T1w - Category Classifier (group average) - Remembered Items')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs','group_level_category_decoding_during_removal_remember.png'))
    plt.clf()

    ax=sns.lineplot(data=group_forgot_evidence_df.loc[(group_forgot_evidence_df['condition']=='maintain') & (group_forgot_evidence_df['evidence_class']=='scenes')], x='TR',y='evidence',color='green',label='maintain', ci=68)
    ax=sns.lineplot(data=group_forgot_evidence_df.loc[(group_forgot_evidence_df['condition']=='replace') & (group_forgot_evidence_df['evidence_class']=='scenes')], x='TR',y='evidence',color='blue',label='replace-old', ci=68)
    ax=sns.lineplot(data=group_forgot_evidence_df.loc[(group_forgot_evidence_df['condition']=='replace') & (group_forgot_evidence_df['evidence_class']=='faces')], x='TR',y='evidence',color='skyblue',label='replace-new',ci=68)
    ax=sns.lineplot(data=group_forgot_evidence_df.loc[(group_forgot_evidence_df['condition']=='suppress') & (group_forgot_evidence_df['evidence_class']=='scenes')], x='TR',y='evidence',color='red',label='suppress',ci=68)

    plt.legend()
    ax.set(xlabel='TR', ylabel='Category Classifier Evidence', title='T1w - Category Classifier (group average) - Forgot Items')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs','group_level_category_decoding_during_removal_forgot.png'))
    plt.clf()    

    #we have now plotted the evidence of the remembered items and the forgotten items separately, now I wanna plot the difference between remembered and forgotten
    group_diff_evidence_df=group_remember_evidence_df.copy(deep=True)
    group_diff_evidence_df['evidence']=group_remember_evidence_df['evidence']-group_forgot_evidence_df['evidence']

    ax=sns.lineplot(data=group_diff_evidence_df.loc[(group_diff_evidence_df['condition']=='maintain') & (group_diff_evidence_df['evidence_class']=='scenes')], x='TR',y='evidence',color='green',label='maintain', ci=68)
    ax=sns.lineplot(data=group_diff_evidence_df.loc[(group_diff_evidence_df['condition']=='replace') & (group_diff_evidence_df['evidence_class']=='scenes')], x='TR',y='evidence',color='blue',label='replace-old', ci=68)
    ax=sns.lineplot(data=group_diff_evidence_df.loc[(group_diff_evidence_df['condition']=='replace') & (group_diff_evidence_df['evidence_class']=='faces')], x='TR',y='evidence',color='skyblue',label='replace-new',ci=68)
    ax=sns.lineplot(data=group_diff_evidence_df.loc[(group_diff_evidence_df['condition']=='suppress') & (group_diff_evidence_df['evidence_class']=='scenes')], x='TR',y='evidence',color='red',label='suppress',ci=68)
    ax.axhline(0,color='k',linestyle='--')

    plt.legend()
    ax.set(xlabel='TR', ylabel='Category Classifier Evidence (Remember - Forgot)', title='T1w - Category Classifier (group average): Remember-Forgot')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs','group_level_category_decoding_during_removal_difference.png'))
    plt.clf() 

    #plot the traces of remembered and forgotten items during suppress and replace conditions:
    group_remember_evidence_df['memory']='remembered'
    group_forgot_evidence_df['memory']='forgotten'
    group_combined_evidence_df=pd.concat([group_remember_evidence_df,group_forgot_evidence_df],ignore_index=True,sort=False)

    ax=sns.lineplot(data=group_combined_evidence_df.loc[(group_combined_evidence_df['evidence_class']=='scenes')], x='TR',y='evidence',hue='condition',style='memory',palette=['green','blue','red'],ci=68)

    plt.legend()
    ax.set(xlabel='TR', ylabel='Category Classifier Evidence', title='T1w - Category Classifier (group average): Remember & Forgot')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs','group_level_category_decoding_during_removal_RandF.png'))
    plt.clf() 

    #plot the difference between remembered and forgotten, but for each condition separately:
    ax=sns.lineplot(data=group_diff_evidence_df.loc[(group_diff_evidence_df['condition']=='maintain') & (group_diff_evidence_df['evidence_class']=='scenes')], x='TR',y='evidence',color='green',label='maintain', ci=68)
    ax.axhline(0,color='k',linestyle='--')
    plt.legend()
    ax.set(xlabel='TR', ylabel='Category Classifier Evidence (Remember - Forgot)')
    ax.set_title('T1w - Category Classifier during Maintain (group average): Remember-Forgot', loc='center', wrap=True)    
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs','group_level_category_decoding_during_maintain_difference.png'))
    plt.clf() 

    ax=sns.lineplot(data=group_diff_evidence_df.loc[(group_diff_evidence_df['condition']=='replace') & (group_diff_evidence_df['evidence_class']=='scenes')], x='TR',y='evidence',color='blue',label='replace-old', ci=68)
    ax=sns.lineplot(data=group_diff_evidence_df.loc[(group_diff_evidence_df['condition']=='replace') & (group_diff_evidence_df['evidence_class']=='faces')], x='TR',y='evidence',color='skyblue',label='replace-new',ci=68)    
    ax.axhline(0,color='k',linestyle='--')
    plt.legend()
    ax.set(xlabel='TR', ylabel='Category Classifier Evidence (Remember - Forgot)')
    ax.set_title('T1w - Category Classifier during Replace (group average): Remember-Forgot', loc='center', wrap=True)    
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs','group_level_category_decoding_during_replace_difference.png'))
    plt.clf()

    ax=sns.lineplot(data=group_diff_evidence_df.loc[(group_diff_evidence_df['condition']=='suppress') & (group_diff_evidence_df['evidence_class']=='scenes')], x='TR',y='evidence',color='red',label='suppress',ci=68)
    ax.axhline(0,color='k',linestyle='--')
    plt.legend()
    ax.set(xlabel='TR', ylabel='Category Classifier Evidence (Remember - Forgot)')
    ax.set_title('T1w - Category Classifier during Suppress (group average): Remember-Forgot', loc='center', wrap=True)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs','group_level_category_decoding_during_suppress_difference.png'))
    plt.clf()

    #want to get the trajectory for removal of an item in WM (classifier evidence of removal - maintain):
    maintain_df=group_evidence_df.loc[(group_evidence_df['condition']=='maintain') & (group_evidence_df['evidence_class']=='scenes')]
    replace_df=group_evidence_df.loc[(group_evidence_df['condition']=='replace') & (group_evidence_df['evidence_class']=='scenes')]
    suppress_df=group_evidence_df.loc[(group_evidence_df['condition']=='suppress') & (group_evidence_df['evidence_class']=='scenes')]

    replace_df['evidence']=replace_df['evidence'].values-maintain_df['evidence'].values
    suppress_df['evidence']=suppress_df['evidence'].values-maintain_df['evidence'].values

    ax=sns.lineplot(data=replace_df, x='TR',y='evidence',color='blue',label='replace',ci=68)
    ax=sns.lineplot(data=suppress_df, x='TR', y='evidence', color='red',label='suppress',ci=68)
    ax.axhline(0,color='k',linestyle='--')
    plt.legend()
    ax.set(xlabel='TR', ylabel='Category Classifier Evidence (Removal - Maintain)')
    ax.set_title('T1w - Category Classifier - Trajectory for removal of an item from WM', loc='center', wrap=True)    
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs','group_level_category_decoding_removal_minus_maintain.png'))
    plt.clf()    

    #want to get the trajectory for removal of an item in WM (classifier evidence of removal - maintain) but focused on FORGOTTEN ITEMS:
    maintain_df=group_evidence_df.loc[(group_evidence_df['condition']=='maintain') & (group_evidence_df['evidence_class']=='scenes')]
    forgot_replace_df=group_forgot_evidence_df.loc[(group_forgot_evidence_df['condition']=='replace') & (group_forgot_evidence_df['evidence_class']=='scenes')]
    forgot_suppress_df=group_forgot_evidence_df.loc[(group_forgot_evidence_df['condition']=='suppress') & (group_forgot_evidence_df['evidence_class']=='scenes')]

    forgot_replace_df['evidence']=forgot_replace_df['evidence'].values-maintain_df['evidence'].values
    forgot_suppress_df['evidence']=forgot_suppress_df['evidence'].values-maintain_df['evidence'].values

    ax=sns.lineplot(data=forgot_replace_df, x='TR',y='evidence',color='blue',label='replace',ci=68)
    ax=sns.lineplot(data=forgot_suppress_df, x='TR', y='evidence', color='red',label='suppress',ci=68)
    ax.axhline(0,color='k',linestyle='--')
    plt.legend()
    ax.set(xlabel='TR', ylabel='Category Classifier Evidence (Removal - Maintain)')
    ax.set_title('T1w - Category Classifier (Forgotten Items) - Trajectory for removal of an item from WM', loc='center', wrap=True)    
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs','group_level_category_decoding_removal_minus_maintain_forgot.png'))
    plt.clf()

    #want to get the trajectory for removal of an item in WM (classifier evidence of removal - maintain) but focused on REMEMBERED ITEMS:
    maintain_df=group_evidence_df.loc[(group_evidence_df['condition']=='maintain') & (group_evidence_df['evidence_class']=='scenes')]
    remember_replace_df=group_remember_evidence_df.loc[(group_remember_evidence_df['condition']=='replace') & (group_remember_evidence_df['evidence_class']=='scenes')]
    remember_suppress_df=group_remember_evidence_df.loc[(group_remember_evidence_df['condition']=='suppress') & (group_remember_evidence_df['evidence_class']=='scenes')]

    remember_replace_df['evidence']=remember_replace_df['evidence'].values-maintain_df['evidence'].values
    remember_suppress_df['evidence']=remember_suppress_df['evidence'].values-maintain_df['evidence'].values

    ax=sns.lineplot(data=remember_replace_df, x='TR',y='evidence',color='blue',label='replace',ci=68)
    ax=sns.lineplot(data=remember_suppress_df, x='TR', y='evidence', color='red',label='suppress',ci=68)
    ax.axhline(0,color='k',linestyle='--')
    plt.legend()
    ax.set(xlabel='TR', ylabel='Category Classifier Evidence (Removal - Maintain)')
    ax.set_title('T1w - Category Classifier (Forgotten Items) - Trajectory for removal of an item from WM', loc='center', wrap=True)    
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs','group_level_category_decoding_removal_minus_maintain_remember.png'))
    plt.clf()        

def check_anova_stats():
    group_evidence_df=pd.DataFrame()
    for subID in subIDs:
        temp_subject_df=organize_evidence(subID)
        group_evidence_df=pd.concat([group_evidence_df,temp_subject_df],ignore_index=True, sort=False)

    scene_evis=group_evidence_df.loc[group_evidence_df['evidence_class']=='scenes']   
    
    ANOVA_results={}
    ttest_sm={}
    ttest_rm={}
    ttest_sr={}    
    for i in scene_evis.TR.unique():
        temp_df=scene_evis.loc[scene_evis.TR==i]
        ANOVA_results[f"TR {i}"]=AnovaRM(data=temp_df, depvar='evidence', subject='sub', within=['condition']).fit()
        print(f"TR {i}: ")
        print(ANOVA_results[f"TR {i}"].summary())

        temp_suppress=temp_df.loc[temp_df.condition=='suppress']['evidence'].values
        temp_maintain=temp_df.loc[temp_df.condition=='maintain']['evidence'].values
        temp_replace=temp_df.loc[temp_df.condition=='replace']['evidence'].values

        _, ttest_sm[f"TR {i}"]=stats.ttest_rel(temp_suppress,temp_maintain)
        _, ttest_rm[f"TR {i}"]=stats.ttest_rel(temp_replace,temp_maintain)
        _, ttest_sr[f"TR {i}"]=stats.ttest_rel(temp_suppress,temp_replace)


    ttest_sm_values=list(ttest_sm.values())
    ttest_rm_values=list(ttest_rm.values())
    ttest_sr_values=list(ttest_sr.values())

    ttest_sm_corrected=multipletests(ttest_sm_values,alpha=0.05,method='fdr_tsbky')
    ttest_rm_corrected=multipletests(ttest_rm_values,alpha=0.05,method='fdr_tsbky')
    ttest_sr_corrected=multipletests(ttest_sr_values,alpha=0.05,method='fdr_tsbky')    


    return ANOVA_results, ttest_sm, ttest_rm, ttest_sr

def classification_post(subID):
    #decode the category information during the post-removal phase, looking at the encoding period of each item
    task = 'preremoval'
    task2 = 'postremoval'
    space = 'T1w'
    ROIs = ['VVS']
    n_iters = 1


    shift_size_TR = 5 
    rest_tag = 0

    print(f"\n***** Running category level classification for sub {subID} {task} {space} with ROIs {ROIs}...")

    # get data:
    full_data = load_process_data(subID, task, space, ROIs)
    print(f"Full_data shape: {full_data.shape}")

    # get labels:
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
    run_pairs=[]
    
    for X, Y, groups, run_pair in subsample_by_runs(full_data, label_df): 
        print(f"Running model fitting...")
        print("shape of X & Y:", X.shape, Y.shape)
        assert len(X) == len(Y), f"Length of X ({len(X)}) doesn't match length of Y({len(Y)})"

        # model fitting 
        score, auc_score, cm, trained_model, fpr = fit_model(X, Y, groups, save=False, v=True)
        
        scores.append(score)
        auc_scores.append(auc_score)
        cms.append(cm)
        trained_models.append(trained_model)
        fprs.append(fpr)
        run_pairs.append(run_pair)

    scores = np.mean(scores,axis=1) #get the mean of each iteration (since theyre in pairs for the train/test pair)
    auc_scores = np.mean(auc_scores,axis=1) #get the auc mean of each iteration (since theyre in pairs for the train/test pair)
    cms = np.stack(cms)

    #need to find the highest score/auc_score to maximize decoding. So we want to select the best combo of train/test (since I get scores for each set)
    best_model=np.where(auc_scores==max(auc_scores))[0][0] #finding the max via auc_score
    train_pairs=run_pairs[best_model]

    print(f"\n***** Running category level classification for sub {subID} {task2} {space} with ROIs {ROIs}...")

    # get data: all_ROI_vox x all_runs_time
    full_data2 = load_process_data(subID, task2, space, ROIs)
    print(f"Full_data shape: {full_data2.shape}")

    # get labels
    label_df2 = get_shifted_labels(task2, 5, rest_tag) #switched it to 4 since it seemed 1 TR off when looking at the evidence values
    print(f"Category label shape: {label_df2.shape}") 


    print(f"\n ***** Decoding the category content in {task2}")

    #will need to build in a real selection method, but for now can take the first versions
    predicts, evidence, true = decode(train_pairs, full_data, label_df, full_data2, label_df2) #this will decode the data using the already trained model and feature selection method

    #need to then save the evidence:
    evidence_df=pd.DataFrame(data=label_df2) #take the study DF for this subject
    evidence_df.drop(columns=evidence_df.columns[0], axis=1, inplace=True) #now drop the extra index column
    evidence_df['rest_evi']=evidence[:,0] #add in the evidence values for rest
    evidence_df['scene_evi']=evidence[:,1] #add in the evidence values for scenes
    evidence_df['face_evi']=evidence[:,2] #add in the evidence values for faces

    #this will export the subject level evidence to the subject's folder
    sub_dir = os.path.join(data_dir, f"sub-{subID}")
    out_fname_template = f"sub-{subID}_{space}_trained-{task}_tested-{task2}_evidence.csv"            
    print("\n *** Saving evidence values with subject dataframe ***")
    evidence_df.to_csv(os.path.join(sub_dir,out_fname_template))

    #get the average evidence for items maintained, replaced, suppressed, and preexposed:
    evidence_df.reset_index(inplace=True,drop=True) #have to reset the index since its wrong after the shift
    maintain_ind=np.where((evidence_df['old_novel']==1) & (evidence_df['condition']==1) & (evidence_df['stim_present']==1))
    maintain_df=evidence_df.loc[maintain_ind]
    maintain_imgs=np.unique(maintain_df['image_id'].values)
    maintain_post_evi={}
    for img in maintain_imgs:
        maintain_post_evi[img]=maintain_df.loc[maintain_df['image_id']==img].scene_evi.mean() #this takes the average of the scene evidence during the 2 TR of viewing

    replace_ind=np.where((evidence_df['old_novel']==1) & (evidence_df['condition']==2) & (evidence_df['stim_present']==1))
    replace_df=evidence_df.loc[replace_ind]
    replace_imgs=np.unique(replace_df['image_id'].values)
    replace_post_evi={}
    for img in replace_imgs:
        replace_post_evi[img]=replace_df.loc[replace_df['image_id']==img].scene_evi.mean()

    suppress_ind=np.where((evidence_df['old_novel']==1) & (evidence_df['condition']==3) & (evidence_df['stim_present']==1))
    suppress_df=evidence_df.loc[suppress_ind]
    suppress_imgs=np.unique(suppress_df['image_id'].values)
    suppress_post_evi={}
    for img in suppress_imgs:
        suppress_post_evi[img]=suppress_df.loc[suppress_df['image_id']==img].scene_evi.mean()  

    preexp_ind=np.where((evidence_df['old_novel']==3) & (evidence_df['stim_present']==1))
    preexp_df=evidence_df.loc[preexp_ind]
    preexp_imgs=np.unique(preexp_df['image_id'].values)
    preexp_post_evi={}
    for img in preexp_imgs:
        preexp_post_evi[img]=preexp_df.loc[preexp_df['image_id']==img].scene_evi.mean()   

    temp_maintain=pd.DataFrame(columns=['image_id','evidence','operation'])
    temp_maintain['image_id']=maintain_post_evi.keys()
    temp_maintain['evidence']=maintain_post_evi.values()
    temp_maintain['operation']='maintain'

    temp_replace=pd.DataFrame(columns=['image_id','evidence','operation'])
    temp_replace['image_id']=replace_post_evi.keys()
    temp_replace['evidence']=replace_post_evi.values()
    temp_replace['operation']='replace'

    temp_suppress=pd.DataFrame(columns=['image_id','evidence','operation'])
    temp_suppress['image_id']=suppress_post_evi.keys()
    temp_suppress['evidence']=suppress_post_evi.values()
    temp_suppress['operation']='suppress'

    temp_preexp=pd.DataFrame(columns=['image_id','evidence','operation'])
    temp_preexp['image_id']=preexp_post_evi.keys()
    temp_preexp['evidence']=preexp_post_evi.values()
    temp_preexp['operation']='preexposed'

    post_removal_df=pd.DataFrame(columns=['image_id','evidence','operation'])
    post_removal_df=pd.concat([temp_maintain,temp_replace,temp_suppress,temp_preexp])

    sub_dir = os.path.join(data_dir, f"sub-{subID}")
    out_fname_template = f"sub-{subID}_{space}_trained-{task}_tested-{task2}_evidence_sorted.csv"            
    print("\n *** Saving evidence values with subject dataframe ***")
    post_removal_df.to_csv(os.path.join(sub_dir,out_fname_template))

    maintain_avg=sum(maintain_post_evi.values()) / len(maintain_post_evi.values())
    replace_avg=sum(replace_post_evi.values()) / len(replace_post_evi.values())
    suppress_avg=sum(suppress_post_evi.values()) / len(suppress_post_evi.values())
    preexp_avg=sum(preexp_post_evi.values()) / len(preexp_post_evi.values())

    subject_avg={'maintain':maintain_avg,'replace':replace_avg,'suppress':suppress_avg,'pre-exp':preexp_avg}

    return subject_avg

def group_post():
    group_df=pd.DataFrame()
    for subID in subIDs:
        temp_sub_average=classification_post(subID)
        temp_df=pd.DataFrame(data=temp_sub_average,index=[subID])

        group_df=pd.concat([group_df,temp_df])

    melt_group_df=group_df.melt()
    melt_group_df.rename(columns={'variable':'operation','value':'evidence'},inplace=True)
    ax=sns.barplot(data=melt_group_df,x='operation',y='evidence',palette=['green','blue','red','gray'],ci=68)
    ax.set(xlabel='Operation', ylabel='Category Classifier Evidence', title='T1w - Category Classifier (group average) - Post Removal')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs','group_level_category_decoding_during_postremoval.png'))
    plt.clf()

def nminus1(subID,save=True):
    task = 'preremoval'
    task2 = 'study'
    space = 'T1w'
    ROIs = ['VVS']

    print("\n *** loading evidence values from subject dataframe ***")

    sub_dir = os.path.join(data_dir, f"sub-{subID}")
    in_fname_template = f"sub-{subID}_{space}_trained-{task}_tested-{task2}_evidence.csv"   

    sub_df=pd.read_csv(os.path.join(sub_dir,in_fname_template))  
    sub_df.drop(columns=sub_df.columns[0], axis=1, inplace=True) #now drop the extra index column

    sub_images,sub_index=np.unique(sub_df['image_id'], return_index=True) #this searches through the dataframe to find each occurance of the image_id. This allows me to find the start of each trial, and linked to the image_ID #

    #now to sort the trials, we need to figure out what the operation performed is:
    sub_condition_list=sub_df['condition'][sub_index].values.astype(int) #so using the above indices, we will now grab what the condition is of each image

    counter=0
    maintain_trials={}
    replace_trials={}
    suppress_trials={}

    for x in range(len(sub_condition_list)):
        if x==0:
            continue
        i=sub_condition_list[x-1]
        if i==0: #this is the first rest period (because we had to shift of hemodynamics. So this "0" condition is nothing)
            print('i==0')
            continue
        elif i==1:
            temp_image=sub_images[x]
            maintain_trials[temp_image]=sub_df[['rest_evi','scene_evi','face_evi']][sub_index[x]-1:sub_index[x]+1].values

        elif i==2:
            temp_image=sub_images[x]            
            replace_trials[temp_image]=sub_df[['rest_evi','scene_evi','face_evi']][sub_index[x]-1:sub_index[x]+1].values

        elif i==3:
            temp_image=sub_images[x]            
            suppress_trials[temp_image]=sub_df[['rest_evi','scene_evi','face_evi']][sub_index[x]-1:sub_index[x]+1].values

    #now that the trials are sorted, we need to get the subject average for each condition:
    avg_maintain=pd.DataFrame(data=np.dstack(maintain_trials.values()).mean(axis=2))
    avg_replace=pd.DataFrame(data=np.dstack(replace_trials.values()).mean(axis=2))
    avg_suppress=pd.DataFrame(data=np.dstack(suppress_trials.values()).mean(axis=2))

    avg_maintain_diff=avg_maintain[1]-avg_maintain[2] #want to also look at difference between Scenes and Faces to evaluate fidelity
    avg_replace_diff=avg_replace[1]-avg_replace[2]
    avg_suppress_diff=avg_suppress[1]-avg_suppress[2]

    #now I will have to change the structure to be able to plot in seaborn:
    avg_maintain=avg_maintain.T.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_maintain['sub']=np.repeat(subID,len(avg_maintain)) #input the subject so I can stack melted dfs
    avg_maintain['evidence_class']=np.tile(['rest','scenes','faces'],2) #add in the labels so we know what each data point is refering to
    avg_maintain.rename(columns={'variable':'TR','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_maintain['condition']='maintain' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_replace=avg_replace.T.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_replace['sub']=np.repeat(subID,len(avg_replace)) #input the subject so I can stack melted dfs
    avg_replace['evidence_class']=np.tile(['rest','scenes','faces'],2) #add in the labels so we know what each data point is refering to
    avg_replace.rename(columns={'variable':'TR','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_replace['condition']='replace' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_suppress=avg_suppress.T.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_suppress['sub']=np.repeat(subID,len(avg_suppress)) #input the subject so I can stack melted dfs
    avg_suppress['evidence_class']=np.tile(['rest','scenes','faces'],2) #add in the labels so we know what each data point is refering to
    avg_suppress.rename(columns={'variable':'TR','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_suppress['condition']='suppress' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_subject_df= pd.concat([avg_maintain,avg_replace,avg_suppress], ignore_index=True, sort=False)

###### repeating the above section but for the difference between scenes and faces

    #now I will have to change the structure to be able to plot in seaborn:
    avg_maintain_diff=pd.DataFrame(data=avg_maintain_diff)
    avg_maintain_diff=avg_maintain_diff.T.melt()
    avg_maintain_diff['sub']=np.repeat(subID,len(avg_maintain_diff)) #input the subject so I can stack melted dfs
    avg_maintain_diff['evidence_class']='diff'
    avg_maintain_diff.rename(columns={'variable':'TR','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_maintain_diff['condition']='maintain' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_replace_diff=pd.DataFrame(data=avg_replace_diff)
    avg_replace_diff=avg_replace_diff.T.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_replace_diff['sub']=np.repeat(subID,len(avg_replace_diff)) #input the subject so I can stack melted dfs
    avg_replace_diff['evidence_class']='diff'
    avg_replace_diff.rename(columns={'variable':'TR','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_replace_diff['condition']='replace' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_suppress_diff=pd.DataFrame(data=avg_suppress_diff)
    avg_suppress_diff=avg_suppress_diff.T.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_suppress_diff['sub']=np.repeat(subID,len(avg_suppress_diff)) #input the subject so I can stack melted dfs
    avg_suppress_diff['evidence_class']='diff'
    avg_suppress_diff.rename(columns={'variable':'TR','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_suppress_diff['condition']='suppress' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_subject_diff_df= pd.concat([avg_maintain_diff,avg_replace_diff,avg_suppress_diff], ignore_index=True, sort=False)

    # save for future use
    if save: 
        sub_dir = os.path.join(data_dir, f"sub-{subID}")
        out_fname_template = f"sub-{subID}_{space}_{task2}_nminus1_evidence_dataframe.csv"  
        print(f"\n Saving the sorted evidence dataframe for {subID} - phase: {task2} - as {out_fname_template}")
        avg_subject_df.to_csv(os.path.join(sub_dir,out_fname_template))

        out_fname_template_diff = f"sub-{subID}_{space}_{task2}_nminus1_evidence_difference_dataframe.csv"  
        print(f"\n Saving the sorted difference evidence dataframe for {subID} - phase: {task2} - as {out_fname_template_diff}")
        avg_subject_df.to_csv(os.path.join(sub_dir,out_fname_template_diff))        
    return avg_subject_df, avg_subject_diff_df

def visualize_nminus1_evidence():
    group_evidence_df=pd.DataFrame()
    group_evidence_diff_df=pd.DataFrame()
    for subID in subIDs:
        temp_subject_df,temp_subject_diff_df=nminus1(subID)
        group_evidence_df=pd.concat([group_evidence_df,temp_subject_df],ignore_index=True, sort=False)
        group_evidence_diff_df=pd.concat([group_evidence_diff_df,temp_subject_diff_df],ignore_index=True, sort=False)


    ax=sns.barplot(data=group_evidence_df.loc[(group_evidence_df['evidence_class']=='scenes')], x='TR',y='evidence',hue='condition' ,ci=68)
    ax.set(xlabel='TR', ylabel='Category Classifier Evidence', title='T1w - Average Category Classifier evidence of N-1 (group average)')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs','group_level_category_decoding_nminus1_TR.png'))
    plt.clf()

    ax=sns.barplot(data=group_evidence_diff_df.loc[(group_evidence_diff_df['evidence_class']=='diff')], x='TR',y='evidence',hue='condition',ci=68)
    ax.set(xlabel='TR', ylabel='Category Classifier Evidence (Scenes - Faces)')
    ax.set_title('T1w - Average Category Classifier evidence of N-1 - Scene minus Face evidence (group average)', loc='center', wrap=True)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs','group_level_category_decoding_nminus1_diff_TR.png'))
    plt.clf()    

    ax=sns.barplot(data=group_evidence_df.loc[(group_evidence_df['evidence_class']=='scenes')], x='condition',y='evidence',ci=68)
    ax.set(xlabel='Operation of N-1', ylabel='Category Classifier Evidence', title='T1w - Average Category Classifier evidence of N-1 (group average)')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs','group_level_category_decoding_nminus1_avg.png'))
    plt.clf()

    ax=sns.barplot(data=group_evidence_diff_df.loc[(group_evidence_diff_df['evidence_class']=='diff')], x='condition',y='evidence',ci=68)
    ax.set(xlabel='Operation of N-1', ylabel='Category Classifier Evidence (Scenes - Faces)')
    ax.set_title('T1w - Average Category Classifier evidence of N-1 - Scene minus Face evidence (group average)', loc='center', wrap=True)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs','group_level_category_decoding_nminus1_diff_avg.png'))
    plt.clf()      
