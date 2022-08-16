#Operation Classifier during Study phase
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
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, PredefinedSplit  #train_test_split, PredefinedSplit, cross_validate, cross_val_predict, 
from sklearn.feature_selection import SelectFpr, f_classif  #VarianceThreshold, SelectKBest, 
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from nilearn.input_data import NiftiMasker,  MultiNiftiMasker
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
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
ROIs = ['wholebrain']
shift_sizes_TR = [5]

save=1
shift_size_TR = shift_sizes_TR[0]


stim_labels = {0: "Rest",
                1: "Scenes",
                2: "Faces"}

operation_labels = {0: "Rest",
                    1: "Maintain",
                    2: "Replace",
                    3: "Suppress"}

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
    if mask_ROIS == ['wholebrain']:  # 'wholebrain'
        # whole brain masks: 1 for each run
        fname_template = f"sub-{subID}_task-study_run-{{}}_space-{space_long}_desc-{{}}.nii.gz" #reseting the template, since we need to load study mask to ensure the same voxels
        runs=np.arange(3) + 1  

        mask_fnames = [fname_template.format(i, "brain_mask") for i in runs]
        mask_paths = [os.path.join(data_dir, f"sub-{subID}", "func", fname) for fname in mask_fnames]
    else:
        # ROI masks: 1 for each ROI
        mask_fnames = [f"{ROI}_{task}_{space}_mask.nii.gz" for ROI in mask_ROIS]
        mask_paths = [os.path.join(data_dir, f"sub-{subID}", "new_mask", fname) for fname in mask_fnames]

    if task=='study':
        runs=np.arange(3) + 1  
    else:
        runs=np.arange(6) + 1
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
    if mask_ROIS == ['wholebrain']:  # 'wholebrain'
        cleaned_bolds = [None for _ in range(len(runs))]
        # all files are by nruns
        #selecting the smallest mask of the three to give a common set of voxels:
        temp=[sum(sum(sum(masks[0].get_fdata()==1))),sum(sum(sum(masks[1].get_fdata()==1))),sum(sum(sum(masks[2].get_fdata()==1)))]
        mask_ind=np.argmin(temp)
        mask=masks[mask_ind]
        for runi, (bold, confound) in enumerate(zip(bolds, confounds_cleaned)):
            print(f"Processing run {runi}...")
            masked = apply_mask(mask=mask.get_fdata(), target=bold.get_fdata())
            # *** clean: confound length in time; transpose signal to (time x vox) for cleaning & model fitting ***
            cleaned_bolds[runi] = clean(masked, confounds=confound, t_r=1, detrend=False, standardize='zscore')
            print("cleaned shape: ", cleaned_bolds[runi].shape)

        # {ROI: time x vox}
        preproc_data = {'wholebrain': np.vstack(cleaned_bolds)}

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

def fit_model(X, Y, runs, save=False, out_fname=None, v=False):
    if v: print("\n***** Fitting model...")

    scores = []
    auc_scores = []
    cms = []
    best_Cs = []
    evidences = []
    roc_aucs=[]

    tested_labels=[]
    pred_probs=[]
    y_scores=[]

    ps = PredefinedSplit(runs)
    for train_inds, test_inds in ps.split():
        X_train, X_test, y_train, y_test = X[train_inds], X[test_inds], Y[train_inds], Y[test_inds]
        
        # feature selection and transformation
        ffpr = SelectFpr(f_classif, alpha=0.01).fit(X_train, y_train)
        X_train_sub = ffpr.transform(X_train)
        X_test_sub = ffpr.transform(X_test)

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

        y_score = lr.decision_function(X_test_sub)
        n_classes=np.unique(y_test).size
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            temp_y=np.zeros(y_test.size)
            label_ind=np.where(y_test==(i+1))
            temp_y[label_ind]=1

            fpr[i], tpr[i], _ = roc_curve(temp_y, y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        auc_score = roc_auc_score(y_test, lr.predict_proba(X_test_sub), multi_class='ovr')
        preds = lr.predict(X_test_sub)
        pred_prob=lr.predict_proba(X_test_sub)
        # confusion matrix
        true_counts = np.asarray([np.sum(y_test == i) for i in [1,2,3]])
        cm = confusion_matrix(y_test, preds, labels=list([1,2,3])) / true_counts[:,None] * 100

        scores.append(score)
        auc_scores.append(auc_score)
        cms.append(cm)
        #calculate evidence values
        evidence=(1. / (1. + np.exp(-lr.decision_function(X_test_sub))))
        evidences.append(evidence) 
        roc_aucs.append(roc_auc)

        tested_labels.append(y_test)
        pred_probs.append(pred_prob)
        y_scores.append(y_score)

    roc_aucs = pd.DataFrame(data=roc_aucs)
    scores = np.asarray(scores)
    auc_scores = np.asarray(auc_scores)
    cms = np.stack(cms)
    evidences = np.stack(evidences)

    tested_labels=np.stack(tested_labels)
    pred_probs=np.stack(pred_probs)
    y_scores=np.stack(y_scores)

    if v: print(f"\nClassifier score: \n"
        f"scores: {scores.mean()} +/- {scores.std()}\n"
        f"auc scores: {auc_scores.mean()} +/- {auc_scores.std()}\n"
        f"best Cs: {best_Cs}\n"
        f"average confution matrix:\n"
        f"{cms.mean(axis=0)}")

    return scores, auc_scores, cms, evidences, roc_aucs, tested_labels, y_scores

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

def classification(subID):
    task = 'study'
    space = 'MNI' #T1w
    ROIs = ['wholebrain']
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
    evidence = []

    X, Y, runs, imgs = sample_for_training(full_data, label_df)
    
    print(f"Running model fitting and cross-validation...")

    # model fitting 
    score, auc_score, cm, evidence, roc_auc, tested_labels, y_scores = fit_model(X, Y, runs, save=False, v=True)

    mean_score=score.mean()

    print(f"\n***** Average results for sub {subID} - {task} - {space}: Score={mean_score} ")

    #want to save the AUC results in such a way that I can also add in the content average later:
    auc_df=pd.DataFrame(columns=['AUC','Content','Sub'],index=['Maintain','Replace','Suppress'])
    auc_df.loc['Maintain']['AUC']=roc_auc.loc[:,0].mean() #Because the above script calculates these based on a leave-one-run-out. We will have an AUC for Maintain, Replace and Suppress per iteration (3 total). So taking the mean of each operation
    auc_df.loc['Replace']['AUC']=roc_auc.loc[:,1].mean()
    auc_df.loc['Suppress']['AUC']=roc_auc.loc[:,2].mean()
    auc_df['Sub']=subID

    sub_dir = os.path.join(data_dir, f"sub-{subID}")
    out_fname_template_auc = f"sub-{subID}_{space}_{task}_operation_auc.csv"            
    print("\n *** Saving AUC values with subject dataframe ***")
    auc_df.to_csv(os.path.join(sub_dir,out_fname_template_auc))    

    #need to then save the evidence:
    evidence_df=pd.DataFrame(columns=['runs','operation','image_id']) #take the study DF for this subject
    evidence_df['runs']=runs
    evidence_df['operation']=Y
    evidence_df['image_id']=imgs
    evidence_df['maintain_evi']=np.vstack(evidence)[:,0] #add in the evidence values for maintain
    evidence_df['replace_evi']=np.vstack(evidence)[:,1] #add in the evidence values for replace
    evidence_df['suppress_evi']=np.vstack(evidence)[:,2] #add in the evidence values for suppress

    #this will export the subject level evidence to the subject's folder
    sub_dir = os.path.join(data_dir, f"sub-{subID}")
    out_fname_template = f"sub-{subID}_{space}_{task}_operation_evidence.csv"            
    print("\n *** Saving evidence values with subject dataframe ***")
    evidence_df.to_csv(os.path.join(sub_dir,out_fname_template))        

def organize_evidence(subID,space,task,save=True):
    ROIs = ['wholebrain']

    print("\n *** loading evidence values from subject dataframe ***")

    sub_dir = os.path.join(data_dir, f"sub-{subID}")
    in_fname_template = f"sub-{subID}_{space}_{task}_operation_evidence.csv"   

    sub_df=pd.read_csv(os.path.join(sub_dir,in_fname_template))  
    sub_df.drop(columns=sub_df.columns[0], axis=1, inplace=True) #now drop the extra index column

    sub_images,sub_index=np.unique(sub_df['image_id'], return_index=True) #this searches through the dataframe to find each occurance of the image_id. This allows me to find the start of each trial, and linked to the image_ID #

    #now to sort the trials, we need to figure out what the operation performed is:
    sub_condition_list=sub_df['operation'][sub_index].values.astype(int) #so using the above indices, we will now grab what the operation is on each image

    counter=0
    maintain_trials={}
    replace_trials={}
    suppress_trials={}

    if task=='study': x=9
    if task=='postremoval': x=5


    for i in sub_condition_list:
        if i==0: #this is the first rest period (because we had to shift of hemodynamics. So this "0" condition is nothing)
            print('i==0')
            counter+=1                         
            continue
        elif i==1:
            temp_image=sub_images[counter]
            maintain_trials[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][sub_index[counter]:sub_index[counter]+x].values
            counter+=1

        elif i==2:
            temp_image=sub_images[counter]            
            replace_trials[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][sub_index[counter]:sub_index[counter]+x].values
            counter+=1

        elif i==3:
            temp_image=sub_images[counter]            
            suppress_trials[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][sub_index[counter]:sub_index[counter]+x].values
            counter+=1

    #now that the trials are sorted, we need to get the subject average for each condition:
    avg_maintain=pd.DataFrame(data=np.dstack(maintain_trials.values()).mean(axis=2))
    avg_replace=pd.DataFrame(data=np.dstack(replace_trials.values()).mean(axis=2))
    avg_suppress=pd.DataFrame(data=np.dstack(suppress_trials.values()).mean(axis=2))

    #now I will have to change the structure to be able to plot in seaborn:
    avg_maintain=avg_maintain.T.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_maintain['sub']=np.repeat(subID,len(avg_maintain)) #input the subject so I can stack melted dfs
    avg_maintain['evidence_class']=np.tile(['maintain','replace','suppress'],x) #add in the labels so we know what each data point is refering to
    avg_maintain.rename(columns={'variable':'TR','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_maintain['condition']='maintain' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_replace=avg_replace.T.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_replace['sub']=np.repeat(subID,len(avg_replace)) #input the subject so I can stack melted dfs
    avg_replace['evidence_class']=np.tile(['maintain','replace','suppress'],x) #add in the labels so we know what each data point is refering to
    avg_replace.rename(columns={'variable':'TR','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_replace['condition']='replace' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_suppress=avg_suppress.T.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_suppress['sub']=np.repeat(subID,len(avg_suppress)) #input the subject so I can stack melted dfs
    avg_suppress['evidence_class']=np.tile(['maintain','replace','suppress'],x) #add in the labels so we know what each data point is refering to
    avg_suppress.rename(columns={'variable':'TR','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_suppress['condition']='suppress' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_subject_df= pd.concat([avg_maintain,avg_replace,avg_suppress], ignore_index=True, sort=False)

    # save for future use
    if save: 
        sub_dir = os.path.join(data_dir, f"sub-{subID}")
        out_fname_template = f"sub-{subID}_{space}_{task}_evidence_dataframe.csv"  
        print(f"\n Saving the sorted evidence dataframe for {subID} - phase: {task} - as {out_fname_template}")
        avg_subject_df.to_csv(os.path.join(sub_dir,out_fname_template))
    return avg_subject_df    

def organize_memory_evidence(subID,space,task,save=True):
    ROIs = ['wholebrain']

    print("\n *** loading evidence values from subject dataframe ***")

    sub_dir = os.path.join(data_dir, f"sub-{subID}")
    in_fname_template = f"sub-{subID}_{space}_{task}_operation_evidence.csv"   

    sub_df=pd.read_csv(os.path.join(sub_dir,in_fname_template))  
    sub_df.drop(columns=sub_df.columns[0], axis=1, inplace=True) #now drop the extra index column

    sub_images,sub_index=np.unique(sub_df['image_id'], return_index=True) #this searches through the dataframe to find each occurance of the image_id. This allows me to find the start of each trial, and linked to the image_ID #

    #now to sort the trials, we need to figure out what the operation performed is:
    sub_condition_list=sub_df['operation'][sub_index].values.astype(int) #so using the above indices, we will now grab what the condition is of each image

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

    if task=='study': x=9
    if task=='postremoval': x=5

    for i in sub_condition_list:
        if i==0: #this is the first rest period (because we had to shift of hemodynamics. So this "0" condition is nothing)
            print('i==0')
            counter+=1                         
            continue
        elif i==1:
            temp_image=sub_images[counter]
            temp_memory=sub_df[sub_df['image_id']==temp_image]['memory'].values[0] #this grabs the first index of the images, and checks the memory outcome

            if temp_memory==1:
                maintain_remember_trials[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][sub_index[counter]:sub_index[counter]+x].values
            elif temp_memory==0:
                maintain_forgot_trials[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][sub_index[counter]:sub_index[counter]+x].values
            counter+=1

        elif i==2:
            temp_image=sub_images[counter]   
            temp_memory=sub_df[sub_df['image_id']==temp_image]['memory'].values[0] #this grabs the first index of the images, and checks the memory outcome

            if temp_memory==1:
                replace_remember_trials[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][sub_index[counter]:sub_index[counter]+x].values
            elif temp_memory==0:
                replace_forgot_trials[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][sub_index[counter]:sub_index[counter]+x].values
            counter+=1

        elif i==3:
            temp_image=sub_images[counter]
            temp_memory=sub_df[sub_df['image_id']==temp_image]['memory'].values[0] #this grabs the first index of the images, and checks the memory outcome

            if temp_memory==1:
                suppress_remember_trials[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][sub_index[counter]:sub_index[counter]+x].values
            elif temp_memory==0:
                suppress_forgot_trials[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][sub_index[counter]:sub_index[counter]+x].values
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
    avg_remember_maintain['evidence_class']=np.tile(['maintain','replace','suppress'],x) #add in the labels so we know what each data point is refering to
    avg_remember_maintain.rename(columns={'variable':'TR','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_remember_maintain['condition']='maintain' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_forgot_maintain=avg_forgot_maintain.T.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_forgot_maintain['sub']=np.repeat(subID,len(avg_forgot_maintain)) #input the subject so I can stack melted dfs
    avg_forgot_maintain['evidence_class']=np.tile(['maintain','replace','suppress'],x) #add in the labels so we know what each data point is refering to
    avg_forgot_maintain.rename(columns={'variable':'TR','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_forgot_maintain['condition']='maintain' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    ####

    avg_remember_replace=avg_remember_replace.T.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_remember_replace['sub']=np.repeat(subID,len(avg_remember_replace)) #input the subject so I can stack melted dfs
    avg_remember_replace['evidence_class']=np.tile(['maintain','replace','suppress'],x) #add in the labels so we know what each data point is refering to
    avg_remember_replace.rename(columns={'variable':'TR','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_remember_replace['condition']='replace' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_forgot_replace=avg_forgot_replace.T.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_forgot_replace['sub']=np.repeat(subID,len(avg_forgot_replace)) #input the subject so I can stack melted dfs
    avg_forgot_replace['evidence_class']=np.tile(['maintain','replace','suppress'],x) #add in the labels so we know what each data point is refering to
    avg_forgot_replace.rename(columns={'variable':'TR','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_forgot_replace['condition']='replace' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    ####

    avg_remember_suppress=avg_remember_suppress.T.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_remember_suppress['sub']=np.repeat(subID,len(avg_remember_suppress)) #input the subject so I can stack melted dfs
    avg_remember_suppress['evidence_class']=np.tile(['maintain','replace','suppress'],x) #add in the labels so we know what each data point is refering to
    avg_remember_suppress.rename(columns={'variable':'TR','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_remember_suppress['condition']='suppress' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_forgot_suppress=avg_forgot_suppress.T.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_forgot_suppress['sub']=np.repeat(subID,len(avg_forgot_suppress)) #input the subject so I can stack melted dfs
    avg_forgot_suppress['evidence_class']=np.tile(['maintain','replace','suppress'],x) #add in the labels so we know what each data point is refering to
    avg_forgot_suppress.rename(columns={'variable':'TR','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_forgot_suppress['condition']='suppress' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_remember_subject_df= pd.concat([avg_remember_maintain,avg_remember_replace,avg_remember_suppress], ignore_index=True, sort=False)

    avg_forgot_subject_df= pd.concat([avg_forgot_maintain,avg_forgot_replace,avg_forgot_suppress], ignore_index=True, sort=False)


    # save for future use
    if save: 
        sub_dir = os.path.join(data_dir, f"sub-{subID}")
        out_fname_template = f"sub-{subID}_{space}_{task}_remember_evidence_dataframe.csv"  
        print(f"\n Saving the sorted remebered evidence dataframe for {subID} - phase: {task} - as {out_fname_template}")
        avg_remember_subject_df.to_csv(os.path.join(sub_dir,out_fname_template))

        out_fname_template2 = f"sub-{subID}_{space}_{task}_forgot_evidence_dataframe.csv"  
        print(f"\n Saving the sorted forgot evidence dataframe for {subID} - phase: {task} - as {out_fname_template}")
        avg_forgot_subject_df.to_csv(os.path.join(sub_dir,out_fname_template2))        

    return avg_remember_subject_df, avg_forgot_subject_df    

def visualize_evidence():
    space='T1w'

    group_evidence_df=pd.DataFrame()
    for subID in subIDs:
        temp_subject_df=organize_evidence(subID,space,'study')
        group_evidence_df=pd.concat([group_evidence_df,temp_subject_df],ignore_index=True, sort=False)

    ax=sns.lineplot(data=group_evidence_df.loc[(group_evidence_df['condition']=='maintain') & (group_evidence_df['evidence_class']=='maintain')], x='TR',y='evidence',color='green',label='maintain', ci=68)
    ax=sns.lineplot(data=group_evidence_df.loc[(group_evidence_df['condition']=='replace') & (group_evidence_df['evidence_class']=='replace')], x='TR',y='evidence',color='blue',label='replace', ci=68)
    ax=sns.lineplot(data=group_evidence_df.loc[(group_evidence_df['condition']=='suppress') & (group_evidence_df['evidence_class']=='suppress')], x='TR',y='evidence',color='red',label='suppress',ci=68)

    plt.legend()
    ax.set(xlabel='TR', ylabel='Operation Classifier Evidence', title=f'{space} - Operation Classifier (group average)')
    ax.set_ylim([0.3,0.9])
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs',f'group_level_{space}_operation_decoding_during_removal.png'))
    plt.clf()

    #now I want to sort the data based on the outcome of that item:
    group_remember_evidence_df=pd.DataFrame()
    group_forgot_evidence_df=pd.DataFrame()

    for subID in subIDs:
        temp_remember_subject_df, temp_forgot_subject_df=organize_memory_evidence(subID,space,'study')

        group_remember_evidence_df=pd.concat([group_remember_evidence_df,temp_remember_subject_df],ignore_index=True, sort=False)    
        group_forgot_evidence_df=pd.concat([group_forgot_evidence_df,temp_forgot_subject_df],ignore_index=True, sort=False)    

    ax=sns.lineplot(data=group_remember_evidence_df.loc[(group_remember_evidence_df['condition']=='maintain') & (group_remember_evidence_df['evidence_class']=='maintain')], x='TR',y='evidence',color='green',label='maintain', ci=68)
    ax=sns.lineplot(data=group_remember_evidence_df.loc[(group_remember_evidence_df['condition']=='replace') & (group_remember_evidence_df['evidence_class']=='replace')], x='TR',y='evidence',color='blue',label='replace', ci=68)
    ax=sns.lineplot(data=group_remember_evidence_df.loc[(group_remember_evidence_df['condition']=='suppress') & (group_remember_evidence_df['evidence_class']=='suppress')], x='TR',y='evidence',color='red',label='suppress',ci=68)

    plt.legend()
    ax.set(xlabel='TR', ylabel='Operation Classifier Evidence', title=f'{space} - Operation Classifier (group average) - Remembered Items')
    ax.set_ylim([0.3,0.9])    
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs',f'group_level_{space}_operation_decoding_during_removal_remember.png'))
    plt.clf()

    ax=sns.lineplot(data=group_forgot_evidence_df.loc[(group_forgot_evidence_df['condition']=='maintain') & (group_forgot_evidence_df['evidence_class']=='maintain')], x='TR',y='evidence',color='green',label='maintain', ci=68)
    ax=sns.lineplot(data=group_forgot_evidence_df.loc[(group_forgot_evidence_df['condition']=='replace') & (group_forgot_evidence_df['evidence_class']=='replace')], x='TR',y='evidence',color='blue',label='replace', ci=68)
    ax=sns.lineplot(data=group_forgot_evidence_df.loc[(group_forgot_evidence_df['condition']=='suppress') & (group_forgot_evidence_df['evidence_class']=='suppress')], x='TR',y='evidence',color='red',label='suppress',ci=68)

    plt.legend()
    ax.set(xlabel='TR', ylabel='Operation Classifier Evidence', title=f'{space} - Operation Classifier (group average) - Forgot Items')
    ax.set_ylim([0.3,0.9])    
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs',f'group_level_{space}_operation_decoding_during_removal_forgot.png'))
    plt.clf()        

    #we have now plotted the evidence of the remembered items and the forgotten items separately, now I wanna plot the difference between remembered and forgotten
    group_diff_evidence_df=group_remember_evidence_df.copy(deep=True)
    group_diff_evidence_df['evidence']=group_remember_evidence_df['evidence']-group_forgot_evidence_df['evidence']    

    ax=sns.lineplot(data=group_diff_evidence_df.loc[(group_diff_evidence_df['condition']=='maintain') & (group_diff_evidence_df['evidence_class']=='maintain')], x='TR',y='evidence',color='green',label='maintain', ci=68)
    ax=sns.lineplot(data=group_diff_evidence_df.loc[(group_diff_evidence_df['condition']=='replace') & (group_diff_evidence_df['evidence_class']=='replace')], x='TR',y='evidence',color='blue',label='replace', ci=68)
    ax=sns.lineplot(data=group_diff_evidence_df.loc[(group_diff_evidence_df['condition']=='suppress') & (group_diff_evidence_df['evidence_class']=='suppress')], x='TR',y='evidence',color='red',label='suppress',ci=68)
    ax.axhline(0,color='k',linestyle='--')

    plt.legend()
    ax.set(xlabel='TR', ylabel='Operation Classifier Evidence (Remember - Forgot)', title=f'{space} - Category Classifier (group average): Remember-Forgot')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs',f'group_level_{space}_operation_decoding_during_removal_difference.png'))
    plt.clf()     

    #plot the traces of remembered and forgotten items during suppress and replace conditions:
    group_remember_evidence_df['memory']='remembered'
    group_forgot_evidence_df['memory']='forgotten'
    group_combined_evidence_df=pd.concat([group_remember_evidence_df,group_forgot_evidence_df],ignore_index=True,sort=False)

    ax=sns.lineplot(data=group_combined_evidence_df.loc[(group_combined_evidence_df['condition']=='maintain') & (group_combined_evidence_df['evidence_class']=='maintain')], x='TR',y='evidence',style='memory',color='green',label='maintain',ci=68)
    ax=sns.lineplot(data=group_combined_evidence_df.loc[(group_combined_evidence_df['condition']=='replace') & (group_combined_evidence_df['evidence_class']=='replace')], x='TR',y='evidence',style='memory',color='blue',label='replace',ci=68)
    ax=sns.lineplot(data=group_combined_evidence_df.loc[(group_combined_evidence_df['condition']=='suppress') & (group_combined_evidence_df['evidence_class']=='suppress')], x='TR',y='evidence',style='memory',color='red',label='suppress',ci=68)

    plt.legend()
    ax.set(xlabel='TR', ylabel='Operation Classifier Evidence', title=f'{space} - Operation Classifier (group average): Remember & Forgot')
    ax.set_ylim([0.3,0.9])    
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs',f'group_level_{space}_operation_decoding_during_removal_RandF.png'))
    plt.clf()     

    #plot the difference between remembered and forgotten, but for each condition separately:
    ax=sns.lineplot(data=group_diff_evidence_df.loc[(group_diff_evidence_df['condition']=='maintain') & (group_diff_evidence_df['evidence_class']=='maintain')], x='TR',y='evidence',color='green',label='maintain', ci=68)
    ax.axhline(0,color='k',linestyle='--')
    plt.legend()
    ax.set(xlabel='TR', ylabel='Operation Classifier Evidence (Remember - Forgot)')
    ax.set_title(f'{space} - Operation Classifier during Maintain (group average): Remember-Forgot', loc='center', wrap=True)    
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs',f'group_level_{space}_operation_decoding_during_maintain_difference.png'))
    plt.clf() 

    ax=sns.lineplot(data=group_diff_evidence_df.loc[(group_diff_evidence_df['condition']=='replace') & (group_diff_evidence_df['evidence_class']=='replace')], x='TR',y='evidence',color='blue',label='replace', ci=68)
    ax.axhline(0,color='k',linestyle='--')
    plt.legend()
    ax.set(xlabel='TR', ylabel='Operation Classifier Evidence (Remember - Forgot)')
    ax.set_title(f'{space} - Operation Classifier during Replace (group average): Remember-Forgot', loc='center', wrap=True)    
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs',f'group_level_{space}_operation_decoding_during_replace_difference.png'))
    plt.clf()

    ax=sns.lineplot(data=group_diff_evidence_df.loc[(group_diff_evidence_df['condition']=='suppress') & (group_diff_evidence_df['evidence_class']=='suppress')], x='TR',y='evidence',color='red',label='suppress',ci=68)
    ax.axhline(0,color='k',linestyle='--')
    plt.legend()
    ax.set(xlabel='TR', ylabel='Operation Classifier Evidence (Remember - Forgot)')
    ax.set_title(f'{space} - Operation Classifier during Suppress (group average): Remember-Forgot', loc='center', wrap=True)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs',f'group_level_{space}_operation_decoding_during_suppress_difference.png'))
    plt.clf()

def train1_test2(X, Y, X2, Y2, save=False, out_fname=None):
    print("\n***** Fitting model...")

    evidences = []
    predicts = []
    true = []
    best_Cs = []

    X_train, X_test, y_train, y_test = X, X2, Y, Y2
    
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

    predicts = lr.predict(X_test_sub)
    #calculate evidence values
    evidence=(1. / (1. + np.exp(-lr.decision_function(X_test_sub))))
    evidences.append(evidence)  

    true = y_test
    evidences = np.stack(evidences)

    return predicts, evidences, true

def post_classification(subID):
    task = 'study'
    task2='postremoval'
    space = 'MNI' #T1w
    ROIs = ['wholebrain']
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
    
    # set up variable for decoding
    
    true = []
    predicts=[]
    evidence = []

    X, Y, runs, imgs = sample_for_training(full_data, label_df)

    #now need to load in the postremoval data:
    print(f"Loading in {task2} data:")
    # get data:
    full_data_2 = load_process_data(subID, task2, space, ROIs)

    print(f"Full_data shape: {full_data_2.shape}")

    # get labels:
    label_df_2 = get_shifted_labels(task2, shift_size_TR, rest_tag)
    label_df_2.reset_index(inplace=True, drop=True)

    print(f"Category label shape: {label_df_2.shape}")    

    task2_inds = np.where(label_df_2['old_novel']==1)
    Y2= label_df_2['condition'][task2_inds[0]].values
    X2 = full_data_2[task2_inds[0]]
    imgs = label_df_2['image_id'][task2_inds[0]].values
    stim = label_df_2['stim_present'][task2_inds[0]].values

    print(f"Running model fitting on {task} and testing on {task2}...")

    # model fitting 
    predicts, evidence, true = train1_test2(X, Y, X2, Y2, save=False)

    #need to then save the evidence:
    evidence_df=pd.DataFrame(columns=['stim_present','operation','predicts','image_id']) #take the study DF for this subject
    evidence_df['stim_present']=stim
    evidence_df['operation']=Y2
    evidence_df['predicts']=predicts
    evidence_df['image_id']=imgs
    evidence_df['maintain_evi']=np.vstack(evidence)[:,0] #add in the evidence values for maintain
    evidence_df['replace_evi']=np.vstack(evidence)[:,1] #add in the evidence values for replace
    evidence_df['suppress_evi']=np.vstack(evidence)[:,2] #add in the evidence values for suppress

    #this will export the subject level evidence to the subject's folder
    sub_dir = os.path.join(data_dir, f"sub-{subID}")
    out_fname_template = f"sub-{subID}_{space}_{task2}_operation_evidence.csv"            
    print("\n *** Saving evidence values with subject dataframe ***")
    evidence_df.to_csv(os.path.join(sub_dir,out_fname_template))        

def visualize_post_evidence(space):
    group_evidence_df=pd.DataFrame()
    for subID in subIDs:
        temp_subject_df=organize_evidence(subID,space,'postremoval')   
        group_evidence_df=pd.concat([group_evidence_df,temp_subject_df],ignore_index=True, sort=False)

    ax=sns.lineplot(data=group_evidence_df.loc[(group_evidence_df['condition']=='maintain') & (group_evidence_df['evidence_class']=='maintain')], x='TR',y='evidence',color='green',label='maintain', ci=68)
    ax=sns.lineplot(data=group_evidence_df.loc[(group_evidence_df['condition']=='replace') & (group_evidence_df['evidence_class']=='replace')], x='TR',y='evidence',color='blue',label='replace', ci=68)
    ax=sns.lineplot(data=group_evidence_df.loc[(group_evidence_df['condition']=='suppress') & (group_evidence_df['evidence_class']=='suppress')], x='TR',y='evidence',color='red',label='suppress',ci=68)

    plt.legend()
    ax.set(xlabel='TR', ylabel='Operation Classifier Evidence', title=f'{space} - Operation Classifier on Post-Removal Phase (group average)')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs',f'group_level_{space}_operation_decoding_during_postremoval.png'))
    plt.clf()        
         
    #now I want to sort the data based on the outcome of that item:
    group_remember_evidence_df=pd.DataFrame()
    group_forgot_evidence_df=pd.DataFrame()

    for subID in subIDs:
        temp_remember_subject_df, temp_forgot_subject_df=organize_memory_evidence(subID,space,'postremoval')

        group_remember_evidence_df=pd.concat([group_remember_evidence_df,temp_remember_subject_df],ignore_index=True, sort=False)    
        group_forgot_evidence_df=pd.concat([group_forgot_evidence_df,temp_forgot_subject_df],ignore_index=True, sort=False)    

    ax=sns.lineplot(data=group_remember_evidence_df.loc[(group_remember_evidence_df['condition']=='maintain') & (group_remember_evidence_df['evidence_class']=='maintain')], x='TR',y='evidence',color='green',label='maintain', ci=68)
    ax=sns.lineplot(data=group_remember_evidence_df.loc[(group_remember_evidence_df['condition']=='replace') & (group_remember_evidence_df['evidence_class']=='replace')], x='TR',y='evidence',color='blue',label='replace', ci=68)
    ax=sns.lineplot(data=group_remember_evidence_df.loc[(group_remember_evidence_df['condition']=='suppress') & (group_remember_evidence_df['evidence_class']=='suppress')], x='TR',y='evidence',color='red',label='suppress',ci=68)

    plt.legend()
    ax.set(xlabel='TR', ylabel='Operation Classifier Evidence', title=f'{space} - Operation Classifier on Post-Removal (group average) - Remembered Items')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs',f'group_level_{space}_operation_decoding_during_postremoval_remember.png'))
    plt.clf()

    ax=sns.lineplot(data=group_forgot_evidence_df.loc[(group_forgot_evidence_df['condition']=='maintain') & (group_forgot_evidence_df['evidence_class']=='maintain')], x='TR',y='evidence',color='green',label='maintain', ci=68)
    ax=sns.lineplot(data=group_forgot_evidence_df.loc[(group_forgot_evidence_df['condition']=='replace') & (group_forgot_evidence_df['evidence_class']=='replace')], x='TR',y='evidence',color='blue',label='replace', ci=68)
    ax=sns.lineplot(data=group_forgot_evidence_df.loc[(group_forgot_evidence_df['condition']=='suppress') & (group_forgot_evidence_df['evidence_class']=='suppress')], x='TR',y='evidence',color='red',label='suppress',ci=68)

    plt.legend()
    ax.set(xlabel='TR', ylabel='Operation Classifier Evidence', title=f'{space} - Operation Classifier on Post-Removal (group average) - Forgot Items')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs',f'group_level_{space}_operation_decoding_during_postremoval_forgot.png'))
    plt.clf()   

    #we have now plotted the evidence of the remembered items and the forgotten items separately, now I wanna plot the difference between remembered and forgotten
    group_diff_evidence_df=group_remember_evidence_df.copy(deep=True)
    group_diff_evidence_df['evidence']=group_remember_evidence_df['evidence']-group_forgot_evidence_df['evidence']    

    ax=sns.lineplot(data=group_diff_evidence_df.loc[(group_diff_evidence_df['condition']=='maintain') & (group_diff_evidence_df['evidence_class']=='maintain')], x='TR',y='evidence',color='green',label='maintain', ci=68)
    ax=sns.lineplot(data=group_diff_evidence_df.loc[(group_diff_evidence_df['condition']=='replace') & (group_diff_evidence_df['evidence_class']=='replace')], x='TR',y='evidence',color='blue',label='replace', ci=68)
    ax=sns.lineplot(data=group_diff_evidence_df.loc[(group_diff_evidence_df['condition']=='suppress') & (group_diff_evidence_df['evidence_class']=='suppress')], x='TR',y='evidence',color='red',label='suppress',ci=68)
    ax.axhline(0,color='k',linestyle='--')

    plt.legend()
    ax.set(xlabel='TR', ylabel='Operation Classifier Evidence (Remember - Forgot)', title=f'{space} - Category Classifier on Post-Removal (group average): Remember-Forgot')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs',f'group_level_{space}_operation_decoding_during_postremoval_difference.png'))
    plt.clf()                       

def organize_evidence_timewindow(subID,space,task,save=True):
    ROIs = ['wholebrain']

    print("\n *** loading evidence values from subject dataframe ***")

    sub_dir = os.path.join(data_dir, f"sub-{subID}")
    in_fname_template = f"sub-{subID}_{space}_{task}_operation_evidence.csv"   

    sub_df=pd.read_csv(os.path.join(sub_dir,in_fname_template))  
    sub_df.drop(columns=sub_df.columns[0], axis=1, inplace=True) #now drop the extra index column

    sub_images,sub_index=np.unique(sub_df['image_id'], return_index=True) #this searches through the dataframe to find each occurance of the image_id. This allows me to find the start of each trial, and linked to the image_ID #

    #now to sort the trials, we need to figure out what the operation performed is:
    sub_condition_list=sub_df['operation'][sub_index].values.astype(int) #so using the above indices, we will now grab what the operation is on each image

    counter=0
    maintain_trials_early={}
    replace_trials_early={}
    suppress_trials_early={}
    maintain_trials_late={}
    replace_trials_late={}
    suppress_trials_late={}    

    maintain_trials_overall={}
    replace_trials_overall={}
    suppress_trials_overall={}

    if task=='study': x=8
    if task=='postremoval': x=5


    for i in sub_condition_list:
        if i==0: #this is the first rest period (because we had to shift of hemodynamics. So this "0" condition is nothing)
            print('i==0')
            counter+=1                         
            continue
        elif i==1:
            temp_image=sub_images[counter]
            maintain_trials_early[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][sub_index[counter]:int(sub_index[counter]+(x/2))].values.mean(axis=0)
            maintain_trials_late[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][int(sub_index[counter]+(x/2)):int(sub_index[counter]+x)].values.mean(axis=0)
            maintain_trials_overall[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][int(sub_index[counter]):int(sub_index[counter]+x)].values.mean(axis=0)
            counter+=1

        elif i==2:
            temp_image=sub_images[counter]            
            replace_trials_early[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][sub_index[counter]:int(sub_index[counter]+(x/2))].values.mean(axis=0)
            replace_trials_late[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][int(sub_index[counter]+(x/2)):int(sub_index[counter]+x)].values.mean(axis=0)            
            replace_trials_overall[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][int(sub_index[counter]):int(sub_index[counter]+x)].values.mean(axis=0)            
            counter+=1

        elif i==3:
            temp_image=sub_images[counter]            
            suppress_trials_early[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][sub_index[counter]:int(sub_index[counter]+(x/2))].values.mean(axis=0)
            suppress_trials_late[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][int(sub_index[counter]+(x/2)):int(sub_index[counter]+x)].values.mean(axis=0)
            suppress_trials_overall[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][int(sub_index[counter]):int(sub_index[counter]+x)].values.mean(axis=0)            
            counter+=1

    #now that the trials are sorted, we need to get the subject average for each condition:
    avg_maintain_early=pd.DataFrame(data=np.dstack(maintain_trials_early.values()).mean(axis=2))
    avg_replace_early=pd.DataFrame(data=np.dstack(replace_trials_early.values()).mean(axis=2))
    avg_suppress_early=pd.DataFrame(data=np.dstack(suppress_trials_early.values()).mean(axis=2))

    avg_maintain_late=pd.DataFrame(data=np.dstack(maintain_trials_late.values()).mean(axis=2))
    avg_replace_late=pd.DataFrame(data=np.dstack(replace_trials_late.values()).mean(axis=2))
    avg_suppress_late=pd.DataFrame(data=np.dstack(suppress_trials_late.values()).mean(axis=2))

    avg_maintain_overall=pd.DataFrame(data=np.dstack(maintain_trials_overall.values()).mean(axis=2))
    avg_replace_overall=pd.DataFrame(data=np.dstack(replace_trials_overall.values()).mean(axis=2))
    avg_suppress_overall=pd.DataFrame(data=np.dstack(suppress_trials_overall.values()).mean(axis=2))    

    maintain_early_df=pd.DataFrame(data=np.dstack(maintain_trials_early.values())[0],columns=maintain_trials_early.keys())
    maintain_early_df.drop(index=[1,2],inplace=True)
    maintain_late_df=pd.DataFrame(data=np.dstack(maintain_trials_late.values())[0],columns=maintain_trials_late.keys())
    maintain_late_df.drop(index=[1,2],inplace=True)
    maintain_overall_df=pd.DataFrame(data=np.dstack(maintain_trials_overall.values())[0],columns=maintain_trials_overall.keys())
    maintain_overall_df.drop(index=[1,2],inplace=True)

    replace_early_df=pd.DataFrame(data=np.dstack(replace_trials_early.values())[0],columns=replace_trials_early.keys())
    replace_early_df.drop(index=[0,2],inplace=True)    
    replace_late_df=pd.DataFrame(data=np.dstack(replace_trials_late.values())[0],columns=replace_trials_late.keys())
    replace_late_df.drop(index=[0,2],inplace=True)    
    replace_overall_df=pd.DataFrame(data=np.dstack(replace_trials_overall.values())[0],columns=replace_trials_overall.keys())
    replace_overall_df.drop(index=[0,2],inplace=True)    

    suppress_early_df=pd.DataFrame(data=np.dstack(suppress_trials_early.values())[0],columns=suppress_trials_early.keys())
    suppress_early_df.drop(index=[0,1],inplace=True)        
    suppress_late_df=pd.DataFrame(data=np.dstack(suppress_trials_late.values())[0],columns=suppress_trials_late.keys())
    suppress_late_df.drop(index=[0,1],inplace=True)     
    suppress_overall_df=pd.DataFrame(data=np.dstack(suppress_trials_overall.values())[0],columns=suppress_trials_overall.keys())
    suppress_overall_df.drop(index=[0,1],inplace=True)       

    #now I will have to change the structure to be able to plot in seaborn:
    ### maintain
    avg_maintain_early=maintain_early_df.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_maintain_early['sub']=np.repeat(subID,len(avg_maintain_early)) #input the subject so I can stack melted dfs
    avg_maintain_early['evidence_class']='maintain' #add in the labels so we know what each data point is refering to
    avg_maintain_early.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_maintain_early['condition']='maintain' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_maintain_late=maintain_late_df.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_maintain_late['sub']=np.repeat(subID,len(avg_maintain_late)) #input the subject so I can stack melted dfs
    avg_maintain_late['evidence_class']='maintain' #add in the labels so we know what each data point is refering to
    avg_maintain_late.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_maintain_late['condition']='maintain' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_maintain_overall=maintain_overall_df.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_maintain_overall['sub']=np.repeat(subID,len(avg_maintain_overall)) #input the subject so I can stack melted dfs
    avg_maintain_overall['evidence_class']='maintain' #add in the labels so we know what each data point is refering to
    avg_maintain_overall.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_maintain_overall['condition']='maintain' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    ### replace
    avg_replace_early=replace_early_df.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_replace_early['sub']=np.repeat(subID,len(avg_replace_early)) #input the subject so I can stack melted dfs
    avg_replace_early['evidence_class']='replace' #add in the labels so we know what each data point is refering to
    avg_replace_early.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_replace_early['condition']='replace' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_replace_late=replace_late_df.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_replace_late['sub']=np.repeat(subID,len(avg_replace_late)) #input the subject so I can stack melted dfs
    avg_replace_late['evidence_class']='replace' #add in the labels so we know what each data point is refering to
    avg_replace_late.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_replace_late['condition']='replace' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_replace_overall=replace_overall_df.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_replace_overall['sub']=np.repeat(subID,len(avg_replace_overall)) #input the subject so I can stack melted dfs
    avg_replace_overall['evidence_class']='maintain' #add in the labels so we know what each data point is refering to
    avg_replace_overall.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_replace_overall['condition']='replace' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    ### suppress
    avg_suppress_early=suppress_early_df.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_suppress_early['sub']=np.repeat(subID,len(avg_suppress_early)) #input the subject so I can stack melted dfs
    avg_suppress_early['evidence_class']='suppress' #add in the labels so we know what each data point is refering to
    avg_suppress_early.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_suppress_early['condition']='suppress' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_suppress_late=suppress_late_df.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_suppress_late['sub']=np.repeat(subID,len(avg_suppress_late)) #input the subject so I can stack melted dfs
    avg_suppress_late['evidence_class']='suppress' #add in the labels so we know what each data point is refering to
    avg_suppress_late.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_suppress_late['condition']='suppress' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_suppress_overall=suppress_overall_df.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_suppress_overall['sub']=np.repeat(subID,len(avg_suppress_overall)) #input the subject so I can stack melted dfs
    avg_suppress_overall['evidence_class']='maintain' #add in the labels so we know what each data point is refering to
    avg_suppress_overall.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_suppress_overall['condition']='suppress' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    ### combine
    avg_subject_early_df= pd.concat([avg_maintain_early,avg_replace_early,avg_suppress_early], ignore_index=True, sort=False)

    avg_subject_late_df= pd.concat([avg_maintain_late,avg_replace_late,avg_suppress_late], ignore_index=True, sort=False)

    avg_subject_df= pd.concat([avg_maintain_overall,avg_replace_overall,avg_suppress_overall], ignore_index=True, sort=False)

    # save for future use
    if save: 
        sub_dir = os.path.join(data_dir, f"sub-{subID}")
        out_fname_template = f"sub-{subID}_{space}_{task}_evidence_early_dataframe.csv"  
        print(f"\n Saving the sorted early evidence dataframe for {subID} - phase: {task} - as {out_fname_template}")
        avg_subject_early_df.to_csv(os.path.join(sub_dir,out_fname_template))

        out_fname_template = f"sub-{subID}_{space}_{task}_evidence_late_dataframe.csv"  
        print(f"\n Saving the sorted late evidence dataframe for {subID} - phase: {task} - as {out_fname_template}")
        avg_subject_late_df.to_csv(os.path.join(sub_dir,out_fname_template))     

        out_fname_template = f"sub-{subID}_{space}_{task}_evidence_overall_dataframe.csv"  
        print(f"\n Saving the sorted overall average evidence dataframe for {subID} - phase: {task} - as {out_fname_template}")
        avg_subject_df.to_csv(os.path.join(sub_dir,out_fname_template))        

    return avg_subject_early_df, avg_subject_late_df, avg_subject_df  


def organize_memory_evidence_timewindow(subID,space,task,save=True):
    ROIs = ['wholebrain']

    print("\n *** loading evidence values from subject dataframe ***")

    sub_dir = os.path.join(data_dir, f"sub-{subID}")
    in_fname_template = f"sub-{subID}_{space}_{task}_operation_evidence.csv"   

    sub_df=pd.read_csv(os.path.join(sub_dir,in_fname_template))  
    sub_df.drop(columns=sub_df.columns[0], axis=1, inplace=True) #now drop the extra index column

    sub_images,sub_index=np.unique(sub_df['image_id'], return_index=True) #this searches through the dataframe to find each occurance of the image_id. This allows me to find the start of each trial, and linked to the image_ID #

    #now to sort the trials, we need to figure out what the operation performed is:
    sub_condition_list=sub_df['operation'][sub_index].values.astype(int) #so using the above indices, we will now grab what the operation is on each image

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
    
    maintain_trials_early_r={}
    replace_trials_early_r={}
    suppress_trials_early_r={}
    maintain_trials_late_r={}
    replace_trials_late_r={}
    suppress_trials_late_r={}    

    maintain_trials_overall_r={}
    replace_trials_overall_r={}
    suppress_trials_overall_r={}

    maintain_trials_early_f={}
    replace_trials_early_f={}
    suppress_trials_early_f={}
    maintain_trials_late_f={}
    replace_trials_late_f={}
    suppress_trials_late_f={}    

    maintain_trials_overall_f={}
    replace_trials_overall_f={}
    suppress_trials_overall_f={}

    if task=='study': x=8
    if task=='postremoval': x=5

    for i in sub_condition_list:
        if i==0: #this is the first rest period (because we had to shift of hemodynamics. So this "0" condition is nothing)
            print('i==0')
            counter+=1                         
            continue
        elif i==1:
            temp_image=sub_images[counter]
            temp_memory=sub_df[sub_df['image_id']==temp_image]['memory'].values[0] #this grabs the first index of the images, and checks the memory outcome

            if temp_memory==1:
                maintain_trials_early_r[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][sub_index[counter]:int(sub_index[counter]+(x/2))].values.mean(axis=0)
                maintain_trials_late_r[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][int(sub_index[counter]+(x/2)):int(sub_index[counter]+x)].values.mean(axis=0)
                maintain_trials_overall_r[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][int(sub_index[counter]):int(sub_index[counter]+x)].values.mean(axis=0)
            elif temp_memory==0:
                maintain_trials_early_f[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][sub_index[counter]:int(sub_index[counter]+(x/2))].values.mean(axis=0)
                maintain_trials_late_f[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][int(sub_index[counter]+(x/2)):int(sub_index[counter]+x)].values.mean(axis=0)
                maintain_trials_overall_f[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][int(sub_index[counter]):int(sub_index[counter]+x)].values.mean(axis=0)
            counter+=1

        elif i==2:
            temp_image=sub_images[counter]   
            temp_memory=sub_df[sub_df['image_id']==temp_image]['memory'].values[0] #this grabs the first index of the images, and checks the memory outcome

            if temp_memory==1:
                replace_trials_early_r[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][sub_index[counter]:int(sub_index[counter]+(x/2))].values.mean(axis=0)
                replace_trials_late_r[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][int(sub_index[counter]+(x/2)):int(sub_index[counter]+x)].values.mean(axis=0)            
                replace_trials_overall_r[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][int(sub_index[counter]):int(sub_index[counter]+x)].values.mean(axis=0) 
            elif temp_memory==0:
                replace_trials_early_f[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][sub_index[counter]:int(sub_index[counter]+(x/2))].values.mean(axis=0)
                replace_trials_late_f[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][int(sub_index[counter]+(x/2)):int(sub_index[counter]+x)].values.mean(axis=0)            
                replace_trials_overall_f[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][int(sub_index[counter]):int(sub_index[counter]+x)].values.mean(axis=0) 
            counter+=1

        elif i==3:
            temp_image=sub_images[counter]
            temp_memory=sub_df[sub_df['image_id']==temp_image]['memory'].values[0] #this grabs the first index of the images, and checks the memory outcome

            if temp_memory==1:
                suppress_trials_early_r[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][sub_index[counter]:int(sub_index[counter]+(x/2))].values.mean(axis=0)
                suppress_trials_late_r[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][int(sub_index[counter]+(x/2)):int(sub_index[counter]+x)].values.mean(axis=0)
                suppress_trials_overall_r[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][int(sub_index[counter]):int(sub_index[counter]+x)].values.mean(axis=0)
            elif temp_memory==0:
                suppress_trials_early_f[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][sub_index[counter]:int(sub_index[counter]+(x/2))].values.mean(axis=0)
                suppress_trials_late_f[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][int(sub_index[counter]+(x/2)):int(sub_index[counter]+x)].values.mean(axis=0)
                suppress_trials_overall_f[temp_image]=sub_df[['maintain_evi','replace_evi','suppress_evi']][int(sub_index[counter]):int(sub_index[counter]+x)].values.mean(axis=0)
            counter+=1

    #now that the trials are sorted, we need to get the subject average for each condition:  

    ##maintain
    maintain_early_df_r=pd.DataFrame(data=np.dstack(maintain_trials_early_r.values())[0],columns=maintain_trials_early_r.keys())
    maintain_early_df_r.drop(index=[1,2],inplace=True)
    if maintain_trials_early_f:
        maintain_early_df_f=pd.DataFrame(data=np.dstack(maintain_trials_early_f.values())[0],columns=maintain_trials_early_f.keys())
        maintain_early_df_f.drop(index=[1,2],inplace=True)
    else:
        maintain_early_df_f=pd.DataFrame()

    maintain_late_df_r=pd.DataFrame(data=np.dstack(maintain_trials_late_r.values())[0],columns=maintain_trials_late_r.keys())
    maintain_late_df_r.drop(index=[1,2],inplace=True)
    if maintain_trials_late_f:
        maintain_late_df_f=pd.DataFrame(data=np.dstack(maintain_trials_late_f.values())[0],columns=maintain_trials_late_f.keys())
        maintain_late_df_f.drop(index=[1,2],inplace=True)
    else:
        maintain_late_df_f=pd.DataFrame()

    maintain_overall_df_r=pd.DataFrame(data=np.dstack(maintain_trials_overall_r.values())[0],columns=maintain_trials_overall_r.keys())
    maintain_overall_df_r.drop(index=[1,2],inplace=True)
    if maintain_trials_overall_f:
        maintain_overall_df_f=pd.DataFrame(data=np.dstack(maintain_trials_overall_f.values())[0],columns=maintain_trials_overall_f.keys())
        maintain_overall_df_f.drop(index=[1,2],inplace=True)
    else: 
        maintain_overall_df_f=pd.DataFrame()


    ## replace
    replace_early_df_r=pd.DataFrame(data=np.dstack(replace_trials_early_r.values())[0],columns=replace_trials_early_r.keys())
    replace_early_df_r.drop(index=[0,2],inplace=True)   
    if replace_trials_early_f:
        replace_early_df_f=pd.DataFrame(data=np.dstack(replace_trials_early_f.values())[0],columns=replace_trials_early_f.keys())
        replace_early_df_f.drop(index=[0,2],inplace=True)   
    else: 
        replace_early_df_f=pd.DataFrame()

    replace_late_df_r=pd.DataFrame(data=np.dstack(replace_trials_late_r.values())[0],columns=replace_trials_late_r.keys())
    replace_late_df_r.drop(index=[0,2],inplace=True)    
    if replace_trials_late_f:
        replace_late_df_f=pd.DataFrame(data=np.dstack(replace_trials_late_f.values())[0],columns=replace_trials_late_f.keys())
        replace_late_df_f.drop(index=[0,2],inplace=True)    
    else:
        replace_late_df_f=pd.DataFrame()

    replace_overall_df_r=pd.DataFrame(data=np.dstack(replace_trials_overall_r.values())[0],columns=replace_trials_overall_r.keys())
    replace_overall_df_r.drop(index=[0,2],inplace=True) 
    if replace_trials_overall_f:
        replace_overall_df_f=pd.DataFrame(data=np.dstack(replace_trials_overall_f.values())[0],columns=replace_trials_overall_f.keys())
        replace_overall_df_f.drop(index=[0,2],inplace=True)          
    else:
        replace_overall_df_f=pd.DataFrame()


    ## suppress
    suppress_early_df_r=pd.DataFrame(data=np.dstack(suppress_trials_early_r.values())[0],columns=suppress_trials_early_r.keys())
    suppress_early_df_r.drop(index=[0,1],inplace=True) 
    if suppress_trials_early_f:
        suppress_early_df_f=pd.DataFrame(data=np.dstack(suppress_trials_early_f.values())[0],columns=suppress_trials_early_f.keys())
        suppress_early_df_f.drop(index=[0,1],inplace=True)  
    else:
        suppress_early_df_f=pd.DataFrame()

    suppress_late_df_r=pd.DataFrame(data=np.dstack(suppress_trials_late_r.values())[0],columns=suppress_trials_late_r.keys())
    suppress_late_df_r.drop(index=[0,1],inplace=True)  
    if suppress_trials_late_f:
        suppress_late_df_f=pd.DataFrame(data=np.dstack(suppress_trials_late_f.values())[0],columns=suppress_trials_late_f.keys())
        suppress_late_df_f.drop(index=[0,1],inplace=True)   
    else:
        suppress_late_df_f=pd.DataFrame()

    suppress_overall_df_r=pd.DataFrame(data=np.dstack(suppress_trials_overall_r.values())[0],columns=suppress_trials_overall_r.keys())
    suppress_overall_df_r.drop(index=[0,1],inplace=True)   
    if suppress_trials_overall_f:
        suppress_overall_df_f=pd.DataFrame(data=np.dstack(suppress_trials_overall_f.values())[0],columns=suppress_trials_overall_f.keys())
        suppress_overall_df_f.drop(index=[0,1],inplace=True)      
    else:
        suppress_overall_df_f=pd.DataFrame()


    #now I will have to change the structure to be able to plot in seaborn:
    ### maintain - remember
    avg_maintain_early_r=maintain_early_df_r.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_maintain_early_r['sub']=np.repeat(subID,len(avg_maintain_early_r)) #input the subject so I can stack melted dfs
    avg_maintain_early_r['evidence_class']='maintain' #add in the labels so we know what each data point is refering to
    avg_maintain_early_r.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_maintain_early_r['condition']='maintain' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_maintain_late_r=maintain_late_df_r.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_maintain_late_r['sub']=np.repeat(subID,len(avg_maintain_late_r)) #input the subject so I can stack melted dfs
    avg_maintain_late_r['evidence_class']='maintain' #add in the labels so we know what each data point is refering to
    avg_maintain_late_r.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_maintain_late_r['condition']='maintain' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_maintain_overall_r=maintain_overall_df_r.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_maintain_overall_r['sub']=np.repeat(subID,len(avg_maintain_overall_r)) #input the subject so I can stack melted dfs
    avg_maintain_overall_r['evidence_class']='maintain' #add in the labels so we know what each data point is refering to
    avg_maintain_overall_r.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_maintain_overall_r['condition']='maintain' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject
    ### maintain - forgot
    avg_maintain_early_f=maintain_early_df_f.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_maintain_early_f['sub']=np.repeat(subID,len(avg_maintain_early_f)) #input the subject so I can stack melted dfs
    avg_maintain_early_f['evidence_class']='maintain' #add in the labels so we know what each data point is refering to
    avg_maintain_early_f.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_maintain_early_f['condition']='maintain' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_maintain_late_f=maintain_late_df_f.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_maintain_late_f['sub']=np.repeat(subID,len(avg_maintain_late_f)) #input the subject so I can stack melted dfs
    avg_maintain_late_f['evidence_class']='maintain' #add in the labels so we know what each data point is refering to
    avg_maintain_late_f.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_maintain_late_f['condition']='maintain' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_maintain_overall_f=maintain_overall_df_f.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_maintain_overall_f['sub']=np.repeat(subID,len(avg_maintain_overall_f)) #input the subject so I can stack melted dfs
    avg_maintain_overall_f['evidence_class']='maintain' #add in the labels so we know what each data point is refering to
    avg_maintain_overall_f.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_maintain_overall_f['condition']='maintain' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject


    ### replace - remember
    avg_replace_early_r=replace_early_df_r.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_replace_early_r['sub']=np.repeat(subID,len(avg_replace_early_r)) #input the subject so I can stack melted dfs
    avg_replace_early_r['evidence_class']='replace' #add in the labels so we know what each data point is refering to
    avg_replace_early_r.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_replace_early_r['condition']='replace' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_replace_late_r=replace_late_df_r.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_replace_late_r['sub']=np.repeat(subID,len(avg_replace_late_r)) #input the subject so I can stack melted dfs
    avg_replace_late_r['evidence_class']='replace' #add in the labels so we know what each data point is refering to
    avg_replace_late_r.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_replace_late_r['condition']='replace' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_replace_overall_r=replace_overall_df_r.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_replace_overall_r['sub']=np.repeat(subID,len(avg_replace_overall_r)) #input the subject so I can stack melted dfs
    avg_replace_overall_r['evidence_class']='replace' #add in the labels so we know what each data point is refering to
    avg_replace_overall_r.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_replace_overall_r['condition']='replace' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject
    ### replace - forgot
    avg_replace_early_f=replace_early_df_f.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_replace_early_f['sub']=np.repeat(subID,len(avg_replace_early_f)) #input the subject so I can stack melted dfs
    avg_replace_early_f['evidence_class']='replace' #add in the labels so we know what each data point is refering to
    avg_replace_early_f.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_replace_early_f['condition']='replace' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_replace_late_f=replace_late_df_f.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_replace_late_f['sub']=np.repeat(subID,len(avg_replace_late_f)) #input the subject so I can stack melted dfs
    avg_replace_late_f['evidence_class']='replace' #add in the labels so we know what each data point is refering to
    avg_replace_late_f.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_replace_late_f['condition']='replace' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_replace_overall_f=replace_overall_df_f.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_replace_overall_f['sub']=np.repeat(subID,len(avg_replace_overall_f)) #input the subject so I can stack melted dfs
    avg_replace_overall_f['evidence_class']='replace' #add in the labels so we know what each data point is refering to
    avg_replace_overall_f.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_replace_overall_f['condition']='replace' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject


    ### suppress - remember
    avg_suppress_early_r=suppress_early_df_r.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_suppress_early_r['sub']=np.repeat(subID,len(avg_suppress_early_r)) #input the subject so I can stack melted dfs
    avg_suppress_early_r['evidence_class']='suppress' #add in the labels so we know what each data point is refering to
    avg_suppress_early_r.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_suppress_early_r['condition']='suppress' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_suppress_late_r=suppress_late_df_r.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_suppress_late_r['sub']=np.repeat(subID,len(avg_suppress_late_r)) #input the subject so I can stack melted dfs
    avg_suppress_late_r['evidence_class']='suppress' #add in the labels so we know what each data point is refering to
    avg_suppress_late_r.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_suppress_late_r['condition']='suppress' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_suppress_overall_r=suppress_overall_df_r.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_suppress_overall_r['sub']=np.repeat(subID,len(avg_suppress_overall_r)) #input the subject so I can stack melted dfs
    avg_suppress_overall_r['evidence_class']='suppress' #add in the labels so we know what each data point is refering to
    avg_suppress_overall_r.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_suppress_overall_r['condition']='suppress' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject
    ### suppress - forgot
    avg_suppress_early_f=suppress_early_df_f.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_suppress_early_f['sub']=np.repeat(subID,len(avg_suppress_early_f)) #input the subject so I can stack melted dfs
    avg_suppress_early_f['evidence_class']='suppress' #add in the labels so we know what each data point is refering to
    avg_suppress_early_f.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_suppress_early_f['condition']='suppress' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_suppress_late_f=suppress_late_df_f.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_suppress_late_f['sub']=np.repeat(subID,len(avg_suppress_late_f)) #input the subject so I can stack melted dfs
    avg_suppress_late_f['evidence_class']='suppress' #add in the labels so we know what each data point is refering to
    avg_suppress_late_f.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_suppress_late_f['condition']='suppress' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_suppress_overall_f=suppress_overall_df_f.melt() #now you get 2 columns: variable (TR) and value (evidence)
    avg_suppress_overall_f['sub']=np.repeat(subID,len(avg_suppress_overall_f)) #input the subject so I can stack melted dfs
    avg_suppress_overall_f['evidence_class']='suppress' #add in the labels so we know what each data point is refering to
    avg_suppress_overall_f.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_suppress_overall_f['condition']='suppress' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject


    ### combine
    avg_subject_early_remember_df= pd.concat([avg_maintain_early_r,avg_replace_early_r,avg_suppress_early_r], ignore_index=True, sort=False)
    avg_subject_early_forgot_df= pd.concat([avg_maintain_early_f,avg_replace_early_f,avg_suppress_early_f], ignore_index=True, sort=False)

    avg_subject_late_remember_df= pd.concat([avg_maintain_late_r,avg_replace_late_r,avg_suppress_late_r], ignore_index=True, sort=False)
    avg_subject_late_forgot_df= pd.concat([avg_maintain_late_f,avg_replace_late_f,avg_suppress_late_f], ignore_index=True, sort=False)

    avg_subject_remember_df= pd.concat([avg_maintain_overall_r,avg_replace_overall_r,avg_suppress_overall_r], ignore_index=True, sort=False)
    avg_subject_forgot_df= pd.concat([avg_maintain_overall_f,avg_replace_overall_f,avg_suppress_overall_f], ignore_index=True, sort=False)


    # save for future use
    if save: 
        sub_dir = os.path.join(data_dir, f"sub-{subID}")
        out_fname_template = f"sub-{subID}_{space}_{task}_evidence_remember_early_dataframe.csv"  
        print(f"\n Saving the sorted early evidence remembered dataframe for {subID} - phase: {task} - as {out_fname_template}")
        avg_subject_early_remember_df.to_csv(os.path.join(sub_dir,out_fname_template))

        out_fname_template = f"sub-{subID}_{space}_{task}_evidence_remember_late_dataframe.csv"  
        print(f"\n Saving the sorted late evidence remembered dataframe for {subID} - phase: {task} - as {out_fname_template}")
        avg_subject_late_remember_df.to_csv(os.path.join(sub_dir,out_fname_template))     

        out_fname_template = f"sub-{subID}_{space}_{task}_evidence_remember_overall_dataframe.csv"  
        print(f"\n Saving the sorted overall average evidence remembered dataframe for {subID} - phase: {task} - as {out_fname_template}")
        avg_subject_remember_df.to_csv(os.path.join(sub_dir,out_fname_template))  

        out_fname_template = f"sub-{subID}_{space}_{task}_evidence_forgot_early_dataframe.csv"  
        print(f"\n Saving the sorted early evidence forgotten dataframe for {subID} - phase: {task} - as {out_fname_template}")
        avg_subject_early_forgot_df.to_csv(os.path.join(sub_dir,out_fname_template))

        out_fname_template = f"sub-{subID}_{space}_{task}_evidence_forgot_late_dataframe.csv"  
        print(f"\n Saving the sorted late evidence forgotten dataframe for {subID} - phase: {task} - as {out_fname_template}")
        avg_subject_late_forgot_df.to_csv(os.path.join(sub_dir,out_fname_template))     

        out_fname_template = f"sub-{subID}_{space}_{task}_evidence_forgot_overall_dataframe.csv"  
        print(f"\n Saving the sorted overall average evidence forgotten dataframe for {subID} - phase: {task} - as {out_fname_template}")
        avg_subject_forgot_df.to_csv(os.path.join(sub_dir,out_fname_template))                                 

    return avg_subject_early_remember_df, avg_subject_early_forgot_df, avg_subject_late_remember_df, avg_subject_late_forgot_df, avg_subject_remember_df, avg_subject_forgot_df

def coef_stim_operation(subID,save=True):
    task = 'preremoval'
    task2 = 'study'
    space = 'MNI'
    ROIs = ['VVS']

    print("\n *** loading Category evidence values from subject dataframe ***")

    sub_dir = os.path.join(data_dir, f"sub-{subID}")
    in_fname_template = f"sub-{subID}_{space}_trained-{task}_tested-{task2}_evidence.csv"   

    sub_df=pd.read_csv(os.path.join(sub_dir,in_fname_template))  
    sub_df.drop(columns=sub_df.columns[0], axis=1, inplace=True) #now drop the extra index column

    sub_images,sub_index=np.unique(sub_df['image_id'], return_index=True) #this searches through the dataframe to find each occurance of the image_id. This allows me to find the start of each trial, and linked to the image_ID #

    #now to sort the trials, we need to figure out what the operation performed is:
    sub_condition_list=sub_df['condition'][sub_index].values.astype(int) #so using the above indices, we will now grab what the condition is of each image

    print("\n *** loading Operation evidence values from subject dataframe ***")

    avg_subject_early_df, avg_subject_late_df, avg_subject_overall_df =organize_evidence_timewindow(subID,space,task2)

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
            maintain_trials[temp_image]=sub_df[['rest_evi','scene_evi','face_evi']][sub_index[counter]+2:sub_index[counter]+6].values.mean(axis=0)[1] #taking the average of the 4 operation TRs and then only pulling out the scene evidence
            counter+=1

        elif i==2:
            temp_image=sub_images[counter]            
            replace_trials[temp_image]=sub_df[['rest_evi','scene_evi','face_evi']][sub_index[counter]+2:sub_index[counter]+6].values.mean(axis=0)[1]
            counter+=1

        elif i==3:
            temp_image=sub_images[counter]            
            suppress_trials[temp_image]=sub_df[['rest_evi','scene_evi','face_evi']][sub_index[counter]+2:sub_index[counter]+6].values.mean(axis=0)[1]
            counter+=1

    print("\n *** loading Operation AUC values from subject dataframe ***")
    sub_dir = os.path.join(data_dir, f"sub-{subID}")
    out_fname_template_auc = f"sub-{subID}_{space}_{task}_operation_auc.csv"            
 
    auc_df=pd.read_csv(os.path.join(sub_dir,f"sub-{subID}_{space}_{task2}_operation_auc.csv"),index_col=0) 

    #now that the trials are sorted, we need to get the subject average for each condition:
    avg_maintain_df=pd.DataFrame(data=np.dstack(maintain_trials.values())[0],columns=maintain_trials.keys())
    avg_maintain_df=avg_maintain_df.melt() #now you get 2 columns: variable and value (evidence)
    avg_maintain_df['sub']=np.repeat(subID,len(avg_maintain_df)) #input the subject so I can stack melted dfs
    avg_maintain_df['evidence_class']='scene' #add in the labels so we know what each data point is refering to
    avg_maintain_df.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_maintain_df['condition']='maintain' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_replace_df=pd.DataFrame(data=np.dstack(replace_trials.values())[0],columns=replace_trials.keys())
    avg_replace_df=avg_replace_df.melt() #now you get 2 columns: variable and value (evidence)
    avg_replace_df['sub']=np.repeat(subID,len(avg_replace_df)) #input the subject so I can stack melted dfs
    avg_replace_df['evidence_class']='scene' #add in the labels so we know what each data point is refering to
    avg_replace_df.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_replace_df['condition']='replace' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_suppress_df=pd.DataFrame(data=np.dstack(suppress_trials.values())[0],columns=suppress_trials.keys())
    avg_suppress_df=avg_suppress_df.melt() #now you get 2 columns: variable and value (evidence)
    avg_suppress_df['sub']=np.repeat(subID,len(avg_suppress_df)) #input the subject so I can stack melted dfs
    avg_suppress_df['evidence_class']='scene' #add in the labels so we know what each data point is refering to
    avg_suppress_df.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_suppress_df['condition']='suppress' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject
    
    #use these above arrays to get the operation average for the removal period content evidence:
    auc_df.loc['Maintain','Content']=avg_maintain_df['evidence'].mean()
    auc_df.loc['Replace','Content']=avg_replace_df['evidence'].mean()
    auc_df.loc['Suppress','Content']=avg_suppress_df['evidence'].mean()

    avg_subject_category_df= pd.concat([avg_maintain_df,avg_replace_df,avg_suppress_df], ignore_index=True, sort=False)

    maintain_early_operation_evi=avg_subject_early_df.loc[avg_subject_early_df['condition']=='maintain']['evidence'].values.reshape(-1,1)
    maintain_late_operation_evi=avg_subject_late_df.loc[avg_subject_late_df['condition']=='maintain']['evidence'].values.reshape(-1,1)
    maintain_overall_operation_evi=avg_subject_overall_df.loc[avg_subject_overall_df['condition']=='maintain']['evidence'].values.reshape(-1,1)

    maintain_category_evi=avg_maintain_df['evidence'].values.reshape(-1,1)

    replace_early_operation_evi=avg_subject_early_df.loc[avg_subject_early_df['condition']=='replace']['evidence'].values.reshape(-1,1)
    replace_late_operation_evi=avg_subject_late_df.loc[avg_subject_late_df['condition']=='replace']['evidence'].values.reshape(-1,1)
    replace_overall_operation_evi=avg_subject_overall_df.loc[avg_subject_overall_df['condition']=='replace']['evidence'].values.reshape(-1,1)

    replace_category_evi=avg_replace_df['evidence'].values.reshape(-1,1)

    suppress_early_operation_evi=avg_subject_early_df.loc[avg_subject_early_df['condition']=='suppress']['evidence'].values.reshape(-1,1)
    suppress_late_operation_evi=avg_subject_late_df.loc[avg_subject_late_df['condition']=='suppress']['evidence'].values.reshape(-1,1)
    suppress_overall_operation_evi=avg_subject_overall_df.loc[avg_subject_overall_df['condition']=='suppress']['evidence'].values.reshape(-1,1)

    suppress_category_evi=avg_suppress_df['evidence'].values.reshape(-1,1)    

    subject_coef_df=pd.DataFrame(columns=['sub','condition','beta','timing'],index=[0,1,2,3,4,5,6,7,8])
    subject_coef_df['sub']=subID
    subject_coef_df['condition']=np.repeat(['maintain','replace','suppress'],3)

    #first taking the beta from correlation between early/late operation evidence and category evidence during 2TR
    maintain_early_lr = LinearRegression().fit(maintain_early_operation_evi,maintain_category_evi)
    subject_coef_df.loc[0,'beta']=maintain_early_lr.coef_[0][0]
    subject_coef_df.loc[0,'timing']='early'
    maintain_late_lr = LinearRegression().fit(maintain_late_operation_evi,maintain_category_evi)
    subject_coef_df.loc[1,'beta']=maintain_late_lr.coef_[0][0]
    subject_coef_df.loc[1,'timing']='late'
    maintain_overall_lr = LinearRegression().fit(maintain_overall_operation_evi,maintain_category_evi)
    subject_coef_df.loc[2,'beta']=maintain_overall_lr.coef_[0][0]
    subject_coef_df.loc[2,'timing']='overall'    

    replace_early_lr = LinearRegression().fit(replace_early_operation_evi,replace_category_evi)
    subject_coef_df.loc[3,'beta']=replace_early_lr.coef_[0][0]
    subject_coef_df.loc[3,'timing']='early'
    replace_late_lr = LinearRegression().fit(replace_late_operation_evi,replace_category_evi)
    subject_coef_df.loc[4,'beta']=replace_late_lr.coef_[0][0]
    subject_coef_df.loc[4,'timing']='late'
    replace_overall_lr = LinearRegression().fit(replace_overall_operation_evi,replace_category_evi)
    subject_coef_df.loc[5,'beta']=replace_overall_lr.coef_[0][0]
    subject_coef_df.loc[5,'timing']='overall'    

    suppress_early_lr = LinearRegression().fit(suppress_early_operation_evi,suppress_category_evi)
    subject_coef_df.loc[6,'beta']=suppress_early_lr.coef_[0][0]
    subject_coef_df.loc[6,'timing']='early'
    suppress_late_lr = LinearRegression().fit(suppress_late_operation_evi,suppress_category_evi)
    subject_coef_df.loc[7,'beta']=suppress_late_lr.coef_[0][0]
    subject_coef_df.loc[7,'timing']='late'
    suppress_overall_lr = LinearRegression().fit(suppress_overall_operation_evi,suppress_category_evi)
    subject_coef_df.loc[8,'beta']=suppress_overall_lr.coef_[0][0]
    subject_coef_df.loc[8,'timing']='overall'
    # save for future use
    if save: 
        sub_dir = os.path.join(data_dir, f"sub-{subID}")
        out_fname_template = f"sub-{subID}_{space}_{task2}_coef_dataframe.csv"  
        print(f"\n Saving the beta's from Operation predicting Content for {subID} - space: {space} - as {out_fname_template}")
        subject_coef_df.to_csv(os.path.join(sub_dir,out_fname_template))      

        out_fname_template_auc = f"sub-{subID}_{space}_{task2}_auc_dataframe.csv"  
        print(f"\n Saving the AUC's from Operation along w/ Content evidence for {subID} - space: {space} - as {out_fname_template_auc}")
        auc_df.to_csv(os.path.join(sub_dir,out_fname_template_auc))

    return subject_coef_df, auc_df

def coef_stim_memory_operation(subID,save=True):
    task = 'preremoval'
    task2 = 'study'
    space = 'MNI'
    ROIs = ['VVS']

    print("\n *** loading Category evidence values from subject dataframe ***")

    sub_dir = os.path.join(data_dir, f"sub-{subID}")
    in_fname_template = f"sub-{subID}_{space}_trained-{task}_tested-{task2}_evidence.csv"   

    sub_df=pd.read_csv(os.path.join(sub_dir,in_fname_template))  
    sub_df.drop(columns=sub_df.columns[0], axis=1, inplace=True) #now drop the extra index column

    sub_images,sub_index=np.unique(sub_df['image_id'], return_index=True) #this searches through the dataframe to find each occurance of the image_id. This allows me to find the start of each trial, and linked to the image_ID #

    #now to sort the trials, we need to figure out what the operation performed is:
    sub_condition_list=sub_df['condition'][sub_index].values.astype(int) #so using the above indices, we will now grab what the condition is of each image

    print("\n *** loading Operation evidence values from subject dataframe ***")

    avg_subject_early_remember_df, avg_subject_early_forgot_df, avg_subject_late_remember_df, avg_subject_late_forgot_df, avg_subject_overall_remember_df, avg_subject_overall_forgot_df =organize_memory_evidence_timewindow(subID,space,task2)

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
            maintain_trials[temp_image]=sub_df[['rest_evi','scene_evi','face_evi']][sub_index[counter]+2:sub_index[counter]+6].values.mean(axis=0)[1] #taking the average of the 4 operation TRs and then only pulling out the scene evidence
            counter+=1

        elif i==2:
            temp_image=sub_images[counter]            
            replace_trials[temp_image]=sub_df[['rest_evi','scene_evi','face_evi']][sub_index[counter]+2:sub_index[counter]+6].values.mean(axis=0)[1]
            counter+=1

        elif i==3:
            temp_image=sub_images[counter]            
            suppress_trials[temp_image]=sub_df[['rest_evi','scene_evi','face_evi']][sub_index[counter]+2:sub_index[counter]+6].values.mean(axis=0)[1]
            counter+=1

    #now that the trials are sorted, we need to get the subject average for each condition:
    avg_maintain_df=pd.DataFrame(data=np.dstack(maintain_trials.values())[0],columns=maintain_trials.keys())
    avg_maintain_df=avg_maintain_df.melt() #now you get 2 columns: variable and value (evidence)
    avg_maintain_df['sub']=np.repeat(subID,len(avg_maintain_df)) #input the subject so I can stack melted dfs
    avg_maintain_df['evidence_class']='scene' #add in the labels so we know what each data point is refering to
    avg_maintain_df.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_maintain_df['condition']='maintain' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_replace_df=pd.DataFrame(data=np.dstack(replace_trials.values())[0],columns=replace_trials.keys())
    avg_replace_df=avg_replace_df.melt() #now you get 2 columns: variable and value (evidence)
    avg_replace_df['sub']=np.repeat(subID,len(avg_replace_df)) #input the subject so I can stack melted dfs
    avg_replace_df['evidence_class']='scene' #add in the labels so we know what each data point is refering to
    avg_replace_df.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_replace_df['condition']='replace' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

    avg_suppress_df=pd.DataFrame(data=np.dstack(suppress_trials.values())[0],columns=suppress_trials.keys())
    avg_suppress_df=avg_suppress_df.melt() #now you get 2 columns: variable and value (evidence)
    avg_suppress_df['sub']=np.repeat(subID,len(avg_suppress_df)) #input the subject so I can stack melted dfs
    avg_suppress_df['evidence_class']='scene' #add in the labels so we know what each data point is refering to
    avg_suppress_df.rename(columns={'variable':'image_id','value':'evidence'},inplace=True) #renamed the melted column names 
    avg_suppress_df['condition']='suppress' #now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject
    
    avg_subject_category_df= pd.concat([avg_maintain_df,avg_replace_df,avg_suppress_df], ignore_index=True, sort=False)


    ## maintain
    maintain_early_operation_remember_evi=avg_subject_early_remember_df.loc[avg_subject_early_remember_df['condition']=='maintain']['evidence'].values.reshape(-1,1)
    maintain_late_operation_remember_evi=avg_subject_late_remember_df.loc[avg_subject_late_remember_df['condition']=='maintain']['evidence'].values.reshape(-1,1)
    maintain_overall_operation_remember_evi=avg_subject_overall_remember_df.loc[avg_subject_overall_remember_df['condition']=='maintain']['evidence'].values.reshape(-1,1)

    maintain_early_operation_forgot_evi=avg_subject_early_forgot_df.loc[avg_subject_early_forgot_df['condition']=='maintain']['evidence'].values.reshape(-1,1)
    maintain_late_operation_forgot_evi=avg_subject_late_forgot_df.loc[avg_subject_late_forgot_df['condition']=='maintain']['evidence'].values.reshape(-1,1)
    maintain_overall_operation_forgot_evi=avg_subject_overall_forgot_df.loc[avg_subject_overall_forgot_df['condition']=='maintain']['evidence'].values.reshape(-1,1)

    maintain_remember_category_evi=avg_maintain_df[avg_maintain_df.image_id.isin(avg_subject_early_remember_df.loc[avg_subject_early_remember_df['condition']=='maintain'].image_id)]['evidence'].values.reshape(-1,1)
    maintain_forgot_category_evi=avg_maintain_df[avg_maintain_df.image_id.isin(avg_subject_early_forgot_df.loc[avg_subject_early_forgot_df['condition']=='maintain'].image_id)]['evidence'].values.reshape(-1,1)


    ## replace
    replace_early_operation_remember_evi=avg_subject_early_remember_df.loc[avg_subject_early_remember_df['condition']=='replace']['evidence'].values.reshape(-1,1)
    replace_late_operation_remember_evi=avg_subject_late_remember_df.loc[avg_subject_late_remember_df['condition']=='replace']['evidence'].values.reshape(-1,1)
    replace_overall_operation_remember_evi=avg_subject_overall_remember_df.loc[avg_subject_overall_remember_df['condition']=='replace']['evidence'].values.reshape(-1,1)

    replace_early_operation_forgot_evi=avg_subject_early_forgot_df.loc[avg_subject_early_forgot_df['condition']=='replace']['evidence'].values.reshape(-1,1)
    replace_late_operation_forgot_evi=avg_subject_late_forgot_df.loc[avg_subject_late_forgot_df['condition']=='replace']['evidence'].values.reshape(-1,1)
    replace_overall_operation_forgot_evi=avg_subject_overall_forgot_df.loc[avg_subject_overall_forgot_df['condition']=='replace']['evidence'].values.reshape(-1,1)

    replace_remember_category_evi=avg_replace_df[avg_replace_df.image_id.isin(avg_subject_early_remember_df.loc[avg_subject_early_remember_df['condition']=='replace'].image_id)]['evidence'].values.reshape(-1,1)
    replace_forgot_category_evi=avg_replace_df[avg_replace_df.image_id.isin(avg_subject_early_forgot_df.loc[avg_subject_early_forgot_df['condition']=='replace'].image_id)]['evidence'].values.reshape(-1,1)


    ## suppress
    suppress_early_operation_remember_evi=avg_subject_early_remember_df.loc[avg_subject_early_remember_df['condition']=='suppress']['evidence'].values.reshape(-1,1)
    suppress_late_operation_remember_evi=avg_subject_late_remember_df.loc[avg_subject_late_remember_df['condition']=='suppress']['evidence'].values.reshape(-1,1)
    suppress_overall_operation_remember_evi=avg_subject_overall_remember_df.loc[avg_subject_overall_remember_df['condition']=='suppress']['evidence'].values.reshape(-1,1)

    suppress_early_operation_forgot_evi=avg_subject_early_forgot_df.loc[avg_subject_early_forgot_df['condition']=='suppress']['evidence'].values.reshape(-1,1)
    suppress_late_operation_forgot_evi=avg_subject_late_forgot_df.loc[avg_subject_late_forgot_df['condition']=='suppress']['evidence'].values.reshape(-1,1)
    suppress_overall_operation_forgot_evi=avg_subject_overall_forgot_df.loc[avg_subject_overall_forgot_df['condition']=='suppress']['evidence'].values.reshape(-1,1)    

    suppress_remember_category_evi=avg_suppress_df[avg_suppress_df.image_id.isin(avg_subject_early_remember_df.loc[avg_subject_early_remember_df['condition']=='suppress'].image_id)]['evidence'].values.reshape(-1,1)
    suppress_forgot_category_evi=avg_suppress_df[avg_suppress_df.image_id.isin(avg_subject_early_forgot_df.loc[avg_subject_early_forgot_df['condition']=='suppress'].image_id)]['evidence'].values.reshape(-1,1)


    ## combine
    subject_coef_remember_df=pd.DataFrame(columns=['sub','condition','beta','timing'],index=[0,1,2,3,4,5,6,7,8])
    subject_coef_remember_df['sub']=subID
    subject_coef_remember_df['condition']=np.repeat(['maintain','replace','suppress'],3)

    subject_coef_forgot_df=pd.DataFrame(columns=['sub','condition','beta','timing'],index=[0,1,2,3,4,5,6,7,8])
    subject_coef_forgot_df['sub']=subID
    subject_coef_forgot_df['condition']=np.repeat(['maintain','replace','suppress'],3)    

    #first taking the beta from correlation between early/late operation evidence and category evidence
    ## Remember
    maintain_early_lr = LinearRegression().fit(maintain_early_operation_remember_evi,maintain_remember_category_evi)
    subject_coef_remember_df.loc[0,'beta']=maintain_early_lr.coef_[0][0]
    subject_coef_remember_df.loc[0,'timing']='early'
    maintain_late_lr = LinearRegression().fit(maintain_late_operation_remember_evi,maintain_remember_category_evi)
    subject_coef_remember_df.loc[1,'beta']=maintain_late_lr.coef_[0][0]
    subject_coef_remember_df.loc[1,'timing']='late'
    maintain_overall_lr = LinearRegression().fit(maintain_overall_operation_remember_evi,maintain_remember_category_evi)
    subject_coef_remember_df.loc[2,'beta']=maintain_overall_lr.coef_[0][0]
    subject_coef_remember_df.loc[2,'timing']='overall'    

    replace_early_lr = LinearRegression().fit(replace_early_operation_remember_evi,replace_remember_category_evi)
    subject_coef_remember_df.loc[3,'beta']=replace_early_lr.coef_[0][0]
    subject_coef_remember_df.loc[3,'timing']='early'
    replace_late_lr = LinearRegression().fit(replace_late_operation_remember_evi,replace_remember_category_evi)
    subject_coef_remember_df.loc[4,'beta']=replace_late_lr.coef_[0][0]
    subject_coef_remember_df.loc[4,'timing']='late'
    replace_overall_lr = LinearRegression().fit(replace_overall_operation_remember_evi,replace_remember_category_evi)
    subject_coef_remember_df.loc[5,'beta']=replace_overall_lr.coef_[0][0]
    subject_coef_remember_df.loc[5,'timing']='overall'    

    suppress_early_lr = LinearRegression().fit(suppress_early_operation_remember_evi,suppress_remember_category_evi)
    subject_coef_remember_df.loc[6,'beta']=suppress_early_lr.coef_[0][0]
    subject_coef_remember_df.loc[6,'timing']='early'
    suppress_late_lr = LinearRegression().fit(suppress_late_operation_remember_evi,suppress_remember_category_evi)
    subject_coef_remember_df.loc[7,'beta']=suppress_late_lr.coef_[0][0]
    subject_coef_remember_df.loc[7,'timing']='late'
    suppress_overall_lr = LinearRegression().fit(suppress_overall_operation_remember_evi,suppress_remember_category_evi)
    subject_coef_remember_df.loc[8,'beta']=suppress_overall_lr.coef_[0][0]
    subject_coef_remember_df.loc[8,'timing']='overall'

    ## Forgot
    #some subjects did not forget items, so need to have contingency code:
    try:
        maintain_early_lr = LinearRegression().fit(maintain_early_operation_forgot_evi,maintain_forgot_category_evi)
        subject_coef_forgot_df.loc[0,'beta']=maintain_early_lr.coef_[0][0]
        subject_coef_forgot_df.loc[0,'timing']='early'
        maintain_late_lr = LinearRegression().fit(maintain_late_operation_forgot_evi,maintain_forgot_category_evi)
        subject_coef_forgot_df.loc[1,'beta']=maintain_late_lr.coef_[0][0]
        subject_coef_forgot_df.loc[1,'timing']='late'
        maintain_overall_lr = LinearRegression().fit(maintain_overall_operation_forgot_evi,maintain_forgot_category_evi)
        subject_coef_forgot_df.loc[2,'beta']=maintain_overall_lr.coef_[0][0]
        subject_coef_forgot_df.loc[2,'timing']='overall'    
    except:
        print('No maintain trials forgotten!')
        subject_coef_forgot_df.loc[0,'beta']=0
        subject_coef_forgot_df.loc[0,'timing']='early'
        subject_coef_forgot_df.loc[1,'beta']=0
        subject_coef_forgot_df.loc[1,'timing']='late'        
        subject_coef_forgot_df.loc[2,'beta']=0
        subject_coef_forgot_df.loc[2,'timing']='overall'

    try:
        replace_early_lr = LinearRegression().fit(replace_early_operation_forgot_evi,replace_forgot_category_evi)
        subject_coef_forgot_df.loc[3,'beta']=replace_early_lr.coef_[0][0]
        subject_coef_forgot_df.loc[3,'timing']='early'
        replace_late_lr = LinearRegression().fit(replace_late_operation_forgot_evi,replace_forgot_category_evi)
        subject_coef_forgot_df.loc[4,'beta']=replace_late_lr.coef_[0][0]
        subject_coef_forgot_df.loc[4,'timing']='late'
        replace_overall_lr = LinearRegression().fit(replace_overall_operation_forgot_evi,replace_forgot_category_evi)
        subject_coef_forgot_df.loc[5,'beta']=replace_overall_lr.coef_[0][0]
        subject_coef_forgot_df.loc[5,'timing']='overall'    
    except:
        print('No replace trials forgotten!')        
        subject_coef_forgot_df.loc[3,'beta']=0
        subject_coef_forgot_df.loc[3,'timing']='early'
        subject_coef_forgot_df.loc[4,'beta']=0
        subject_coef_forgot_df.loc[4,'timing']='late'
        subject_coef_forgot_df.loc[5,'beta']=0
        subject_coef_forgot_df.loc[5,'timing']='overall'   

    try:
        suppress_early_lr = LinearRegression().fit(suppress_early_operation_forgot_evi,suppress_forgot_category_evi)
        subject_coef_forgot_df.loc[6,'beta']=suppress_early_lr.coef_[0][0]
        subject_coef_forgot_df.loc[6,'timing']='early'
        suppress_late_lr = LinearRegression().fit(suppress_late_operation_forgot_evi,suppress_forgot_category_evi)
        subject_coef_forgot_df.loc[7,'beta']=suppress_late_lr.coef_[0][0]
        subject_coef_forgot_df.loc[7,'timing']='late'
        suppress_overall_lr = LinearRegression().fit(suppress_overall_operation_forgot_evi,suppress_forgot_category_evi)
        subject_coef_forgot_df.loc[8,'beta']=suppress_overall_lr.coef_[0][0]
        subject_coef_forgot_df.loc[8,'timing']='overall'    
    except:
        print('No suppress trials forgotten!')                
        subject_coef_forgot_df.loc[6,'beta']=0
        subject_coef_forgot_df.loc[6,'timing']='early'
        subject_coef_forgot_df.loc[7,'beta']=0
        subject_coef_forgot_df.loc[7,'timing']='late'
        subject_coef_forgot_df.loc[8,'beta']=0
        subject_coef_forgot_df.loc[8,'timing']='overall'  

    # save for future use
    if save: 
        sub_dir = os.path.join(data_dir, f"sub-{subID}")
        out_fname_template = f"sub-{subID}_{space}_{task2}_coef_remember_dataframe.csv"  
        print(f"\n Saving the remembered beta's from Operation predicting Content for {subID} - space: {space} - as {out_fname_template}")
        subject_coef_remember_df.to_csv(os.path.join(sub_dir,out_fname_template))      

        out_fname_template = f"sub-{subID}_{space}_{task2}_coef_forgot_dataframe.csv"  
        print(f"\n Saving the forgotten beta's from Operation predicting Content for {subID} - space: {space} - as {out_fname_template}")
        subject_coef_forgot_df.to_csv(os.path.join(sub_dir,out_fname_template))            
    return subject_coef_remember_df, subject_coef_forgot_df

def visualize_coef_dfs():
    group_coef_df=pd.DataFrame()
    for subID in subIDs:
        temp_df, _=coef_stim_operation(subID)
        group_coef_df=pd.concat([group_coef_df,temp_df],ignore_index=True, sort=False)

    ax=sns.barplot(data=group_coef_df,x='condition',y='beta',hue='timing')
    plt.legend()
    ax.set(xlabel='Operation on Item', ylabel='Beta', title=f'{space} - Operation prediction on Content Decoding')
    ax.set_ylim([-0.3,0.3])
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs',f'group_level_{space}_operation_category_prediction.png'))
    plt.clf()  

def bootstrap_auc():
    group_auc_df=pd.DataFrame()
    for subID in subIDs:
        _, temp_df=coef_stim_operation(subID)
        group_auc_df=pd.concat([group_auc_df,temp_df], sort=False)

    maintain_df=group_auc_df.loc['Maintain'][['AUC','Content']]
    replace_df=group_auc_df.loc['Replace'][['AUC','Content']]
    suppress_df=group_auc_df.loc['Suppress'][['AUC','Content']]

    maintain_bootstrap=[]
    replace_bootstrap=[]
    suppress_bootstrap=[]

    for i in range(0,1000):
        maintain_itr=resample(maintain_df.values)
        replace_itr=resample(replace_df.values)
        suppress_itr=resample(suppress_df.values)

        maintain_itr_auc=maintain_itr[:,0].reshape(-1,1)
        maintain_itr_content=maintain_itr[:,1].reshape(-1,1)

        replace_itr_auc=replace_itr[:,0].reshape(-1,1)
        replace_itr_content=replace_itr[:,1].reshape(-1,1)

        suppress_itr_auc=suppress_itr[:,0].reshape(-1,1)
        suppress_itr_content=suppress_itr[:,1].reshape(-1,1)

        maintain_bootstrap=np.append(maintain_bootstrap,LinearRegression().fit(maintain_itr_auc,maintain_itr_content).coef_[0][0])
        replace_bootstrap=np.append(replace_bootstrap,LinearRegression().fit(replace_itr_auc,replace_itr_content).coef_[0][0])
        suppress_bootstrap=np.append(suppress_bootstrap,LinearRegression().fit(suppress_itr_auc,suppress_itr_content).coef_[0][0])

    true_maintain_beta=LinearRegression().fit(maintain_df['AUC'].values.reshape(-1,1),maintain_df['Content'].values.reshape(-1,1)).coef_[0][0]
    true_replace_beta=LinearRegression().fit(replace_df['AUC'].values.reshape(-1,1),replace_df['Content'].values.reshape(-1,1)).coef_[0][0]
    true_suppress_beta=LinearRegression().fit(suppress_df['AUC'].values.reshape(-1,1),suppress_df['Content'].values.reshape(-1,1)).coef_[0][0]

    ax=sns.histplot(maintain_bootstrap)
    ax.axvline(np.percentile(maintain_bootstrap,97.5),0,ax.get_ylim()[1],color='k', linestyle='dashed', label='97.5% Boundary')
    ax.set(xlabel='AUC vs. Content Betas', ylabel='Frequency', title=f'Super Subject - Maintian AUC predicting Content Decoding')
    ax.axvline(true_maintain_beta,0,ax.get_ylim()[1],color='g', label='True Maintain Beta')    
    ax.legend()    
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs','Super_subject_maintain_auc_content_prediction.png'))
    plt.clf()

    ax=sns.histplot(replace_bootstrap)
    ax.axvline(np.percentile(replace_bootstrap,97.5),0,ax.get_ylim()[1],color='k', linestyle='dashed', label='97.5% Boundary')
    ax.set(xlabel='AUC vs. Content Betas', ylabel='Frequency', title=f'Super Subject - Replace AUC predicting Content Decoding')
    ax.axvline(true_suppress_beta,0,ax.get_ylim()[1],color='b', label='True Replace Beta')    
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs','Super_subject_replace_auc_content_prediction.png'))
    plt.clf()

    ax=sns.histplot(suppress_bootstrap)
    ax.axvline(np.percentile(suppress_bootstrap,97.5),0,ax.get_ylim()[1],color='k', linestyle='dashed', label='97.5% Boundary')
    ax.axvline(true_suppress_beta,0,ax.get_ylim()[1],color='r', label='True Suppress Beta')
    ax.set(xlabel='AUC vs. Content Betas', ylabel='Frequency', title=f'Super Subject - Suppress AUC predicting Content Decoding')
    ax.legend()    
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs','Super_subject_suppress_auc_content_prediction.png'))
    plt.clf()    


def visualize_coef_dfs_memory():
    group_coef_remember_df=pd.DataFrame()
    group_coef_forgot_df=pd.DataFrame()
    for subID in subIDs:
        temp_r_df, temp_f_df=coef_stim_memory_operation(subID)
        group_coef_remember_df=pd.concat([group_coef_remember_df,temp_r_df],ignore_index=True, sort=False)
        group_coef_forgot_df=pd.concat([group_coef_forgot_df,temp_f_df],ignore_index=True, sort=False)

    ax=sns.barplot(data=group_coef_remember_df,x='condition',y='beta',hue='timing')
    plt.legend()
    ax.set(xlabel='Operation on Item', ylabel='Beta', title=f'{space} - Operation prediction on Remembered Content Decoding')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs',f'group_level_{space}_operation_remembered_category_prediction.png'))
    plt.clf()  

    ax=sns.barplot(data=group_coef_forgot_df,x='condition',y='beta',hue='timing')
    plt.legend()
    ax.set(xlabel='Operation on Item', ylabel='Beta', title=f'{space} - Operation prediction on Forgotten Content Decoding')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir,'figs',f'group_level_{space}_operation_forgotten_category_prediction.png'))
    plt.clf()  
