#Operation Classifier during Study phase - Maintain +/- & Suppress +/-
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
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_curve, auc
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, PredefinedSplit, StratifiedKFold  #train_test_split, PredefinedSplit, cross_validate, cross_val_predict, 
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


#local variant:
# workspace = 'local'
# data_dir = '/Volumes/zbg_eHD/Zachary_Data/repclearbids/derivatives/fmriprep/'
# param_dir = '/Users/zb3663/Desktop/School_Files/Repclear_files/repclear_preprocessed/params/'

#function to load in the confounds file for each run and then select the columns we want for cleaning
def confound_cleaner(confounds):
    COI = ['a_comp_cor_00','a_comp_cor_01','a_comp_cor_02','a_comp_cor_03','a_comp_cor_04','a_comp_cor_05','framewise_displacement','trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']
    for _c in confounds.columns:
        if 'cosine' in _c:
            COI.append(_c)
    confounds = confounds[COI]
    confounds.loc[0,'framewise_displacement'] = confounds.loc[1:,'framewise_displacement'].mean()
    return confounds  

#function to find the intersection between two lists
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

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
    print("\n*** Masking & cleaning bold data...")
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

    #local version
    # subject_design_dir='/Volumes/zbg_eHD/Zachary_Data/repclearbids/derivatives/fmriprep/subject_designs/'


    #using the task tag, we want to get the proper tag to pull the subject and phase specific dataframes
    if task=='preremoval': temp_task='pre-localizer'
    if task=='postremoval': temp_task='post-localizer'
    if task=='study': temp_task='study'

    sub_design=(f"*{subID}*{temp_task}*tr*")
    sub_design_file=find(sub_design,subject_design_dir)
    sub_design_matrix = pd.read_csv(sub_design_file[0]) #this is the correct, TR by TR list of what happened during this subject's study phase

    #now need to pull in the memory results:
    sub_memory=(f'memory_and_familiar*{subID}*')
    sub_memory_file=find(sub_memory,subject_design_dir)
    sub_memory_matrix= pd.read_csv(sub_memory_file[0])

    shifted_df = shift_timing(sub_design_matrix, shift_size_TR, rest_tag)
    shifted_df.reset_index(drop=True,inplace=True) #reset the index to use in the following loop

    #this loop now goes through each index of the dataframe with our labels, and uses the memory file of that subject to insert the results for each image
    #this will then allow us to pull out the remembered and forgotten versions of the operations as a label
    for i in range(len(shifted_df)):
        temp_image_id=shifted_df['image_id'][i]
        if temp_image_id==0:
            continue
        else:
            temp_memory=sub_memory_matrix[sub_memory_matrix['image_num']==temp_image_id]['memory'].values[0]
        shifted_df.loc[shifted_df['image_id']==temp_image_id,'accuracy']=temp_memory

    return shifted_df 

def fit_model(X, Y, runs, save=False, out_fname=None, v=False, balance=False, under_sample=False):
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



    # ps = PredefinedSplit(runs)
    skf = StratifiedKFold(n_splits=3)
    for train_inds, test_inds in skf.split(X, Y):

        X_train, X_test, y_train, y_test = X[train_inds], X[test_inds], Y[train_inds], Y[test_inds]
        #using random under sampling here to reduce learning set:
        if under_sample:
            print('*** Random Under Sampling of unbalanced classes ***')
            rus = RandomUnderSampler(random_state=42, sampling_strategy='auto')
            X_res, y_res = rus.fit_resample(X_train, y_train)

            # feature selection and transformation
            ffpr = SelectFpr(f_classif, alpha=0.01).fit(X_res, y_res)
            X_train_sub = ffpr.transform(X_res)
            X_test_sub = ffpr.transform(X_test)
        else:
            # feature selection and transformation
            ffpr = SelectFpr(f_classif, alpha=0.01).fit(X_train, y_train)
            X_train_sub = ffpr.transform(X_train)
            X_test_sub = ffpr.transform(X_test)

        # train & hyperparam tuning
        parameters ={'C':[.001, .01, .1, 1, 10, 100, 1000]}
        if balance:
            gscv = GridSearchCV(
                LogisticRegression(penalty='l2', solver='lbfgs',max_iter=1000,class_weight='balanced'),
                parameters,
                return_train_score=True)
        else:
            gscv = GridSearchCV(
                LogisticRegression(penalty='l2', solver='lbfgs',max_iter=1000),
                parameters,
                return_train_score=True)
        if under_sample:            
            gscv.fit(X_train_sub, y_res)
            best_Cs.append(gscv.best_params_['C'])
        else:
            gscv.fit(X_train_sub, y_train)
            best_Cs.append(gscv.best_params_['C'])            

        # refit with full data and optimal penalty value
        if balance:
            print('balancing classes via class-weights')
            lr = LogisticRegression(penalty='l2', solver='lbfgs', C=best_Cs[-1],max_iter=1000, class_weight='balanced')
        else:
            lr = LogisticRegression(penalty='l2', solver='lbfgs', C=best_Cs[-1],max_iter=1000)            
        
        if under_sample:
            print('*** Fitting full model ***')
            lr.fit(X_train_sub, y_res)
        else:
            print('*** Fitting full model ***')            
            lr.fit(X_train_sub, y_train)

        # test on held out data
        print('*** Testing on held out data and scoring results ***')
        score = lr.score(X_test_sub, y_test)

        y_score = lr.decision_function(X_test_sub)
        n_classes=np.unique(y_test)
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        counter=0
        for i in n_classes:
            temp_y=np.zeros(y_test.size)
            label_ind=np.where(y_test==(i))
            temp_y[label_ind]=1


            fpr[counter], tpr[counter], _ = roc_curve(temp_y, y_score[:, counter])
            roc_auc[counter] = auc(fpr[counter], tpr[counter])
            counter=counter+1

        auc_score = roc_auc_score(y_test, lr.predict_proba(X_test_sub), multi_class='ovr')
        preds = lr.predict(X_test_sub)
        pred_prob=lr.predict_proba(X_test_sub)
        # confusion matrix
        true_counts = np.asarray([np.sum(y_test == i) for i in ['maintain_r','maintain_f','suppress_r','suppress_f']])
        cm = confusion_matrix(y_test, preds, labels=['maintain_r','maintain_f','suppress_r','suppress_f']) / true_counts[:,None] * 100

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

    #labels we want: Maintain_remember(0), Maintain_forget(1), Suppress_remember(2), Suppress_forget(3)
    print("\n***** Subsampling data points by runs...")

    category_list = label_df['condition']
    accuracy_list = label_df['accuracy']
    stim_on = label_df['stim_present']
    run_list = label_df['run']
    image_list = label_df['image_id']

    # get faces
    oper_inds = np.where((stim_on == 2) | (stim_on == 3))[0]

    remember_inds = np.where((accuracy_list == 1))[0]
    forgot_inds = np.where((accuracy_list == 0))[0]

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


    operation_reg=[]
    for i in oper_inds:
        if (category_list.values[i]==1) & (accuracy_list.values[i]==1):
            operation_reg.append('maintain_r') #Maintain-Remembered label

        elif (category_list.values[i]==1) & (accuracy_list.values[i]==0):
            operation_reg.append('maintain_f') #Maintain-Forgot label

        elif (category_list.values[i]==3) & (accuracy_list.values[i]==1):
            operation_reg.append('suppress_r') #Suppress-Remembered label

        elif (category_list.values[i]==3) & (accuracy_list.values[i]==0):
            operation_reg.append('suppress_f') #Suppress-Forgot label

        elif (category_list.values[i]==2):
        #for now we are ignoring Replace, so this all gets a label of replace, which we can then remove:
            operation_reg.append('replace') #this means that later we will want to remove all replace's from the data to only look at maintain / suppress        

    # operation_reg=category_list.values[oper_inds]
    run_reg=run_list.values[oper_inds]
    image_reg = image_list.values[oper_inds]

    # === get sample_bold & sample_regressor
    sample_bold = []
    sample_runs = run_reg

    sample_bold = full_data[oper_inds]

    #add in code to handle new labeling:
    operation_reg=np.asarray(operation_reg)
    sample_regressor = operation_reg

    #since these are fed right into the classifier, I likely will need to trim out replace conditions here
    replace_inds=np.where(operation_reg=='replace')
    #now take these indicies and then delete out from the arrays
    sample_bold=np.delete(sample_bold,replace_inds,0)
    sample_regressor=np.delete(sample_regressor,replace_inds)
    sample_runs=np.delete(sample_runs,replace_inds)
    image_reg=np.delete(image_reg,replace_inds)


    return sample_bold, sample_regressor, sample_runs, image_reg

def classification(subID):
    task = 'study'
    space = 'MNI' #T1w
    ROIs = ['wholebrain']
    n_iters = 1


    shift_size_TR = shift_sizes_TR[0]
    rest_tag = 0

    print(f"\n***** Running operation classification for sub {subID} {task} {space} with ROIs {ROIs}...")

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
    if sum(Y=='maintain_f')<11:
        (print('Subject does not have enough Forgotten Maintain Trials'))
    print(f"Running model fitting and cross-validation...")

    # model fitting 
    #score, auc_score, cm, evidence, roc_auc, tested_labels, y_scores = fit_model(X, Y, runs, save=False, v=True, balance=True, under_sample=False)

    #under_sample variant
    score, auc_score, cm, evidence, roc_auc, tested_labels, y_scores = fit_model(X, Y, runs, save=False, v=True, balance=False, under_sample=True)

    mean_score=score.mean()

    print(f"\n***** Average results of Operation Success Classification for sub {subID} - {task} - {space}: Score={mean_score} ")

    #want to save the AUC results in such a way that I can also add in the content average later:
    auc_df=pd.DataFrame(columns=['AUC','Content','Sub'],index=['Maintain_R','Maintain_F','Suppress_R','Suppress_F'])
    auc_df.loc['Maintain_R']['AUC']=roc_auc.loc[:,0].mean() #Because the above script calculates these based on a leave-one-run-out. We will have an AUC for Maintain, Replace and Suppress per iteration (3 total). So taking the mean of each operation
    auc_df.loc['Maintain_F']['AUC']=roc_auc.loc[:,1].mean()
    auc_df.loc['Suppress_R']['AUC']=roc_auc.loc[:,2].mean()
    auc_df.loc['Suppress_F']['AUC']=roc_auc.loc[:,3].mean()    
    auc_df['Sub']=subID

    sub_dir = os.path.join(data_dir, f"sub-{subID}")
    out_fname_template_auc = f"sub-{subID}_{space}_{task}_operation_memoryoutcome_auc.csv"            
    print("\n *** Saving AUC values with subject dataframe ***")
    auc_df.to_csv(os.path.join(sub_dir,out_fname_template_auc))    

    #need to then save the evidence:
    evidence_df=pd.DataFrame(columns=['runs','operation','image_id']) #take the study DF for this subject
    evidence_df['runs']=runs
    evidence_df['operation']=Y
    evidence_df['image_id']=imgs
    evidence_df['maintain_R_evi']=np.vstack(evidence)[:,0] #add in the evidence values for maintain_R
    evidence_df['maintain_F_evi']=np.vstack(evidence)[:,1] #add in the evidence values for maintain_F
    evidence_df['suppress_R_evi']=np.vstack(evidence)[:,2] #add in the evidence values for suppress_R
    evidence_df['suppress_F_evi']=np.vstack(evidence)[:,3] #add in the evidence values for suppress_F


    #this will export the subject level evidence to the subject's folder
    sub_dir = os.path.join(data_dir, f"sub-{subID}")
    out_fname_template = f"sub-{subID}_{space}_{task}_operation_memoryoutcome_evidence.csv"            
    print("\n *** Saving evidence values with subject dataframe ***")
    evidence_df.to_csv(os.path.join(sub_dir,out_fname_template)) 


    sub_dir = os.path.join(data_dir, f"sub-{subID}")
    out_fname_template_cm = f"sub-{subID}_{space}_{task}_operation_memoryoutcome_cm.csv"            
    print("\n *** Saving confusion matrix with subject dataframe ***")
    cm_df=pd.DataFrame(data=cm.mean(axis=0))
    cm_df.to_csv(os.path.join(sub_dir,out_fname_template_cm))
    
def binary_classification(subID,condition):
    task = 'study'
    space = 'MNI' #T1w
    ROIs = ['wholebrain']
    n_iters = 1

    shift_size_TR = shift_sizes_TR[0]
    rest_tag = 0

    print(f"\n***** Running {condition} classification for sub {subID} {task} {space} with ROIs {ROIs}...")

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

    X, Y, runs, imgs = sample_for_binarytraining(full_data, label_df, condition)

    print(f"Running model fitting and cross-validation...")

    #under_sample variant
    scores, cms, evidences, roc_aucs, tested_labels, y_scores = fit_binary_model(X, Y, runs, save=False, v=True, balance=False, under_sample=True)

    mean_score=scores.mean()

    print(f"\n***** Average results of Operation Success Classification for sub {subID} - {task} - {space}: Score={mean_score} ")

    #want to save the AUC results in such a way that I can also add in the content average later:
    auc_df=pd.DataFrame(columns=['AUC','Content','Sub'],index=[f'{condition}'])
    auc_df.loc[f'{condition}']['AUC']=roc_aucs['AUC'].mean()
    auc_df['Sub']=subID

    sub_dir = os.path.join(data_dir, f"sub-{subID}")
    out_fname_template_auc = f"sub-{subID}_{space}_{task}_{condition}binary_memoryoutcome_auc.csv"            
    print("\n *** Saving AUC values with subject dataframe ***")
    auc_df.to_csv(os.path.join(sub_dir,out_fname_template_auc))    

    #need to then save the evidence:
    evidence_df=pd.DataFrame(columns=['runs','operation','image_id']) #take the study DF for this subject
    evidence_df['runs']=runs
    evidence_df['operation']=Y
    evidence_df['image_id']=imgs
    evidence_df[f'{condition}_R_evi']=np.vstack(evidences) #add in the evidence values for suppress_R
    evidence_df[f'{condition}_F_evi']=np.vstack(1-evidences) #add in the evidence values for suppress_F


    #this will export the subject level evidence to the subject's folder
    sub_dir = os.path.join(data_dir, f"sub-{subID}")
    out_fname_template = f"sub-{subID}_{space}_{task}_{condition}binary_memoryoutcome_evidence.csv"            
    print("\n *** Saving evidence values with subject dataframe ***")
    evidence_df.to_csv(os.path.join(sub_dir,out_fname_template))        

def sample_for_binarytraining(full_data, label_df, condition, include_rest=False):
    """
    sample data by runs. 
    Return: sampled labels and bold data
    """ 

    #sampling the remembered and forgotten for a given condition 
    print("\n***** Subsampling data points by runs...")

    category_list = label_df['condition']
    accuracy_list = label_df['accuracy']
    stim_on = label_df['stim_present']
    run_list = label_df['run']
    image_list = label_df['image_id']

    # get faces
    oper_inds = np.where((stim_on == 2) | (stim_on == 3))[0]

    remember_inds = np.where((accuracy_list == 1))[0]
    forgot_inds = np.where((accuracy_list == 0))[0]

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

    operation_reg=[]
    keep_inds=[]
    for i in oper_inds:
        if condition == 'maintain':
            if (category_list.values[i]==1) & (accuracy_list.values[i]==1):
                operation_reg.append('maintain_r') #Maintain-Remembered label
                keep_inds.append(i)

            elif (category_list.values[i]==1) & (accuracy_list.values[i]==0):
                operation_reg.append('maintain_f') #Maintain-Forgot label
                keep_inds.append(i)

        elif condition == 'suppress':
            if (category_list.values[i]==3) & (accuracy_list.values[i]==1):
                operation_reg.append('suppress_r') #Suppress-Remembered label
                keep_inds.append(i)

            elif (category_list.values[i]==3) & (accuracy_list.values[i]==0):
                operation_reg.append('suppress_f') #Suppress-Forgot label
                keep_inds.append(i)

        elif condition == 'replace':
            if (category_list.values[i]==2) & (accuracy_list.values[i]==1):
                operation_reg.append('replace_r') #Suppress-Remembered label
                keep_inds.append(i)                

            elif (category_list.values[i]==2) & (accuracy_list.values[i]==0):
                operation_reg.append('replace_f') #Suppress-Forgot label
                keep_inds.append(i)

    # operation_reg=category_list.values[oper_inds]
    run_reg=run_list.values[keep_inds]
    image_reg = image_list.values[keep_inds]

    # === get sample_bold & sample_regressor
    sample_bold = []
    sample_runs = run_reg

    sample_bold = full_data[keep_inds]

    #add in code to handle new labeling:
    operation_reg=np.asarray(operation_reg)
    sample_regressor = operation_reg

    return sample_bold, sample_regressor, sample_runs, image_reg

def fit_binary_model(X, Y, runs, save=False, out_fname=None, v=False, balance=False, under_sample=False):
    if v: print("\n***** Fitting model...")

    scores = []
    auc_scores = []
    cms = []
    best_Cs = []
    evidences = []
    roc_aucs=[]
    roc_dict={k:[] for k in ['train', 'test', 'AUC']}
    tested_labels=[]
    pred_probs=[]
    y_scores=[]

    counter=0
    # ps = PredefinedSplit(runs)
    skf = StratifiedKFold(n_splits=3)
    for train_inds, test_inds in skf.split(X, Y):

        X_train, X_test, y_train, y_test = X[train_inds], X[test_inds], Y[train_inds], Y[test_inds]
        #using random under sampling here to reduce learning set:
        if under_sample:
            print('*** Random Under Sampling of unbalanced classes ***')
            rus = RandomUnderSampler(random_state=42, sampling_strategy='auto')
            X_res, y_res = rus.fit_resample(X_train, y_train)

            # feature selection and transformation
            ffpr = SelectFpr(f_classif, alpha=0.01).fit(X_res, y_res)
            X_train_sub = ffpr.transform(X_res)
            X_test_sub = ffpr.transform(X_test)
        else:
            # feature selection and transformation
            ffpr = SelectFpr(f_classif, alpha=0.01).fit(X_train, y_train)
            X_train_sub = ffpr.transform(X_train)
            X_test_sub = ffpr.transform(X_test)

        # train & hyperparam tuning
        parameters ={'C':[.001, .01, .1, 1, 10, 100, 1000]}
        if balance:
            gscv = GridSearchCV(
                LinearSVC(penalty='l2',max_iter=1000,class_weight='balanced'),
                parameters,
                return_train_score=True)
        else:
            gscv = GridSearchCV(
                LinearSVC(penalty='l2',max_iter=1000),
                parameters,
                return_train_score=True)
        if under_sample:            
            gscv.fit(X_train_sub, y_res)
            best_Cs.append(gscv.best_params_['C'])
        else:
            gscv.fit(X_train_sub, y_train)
            best_Cs.append(gscv.best_params_['C'])            

        # refit with full data and optimal penalty value
        if balance:
            print('balancing classes via class-weights')
            lr = LinearSVC(penalty='l2', C=best_Cs[-1],max_iter=1000, class_weight='balanced')
        else:
            lr = LinearSVC(penalty='l2', C=best_Cs[-1],max_iter=1000)            
        
        if under_sample:
            print('*** Fitting full model ***')
            lr.fit(X_train_sub, y_res)
        else:
            print('*** Fitting full model ***')            
            lr.fit(X_train_sub, y_train)

        # test on held out data
        print('*** Testing on held out data and scoring results ***')
        score = lr.score(X_test_sub, y_test)

        y_score = lr.decision_function(X_test_sub)
        n_classes=np.unique(y_test)


        y_train_pred = lr.decision_function(X_train_sub)    
        y_test_pred = lr.decision_function(X_test_sub)
        train_fpr, train_tpr, tr_thresholds = roc_curve(y_res, y_train_pred,pos_label=f'{condition}_r')
        test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred,pos_label=f'{condition}_r')
            
        roc_dict['train'].append([train_fpr,train_tpr])
        roc_dict['test'].append([test_fpr,test_tpr])
        roc_dict['AUC'].append(auc(test_fpr, test_tpr))

        #auc_score = roc_auc_score(y_test, lr.predict_proba(X_test_sub), multi_class='ovr')
        preds = lr.predict(X_test_sub)
        # confusion matrix
        ## add in variable code to change these strings based on condition input
        true_counts = np.asarray([np.sum(y_test == i) for i in [f'{condition}_r',f'{condition}_f']])
        cm = confusion_matrix(y_test, preds, labels=[f'{condition}_r',f'{condition}_f']) / true_counts[:,None] * 100

        scores.append(score)
        cms.append(cm)
        #calculate evidence values
        evidence=(1. / (1. + np.exp(-lr.decision_function(X_test_sub))))
        evidences.append(evidence) 

        tested_labels.append(y_test)
        y_scores.append(y_score)

    roc_aucs = pd.DataFrame(data=roc_dict)
    scores = np.asarray(scores)
    cms = np.stack(cms)
    evidences = np.concatenate(evidences)

    tested_labels=np.stack(tested_labels)
    y_scores=np.stack(y_scores)

    if v: print(f"\nClassifier score: \n"
        f"scores: {scores.mean()} +/- {scores.std()}\n"
        f"auc scores: {roc_aucs['AUC'].mean()} +/- {roc_aucs['AUC'].std()}\n"
        f"best Cs: {best_Cs}\n"
        f"average confution matrix:\n"
        f"{cms.mean(axis=0)}")

    return scores, cms, evidences, roc_aucs, tested_labels, y_scores

def organize_cms(subID,space,task,save=True):
    ROIs = ['wholebrain']

    print( "\n *** loading in confusion matrix from subject data frame ***")
    sub_dir=os.path.join(data_dir,f"sub-{subID}")
    in_fname_template=f"sub-{subID}_{space}_{task}_operation_memoryoutcome_cm.csv"

    sub_df=pd.pd.read_csv(os.path.join(sub_dir,in_fname_template))  

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

