# modified operation data - between experiment classification


# Imports
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
from sklearn.model_selection import (
    PredefinedSplit,
    cross_validate,
    cross_val_predict,
    GridSearchCV,
)
from sklearn.feature_selection import (
    VarianceThreshold,
    f_classif,
    SelectKBest,
    SelectFpr,
)
from sklearn.preprocessing import StandardScaler
from nilearn.input_data import NiftiMasker, MultiNiftiMasker
from nilearn.image import clean_img
from nilearn.signal import clean
from scipy import stats
from sklearn import preprocessing
import time
from sklearn.metrics import roc_auc_score


subs_rep = ["02", "03", "04"]
subs_clear = ["61", "69", "77"]

TR_shift = [5]
TR_shift_clear = [10]
brain_flag = "MNI"

masks = ["GM_group"]
clear_data = 1  # 0 off / 1 on


def confound_cleaner(confounds):
    COI = [
        "a_comp_cor_00",
        "framewise_displacement",
        "trans_x",
        "trans_y",
        "trans_z",
        "rot_x",
        "rot_y",
        "rot_z",
    ]
    for _c in confounds.columns:
        if "cosine" in _c:
            COI.append(_c)
    confounds = confounds[COI]
    confounds.loc[0, "framewise_displacement"] = confounds.loc[
        1:, "framewise_displacement"
    ].mean()
    return confounds


for num in range(len(subs_rep)):
    start_time = time.time()
    sub_num = subs[num]

    print("Running sub-0%s..." % sub_num)
    # define the subject
    sub = "sub-0%s" % sub_num
    container_path = (
        "/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep"
    )

    bold_path = os.path.join(container_path, sub, "func/")
    os.chdir(bold_path)

    # set up the path to the files and then moved into that directory

    # find the proper nii.gz files
    def find(pattern, path):  # find the pattern we're looking for
        result = []
        for root, dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root, name))
            return result

    bold_files = find("*study*bold*.nii.gz", bold_path)
    wholebrain_mask_path = find("*study*mask*.nii.gz", bold_path)
    anat_path = os.path.join(container_path, sub, "anat/")
    gm_mask_path = find("*MNI_GM_mask*", container_path)
    gm_mask = nib.load(gm_mask_path[0])

    if brain_flag == "MNI":
        pattern = "*MNI*"
        pattern2 = "*MNI152NLin2009cAsym*preproc*"
        brain_mask_path = fnmatch.filter(wholebrain_mask_path, pattern)
        study_files = fnmatch.filter(bold_files, pattern2)
    elif brain_flag == "T1w":
        pattern = "*T1w*"
        pattern2 = "*T1w*preproc*"
        brain_mask_path = fnmatch.filter(wholebrain_mask_path, pattern)
        study_files = fnmatch.filter(bold_files, pattern2)

    brain_mask_path.sort()
    study_files.sort()

    if clear_data == 0:
        if os.path.exists(
            os.path.join(
                bold_path,
                "sub-0%s_%s_study_%s_masked.npy" % (sub_num, brain_flag, mask_flag),
            )
        ):
            study_bold = np.load(
                os.path.join(
                    bold_path,
                    "sub-0%s_%s_study_%s_masked.npy" % (sub_num, brain_flag, mask_flag),
                )
            )
            print("%s %s study Loaded..." % (brain_flag, mask_flag))
            run1_length = int((len(study_bold) / 3))
            run2_length = int((len(study_bold) / 3))
            run3_length = int((len(study_bold) / 3))
        else:
            # select the specific file
            study_run1 = nib.load(study_files[0])
            study_run2 = nib.load(study_files[1])
            study_run3 = nib.load(study_files[2])

            # to be used to filter the data
            # First we are removing the confounds
            # get all the folders within the bold path
            study_confounds_1 = find("*study*1*confounds*.tsv", bold_path)
            study_confounds_2 = find("*study*2*confounds*.tsv", bold_path)
            study_confounds_3 = find("*study*3*confounds*.tsv", bold_path)

            confound_run1 = pd.read_csv(study_confounds_1[0], sep="\t")
            confound_run2 = pd.read_csv(study_confounds_2[0], sep="\t")
            confound_run3 = pd.read_csv(study_confounds_3[0], sep="\t")

            wholebrain_mask1 = nib.load(brain_mask_path[0])

            def apply_mask(mask=None, target=None):
                coor = np.where(mask == 1)
                values = target[coor]
                if values.ndim > 1:
                    values = np.transpose(values)  # swap axes to get feature X sample
                return values

            if mask_flag == "wholebrain":
                study_run1 = apply_mask(
                    mask=(wholebrain_mask1.get_data()), target=(study_run1.get_data())
                )
                study_run2 = apply_mask(
                    mask=(wholebrain_mask1.get_data()), target=(study_run2.get_data())
                )
                study_run3 = apply_mask(
                    mask=(wholebrain_mask1.get_data()), target=(study_run3.get_data())
                )

            elif mask_flag == "vtc":
                whole_study_run1 = apply_mask(
                    mask=(wholebrain_mask1.get_data()), target=(study_run1.get_data())
                )
                whole_study_run2 = apply_mask(
                    mask=(wholebrain_mask1.get_data()), target=(study_run2.get_data())
                )
                whole_study_run3 = apply_mask(
                    mask=(wholebrain_mask1.get_data()), target=(study_run3.get_data())
                )

                study_run1 = apply_mask(
                    mask=(vtc_mask.get_data()), target=(study_run1.get_data())
                )
                study_run2 = apply_mask(
                    mask=(vtc_mask.get_data()), target=(study_run2.get_data())
                )
                study_run3 = apply_mask(
                    mask=(vtc_mask.get_data()), target=(study_run3.get_data())
                )
            elif mask_flag == "GM_group":
                study_run1 = apply_mask(
                    mask=(gm_mask.get_data()), target=(study_run1.get_data())
                )
                study_run2 = apply_mask(
                    mask=(gm_mask.get_data()), target=(study_run2.get_data())
                )
                study_run3 = apply_mask(
                    mask=(gm_mask.get_data()), target=(study_run3.get_data())
                )

            preproc_1 = clean(study_run1, t_r=1, detrend=False, standardize="zscore")
            preproc_2 = clean(study_run2, t_r=1, detrend=False, standardize="zscore")
            preproc_3 = clean(study_run3, t_r=1, detrend=False, standardize="zscore")

            study_bold = np.concatenate((preproc_1, preproc_2, preproc_3))
            # save this data if we didn't have it saved before
            os.chdir(bold_path)
            np.save(
                "sub-0%s_%s_study_%s_masked" % (sub_num, brain_flag, mask_flag),
                study_bold,
            )
            print("%s %s masked data...saved" % (mask_flag, brain_flag))

            # create run array
            run1_length = int((len(study_run1)))
            run2_length = int((len(study_run2)))
            run3_length = int((len(study_run3)))
    else:
        if os.path.exists(
            os.path.join(
                bold_path,
                "sub-0%s_%s_study_%s_masked_cleaned.npy"
                % (sub_num, brain_flag, mask_flag),
            )
        ):
            study_bold = np.load(
                os.path.join(
                    bold_path,
                    "sub-0%s_%s_study_%s_masked_cleaned.npy"
                    % (sub_num, brain_flag, mask_flag),
                )
            )
            print("%s %s study Loaded..." % (brain_flag, mask_flag))
            run1_length = int((len(study_bold) / 3))
            run2_length = int((len(study_bold) / 3))
            run3_length = int((len(study_bold) / 3))
        else:
            # select the specific file
            study_run1 = nib.load(study_files[0])
            study_run2 = nib.load(study_files[1])
            study_run3 = nib.load(study_files[2])

            # to be used to filter the data
            # First we are removing the confounds
            # get all the folders within the bold path
            study_confounds_1 = find("*study*1*confounds*.tsv", bold_path)
            study_confounds_2 = find("*study*2*confounds*.tsv", bold_path)
            study_confounds_3 = find("*study*3*confounds*.tsv", bold_path)

            confound_run1 = pd.read_csv(study_confounds_1[0], sep="\t")
            confound_run2 = pd.read_csv(study_confounds_2[0], sep="\t")
            confound_run3 = pd.read_csv(study_confounds_3[0], sep="\t")

            confound_run1 = confound_cleaner(confound_run1)
            confound_run2 = confound_cleaner(confound_run2)
            confound_run3 = confound_cleaner(confound_run3)

            wholebrain_mask1 = nib.load(brain_mask_path[0])

            def apply_mask(mask=None, target=None):
                coor = np.where(mask == 1)
                values = target[coor]
                if values.ndim > 1:
                    values = np.transpose(values)  # swap axes to get feature X sample
                return values

            if mask_flag == "wholebrain":
                study_run1 = apply_mask(
                    mask=(wholebrain_mask1.get_data()), target=(study_run1.get_data())
                )
                study_run2 = apply_mask(
                    mask=(wholebrain_mask1.get_data()), target=(study_run2.get_data())
                )
                study_run3 = apply_mask(
                    mask=(wholebrain_mask1.get_data()), target=(study_run3.get_data())
                )

            elif mask_flag == "vtc":
                whole_study_run1 = apply_mask(
                    mask=(wholebrain_mask1.get_data()), target=(study_run1.get_data())
                )
                whole_study_run2 = apply_mask(
                    mask=(wholebrain_mask1.get_data()), target=(study_run2.get_data())
                )
                whole_study_run3 = apply_mask(
                    mask=(wholebrain_mask1.get_data()), target=(study_run3.get_data())
                )

                study_run1 = apply_mask(
                    mask=(vtc_mask.get_data()), target=(study_run1.get_data())
                )
                study_run2 = apply_mask(
                    mask=(vtc_mask.get_data()), target=(study_run2.get_data())
                )
                study_run3 = apply_mask(
                    mask=(vtc_mask.get_data()), target=(study_run3.get_data())
                )
            elif mask_flag == "GM_group":
                study_run1 = apply_mask(
                    mask=(gm_mask.get_data()), target=(study_run1.get_data())
                )
                study_run2 = apply_mask(
                    mask=(gm_mask.get_data()), target=(study_run2.get_data())
                )
                study_run3 = apply_mask(
                    mask=(gm_mask.get_data()), target=(study_run3.get_data())
                )

            preproc_1 = clean(
                study_run1,
                confounds=(confound_run1),
                t_r=1,
                detrend=False,
                standardize="zscore",
            )
            preproc_2 = clean(
                study_run2,
                confounds=(confound_run2),
                t_r=1,
                detrend=False,
                standardize="zscore",
            )
            preproc_3 = clean(
                study_run3,
                confounds=(confound_run3),
                t_r=1,
                detrend=False,
                standardize="zscore",
            )

            study_bold = np.concatenate((preproc_1, preproc_2, preproc_3))
            # save this data if we didn't have it saved before
            os.chdir(bold_path)
            np.save(
                "sub-0%s_%s_study_%s_masked_cleaned" % (sub_num, brain_flag, mask_flag),
                study_bold,
            )
            print("%s %s masked & cleaned data...saved" % (mask_flag, brain_flag))

            # create run array
            run1_length = int((len(study_run1)))
            run2_length = int((len(study_run2)))
            run3_length = int((len(study_run3)))
    if num == 0:
        repclear_bold_sub1 = study_bold
    elif num == 1:
        repclear_bold_sub2 = study_bold
    elif num == 2:
        repclear_bold_sub3 = study_bold

repclear_study_bold = np.concatenate(
    (repclear_bold_sub1, repclear_bold_sub2, repclear_bold_sub3)
)

# fill in the run array with run number
run1 = np.full(study_bold_sub1.shape[0], 1)
run2 = np.full(study_bold_sub2.shape[0], 2)
run3 = np.full(study_bold_sub3.shape[0], 3)

run_list = np.concatenate((run1, run2, run3))  # now combine

# load regs / labels
params_dir = "/scratch1/06873/zbretton/repclear_dataset/BIDS/params"
# find the mat file, want to change this to fit "sub"
param_search = "study*events*.csv"
param_file = find(param_search, params_dir)

# gotta index to 0 since it's a list
reg_matrix = pd.read_csv(param_file[0])
reg_operation = reg_matrix["condition"].values
reg_run = reg_matrix["run"].values
reg_present = reg_matrix["stim_present"].values

run1_index = np.where(reg_run == 1)
run2_index = np.where(reg_run == 2)
run3_index = np.where(reg_run == 3)

# need to convert this list to 1-d
# this list is now 1d list, need to add a dimentsionality to it
# Condition:
# 1. maintain
# 2. replace_category
# 3. suppress
stim_list = np.full(len(study_bold), 0)
maintain_list = np.where(
    (reg_operation == 1) & ((reg_present == 2) | (reg_present == 3))
)
suppress_list = np.where(
    (reg_operation == 3) & ((reg_present == 2) | (reg_present == 3))
)
replace_list = np.where(
    (reg_operation == 2) & ((reg_present == 2) | (reg_present == 3))
)
stim_list[maintain_list] = 1
stim_list[suppress_list] = 3
stim_list[replace_list] = 2

oper_list = reg_operation
oper_list = oper_list[:, None]

stim_list = stim_list[:, None]


# Create a function to shift the size, and will do the rest tag
def shift_timing(label_TR, TR_shift_size, tag):
    # Create a short vector of extra zeros or whatever the rest label is
    zero_shift = np.full(TR_shift_size, tag)
    # Zero pad the column from the top
    zero_shift = np.vstack(zero_shift)
    label_TR_shifted = np.vstack((zero_shift, label_TR))
    # Don't include the last rows that have been shifted out of the time line
    label_TR_shifted = label_TR_shifted[0 : label_TR.shape[0], 0]

    return label_TR_shifted


# Apply the function
shift_size = TR_shift  # this is shifting by 10TR
tag = 0  # rest label is 0
stim_list_shift = shift_timing(stim_list, shift_size, tag)  # rest is label 0

sub_stim_list = np.concatenate(
    (stim_list_shift, stim_list_shift, stim_list_shift), axis=None
)

import random


def Diff(li1, li2):
    return list(set(li1) - set(li2))


# tr_to_remove=Diff(rest_times,rest_times_index)

rest_times = np.where(sub_stim_list == 0)


# Extract bold data for non-zero labels
def reshape_data(label_TR_shifted, masked_data_all, run_list):
    label_index = np.nonzero(label_TR_shifted)
    label_index = np.squeeze(label_index)
    # Pull out the indexes
    indexed_data = masked_data_all[label_index, :]
    nonzero_labels = label_TR_shifted[label_index]
    nonzero_runs = run_list[label_index]
    return indexed_data, nonzero_labels, nonzero_runs


sub_stim_list_nr = np.delete(sub_stim_list, rest_times)
sub_stim_list_nr = sub_stim_list_nr.flatten()
sub_stim_list_nr = sub_stim_list_nr[:, None]
sub_study_bold_nr = np.delete(sub_study_bold, rest_times, axis=0)
run_list_nr = np.delete(run_list, rest_times)

# now we need to use this normalized data to run the decoder


# do a L2 estimator
# now the runs are actually the subjects (1-sub002,2-sub003,3-sub004)
# This should work properly where now the held out run, is a held out subject
# since the design is the same, the stim list is repeated 3 times
def L2_xval(data, labels, run_labels):
    scores = []
    predicts = []
    trues = []
    evidences = []
    chosenvoxels = []
    sig_scores = []
    ps = PredefinedSplit(run_labels)
    C_best = []
    for train_index, test_index in ps.split():
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # selectedvoxels=SelectKBest(f_classif,k=k_best).fit(X_train,y_train)
        selectedvoxels = SelectFpr(f_classif, alpha=0.05).fit(
            X_train, y_train
        )  # I compared this method to taking ALL k items in the F-test and filtering by p-value, so I assume this is a better feature selection method

        X_train = selectedvoxels.transform(X_train)
        X_test = selectedvoxels.transform(X_test)

        train_run_ids = run_labels[train_index]
        sp_train = PredefinedSplit(train_run_ids)
        parameters = {"C": [0.01, 0.1, 1, 10, 100, 1000]}
        inner_clf = GridSearchCV(
            LogisticRegression(penalty="l2", solver="liblinear"),
            parameters,
            cv=sp_train,
            return_train_score=True,
        )
        inner_clf.fit(X_train, y_train)
        C_best_i = inner_clf.best_params_["C"]
        C_best.append(C_best_i)

        # fit model
        L2 = LogisticRegression(penalty="l2", C=C_best_i, solver="liblinear")
        L2_out = L2.fit(X_train, y_train)
        # calculate evidence values
        evidence = 1.0 / (1.0 + np.exp(-L2_out.decision_function(X_test)))
        evidences.append(evidence)
        score = L2_out.score(X_test, y_test)  # score the model
        predict = L2_out.predict(X_test)  # what are the predicted classes
        predicts.append(predict)  # append the list over the x-val runs
        true = y_test  # what is the known labels
        scores.append(score)  # append the scores
        trues.append(true)  # append the known labels
        chosenvoxels.append(selectedvoxels.get_support())
        sig_score = roc_auc_score(true, L2_out.predict_proba(X_test), multi_class="ovr")
        sig_scores.append(sig_score)
    return L2_out, scores, predicts, trues, evidences, C_best, chosenvoxels, sig_scores


print("Now running classifier...")

(
    L2_models_nr,
    L2_scores_nr,
    L2_predicts_nr,
    L2_trues_nr,
    L2_evidence_nr,
    L2_costs_nr,
    L2_chosenvoxels_nr,
    L2_sig_scores_nr,
) = L2_xval(sub_study_bold_nr, sub_stim_list_nr, run_list_nr)
L2_subject_score_mean_nr = np.mean(L2_scores_nr)

print("Classifier done!")
print("Now running random Classifier to find chance level...")
# Need to run a randomized classifier to test for validity
n_iters = 1  # How many different permutations
sp = PredefinedSplit(run_list_nr)
clf_score = np.array([])
clf_score_oper = np.array([])

for i in range(n_iters):
    clf_score_i = np.array([])
    permuted_labels = np.random.permutation(sub_stim_list_nr)
    for train, test in sp.split():
        # Pull out the sample data
        train_data = sub_study_bold_nr[train, :]
        test_data = sub_study_bold_nr[test, :]

        # Do voxel selection on all voxels
        selected_voxels = SelectKBest(f_classif, k=2000).fit(
            train_data, sub_stim_list_nr[train]
        )

        # Train and test the classifier
        classifier = LinearSVC()
        clf = classifier.fit(
            selected_voxels.transform(train_data), permuted_labels[train]
        )
        score = clf.score(selected_voxels.transform(test_data), permuted_labels[test])
        clf_score_i = np.hstack((clf_score_i, score))
    clf_score = np.hstack((clf_score, clf_score_i.mean()))
# the output of this should line up to "Chance"

print("done!")
print("----------------------------------")
print("TR Shift: %s" % TR_shift)
# print('Random Stim-Label score: %s' % clf_score)
print("Random Operation Label score: %s" % clf_score)
# print('L2 score: %s - Cost: %s' % (L2_scores,L2_costs))
print(
    "L2 No Rest Score: %s - Cost: %s - ROC/AUC : %s"
    % (L2_scores_nr, L2_costs_nr, L2_sig_scores_nr)
)
# print('L2 Operation No Rest Score: %s - Cost: %s' % (L2_scores_operation_nr,L2_costs_operation_nr))
print("----------------------------------")
# need to save an output per subject here
print("saving data...")
output_table = {
    "random score": clf_score,
    "L2 Average Scores (No Rest)": L2_subject_score_mean_nr,
    "L2 Model (No Rest)": L2_models_nr,
    "L2 Raw Scores (No Rest)": L2_scores_nr,
    "L2 Predictions (No Rest)": L2_predicts_nr,
    "L2 True (No Rest)": L2_trues_nr,
    "L2 Costs (No Rest)": L2_costs_nr,
    "Class List (No Rest)": sub_stim_list_nr,
    "L2 Evidence (No Rest)": L2_evidence_nr,
    "Chosen voxels (No Rest)": L2_chosenvoxels_nr,
    "ROC/AUC": L2_sig_scores_nr,
}

import pickle

if clear_data == 0:
    os.chdir(os.path.join(container_path))
    f = open(
        "btwnsuboperation_%s_%s_%sTR lag_data.pkl" % (brain_flag, mask_flag, TR_shift),
        "wb",
    )
    pickle.dump(output_table, f)
    f.close()
else:
    os.chdir(os.path.join(container_path))
    f = open(
        "btwnsuboperation_%s_%s_%sTR lag_data_cleaned.pkl"
        % (brain_flag, mask_flag, TR_shift),
        "wb",
    )
    pickle.dump(output_table, f)
    f.close()
print("data saved!")
print("btwnsuboperation is now complete")
print("this took:")
print("--- %s seconds ---" % (time.time() - start_time))
