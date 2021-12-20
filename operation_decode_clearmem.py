import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nibabel as nib
from nilearn.image import clean_img
from nilearn.signal import clean

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.feature_selection import SelectFpr, f_classif
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

from cross_exp_classification import get_preprocessed_data, get_shifted_labels, subsample


# global consts
subIDs = ['002', '003', '004']
phases = ['rest', 'preremoval', 'study', 'postremoval']
runs = np.arange(6) + 1  
spaces = {'T1w': 'T1w', 
            'MNI': 'MNI152NLin2009cAsym'}
descs = ['brain_mask', 'preproc_bold']
ROIs = ['VVS', 'PHG', 'FG']
shift_sizes_TR = [5, 6]

stim_labels = {0: "Rest",
                1: "Scenes",
                2: "Faces"}

op_labels = {1: "maintain",
             2: "replace",
             3: "suppress"
            }

workspace = 'scratch'
if workspace == 'work':
    clearmem_dir = '/work/07365/sguo19/frontera/clearmem/'
    repclear_dir = '/work/07365/sguo19/frontera/fmriprep/'
    param_dir = '/work/07365/sguo19/frontera/params/'
    results_dir = '/work/07365/sguo19/model_fitting_results/'
elif workspace == 'scratch':
    clearmem_dir = '/scratch1/07365/sguo19/clearmem/'
    repclear_dir = '/scratch1/07365/sguo19/fmriprep/'
    param_dir = '/scratch1/07365/sguo19/params/'
    results_dir = '/scratch1/07365/sguo19/model_fitting_results/'



def clearmem_op_classification(subID="006"):
    # subs with study phase VTC mask: 006, 036, 077, 079
    shift_size_TR = 10

    task = "study"
    space = "T1w"
    rest_tag = 0
    include_iti = False
    v = True

    # ===== load sub data & labels
    mask_path = os.path.join(repclear_dir, "group_MNI_GM_mask.nii.gz")
    full_data = get_preprocessed_data("clearmem", subID, task, space, mask_path, runs=np.arange(6)+1, save=True)
    print("BOLD shape: ", full_data.shape)

    # label
    label_df = get_shifted_labels("clearmem", task, shift_size_TR, rest_tag)
    label_df["subID"] = np.array([subID for _ in range(len(label_df))])  # just to keep the subsample consistant
    print("Labels shape: ", full_data.shape)

    # subsample
    X, Y_df = subsample("clearmem", full_data, label_df, include_iti=include_iti)
    Y = Y_df["condition"]
    print(f"\nShape after subsample: X: {X.shape}, Y: {Y.shape}")

    # ===== train
    # performance eval 
    scores = []
    auc_scores = []
    cms = []
    models = []

    # decode results
    evids = []
    probs = []
    decode_dfs = []

    print("\nTraining classifier...")
    logo = LeaveOneGroupOut()
    for i, (train_inds, test_inds) in enumerate(logo.split(X, Y, groups=Y_df["run"])):
        X_train, X_test = X[train_inds], X[test_inds]
        y_train, y_test = Y[train_inds], Y[test_inds]
        train_groups = Y_df["run"][train_inds].values
        decode_df = Y_df.iloc[test_inds]

        print(f"\nXval iter {i}, train size {X_train.shape}, test size {X_test.shape}")

        # feature selection
        fpr = SelectFpr(f_classif, alpha=0.01).fit(X_train, y_train)
        X_train_sub = fpr.transform(X_train)
        X_test_sub = fpr.transform(X_test)
        print(f"Shapes after feature selection: X_train_sub: {X_train_sub.shape}, X_test_sub: {X_test_sub.shape}")

        # xval training
        parameters ={'C':[0.01,0.1,1,10,100,1000]}
        logo_in = LeaveOneGroupOut()
        gscv = GridSearchCV(
            LogisticRegression(penalty='l2', solver='liblinear'), 
            parameters,
            cv=logo, return_train_score=True, verbose=5, n_jobs=32)
        gscv.fit(X_train_sub, y_train, groups=train_groups)
        model = gscv 
        
        # test
        score = model.score(X_test_sub, y_test)
        auc_score = roc_auc_score(y_test, model.predict_proba(X_test_sub), multi_class='ovr')
        preds = model.predict(X_test_sub)
        # confusion matrix
        true_counts = np.asarray([np.sum(y_test == i) for i in stim_labels.keys()])
        cm = confusion_matrix(y_test, preds, labels=list(stim_labels.keys())) / true_counts[:,None] * 100

        # decode metrics
        evid = (1. / (1. + np.exp(-model.decision_function(X_test_sub))))
        prob = model.predict_proba(X_test_sub)

        # save
        models.append(model)
        scores.append(score)
        auc_scores.append(auc_score)
        cms.append(cm)
        evids.append(evid)
        probs.append(prob)
        decode_dfs.append(decode_df)

        if v: print(f"\nIter {i} score: \n"
            f"score: {score}, auc score: {auc_score}\n"
            f"confution matrix:\n"
            f"{cm}")

    scores = np.asarray(scores)
    auc_scores = np.asarray(auc_scores)
    cms = np.stack(cms)
    evids = np.stack(evids)  # xval x sample x class
    probs = np.stack(probs)  # xval x sample x class
    decode_dfs = pd.concat(decode_dfs)  # all samples

    print(f"\nClassifier score: \n"
        f"scores: {scores.mean()} +/- {scores.std()}\n"
        f"auc scores: {auc_scores.mean()} +/- {auc_scores.std()}\n"
        f"average confution matrix:\n"
        f"{cms.mean(axis=0)}")

    # save
    out_path = os.path.join(results_dir, "operation_clf", f"sub-{subID}_clearmem_gscv")
    print(f"Saving results to {out_path}...")
    np.savez_compressed(out_path, models=models, scores=scores, auc_scores=auc_scores, cms=cms, evids=evids, probs=probs, decode_dfs=decode_dfs)

def clearmem_op_decode(subID="006"):
    target_ops = [1,2,3]
    
    # load stored data
    file_fname = os.path.join(results_dir, "operation_clf", f"sub-{subID}_clearmem_gscv")
    f = np.load(file_fname, allow_pickle=True)
    df, evids, probs = f["decode_dfs"], f["evids"], f["probs"]

    # create df of trial per row
    run_trial_df = []
    for runi, rdf in df.groupby(["run"]):
        for tid, tdf in rdf.groupby(["trial"]):
            run_trial_df.append(tdf.iloc[0])
    run_trial_df = np.vstack(run_trial_df)
    run_trial_df = pd.DataFrame(run_trial_df, columns=run_df.columns) 

    # put together
    all_op_trialIDs = {}
    all_op_evids = {}
    all_op_probs = {}
    for runi, run_df in run_trial_df.groupby(["run"]):
        # === get trialIDs for each (N-1, N)
        run_op_trialIDs = {}
        for x in target_ops:
            for y in target_ops:
                run_op_trialIDs[f"{x}-{y}"] = []

        for i in range(1, len(run_df)):
            prev = run_df.iloc[i-1]["condition"]
            curr = run_df.iloc[i]["condition"]
            
            # if (prev not in target_ops) or (curr not in target_ops):
            #     continue
            op_dict_count[f"{prev}-{curr}"].append(i)
            
        for k, v in op_dict_count.items():
            op_dict_count[k] = np.array(v)

        # === put measures in
        run_evids = evids[runi]
        run_probs = probs[runi]

        run_op_evids = {}
        run_op_probs = {}
        for opk, opIDs in run_op_trialIDs.items():
            run_op_evids[opk] = run_evids[opIDs]
            run_op_probs[opk] = run_probs[opIDs]  # trial x class

        # === store
        all_op_trialIDs[runi] = run_op_trialIDs
        all_op_evids[runi] = run_op_evids
        all_op_probs[runi] = run_op_probs

    # save file to plot offline
    out_fname = f"sub-{subID}_decode_groups"
    out_path = os.path.join(results_dir, "operation_clf", out_fname)
    print(f"Saving to {out_path}...")
    np.savez_compressed(out_path, trialIDs=all_op_trialIDs, evids=all_op_evids, probs=all_op_probs)

if __name__ == "__main__":
    subID = "006"
    clearmem_op_classification(subID)
    clearmem_op_decode(subID)






