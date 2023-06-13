import warnings
import sys

if not sys.warnoptions:
    warnings.simplefilter("ignore")
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import nibabel as nib
from nilearn.image import clean_img
from nilearn.signal import clean

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.feature_selection import SelectFpr, f_classif
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

from cross_exp_classifier import get_preprocessed_data, get_shifted_labels, subsample


# global consts
subIDs = ["002", "003", "004"]
phases = ["rest", "preremoval", "study", "postremoval"]
runs = np.arange(6) + 1
spaces = {"T1w": "T1w", "MNI": "MNI152NLin2009cAsym"}
descs = ["brain_mask", "preproc_bold"]
ROIs = ["VVS", "PHG", "FG"]
shift_sizes_TR = [5, 6]

# *** clearmem labels
stim_labels = {0: "Rest", 1: "Face", 2: "Fruit", 3: "Scene"}

op_labels = {1: "maintain", 2: "replace", 3: "suppress"}

workspace = "scratch"
if workspace == "work":
    clearmem_dir = "/work/07365/sguo19/frontera/clearmem/"
    repclear_dir = "/work/07365/sguo19/frontera/fmriprep/"
    param_dir = "/work/07365/sguo19/frontera/params/"
    results_dir = "/work/07365/sguo19/model_fitting_results/"
elif workspace == "scratch":
    clearmem_dir = "/scratch1/07365/sguo19/clearmem/"
    repclear_dir = "/scratch1/07365/sguo19/fmriprep/"
    param_dir = "/scratch1/07365/sguo19/params/"
    results_dir = "/scratch1/07365/sguo19/model_fitting_results/"


def clearmem_cate_classification(subID="006", xval=False, C=0.01):
    # subs with study phase VTC mask: 006, 036, 077, 079
    task = "localizer"
    n_runs = 5
    space = "MNI"
    shift_size_TR = 10
    rest_tag = 0
    include_rest = True
    v = True

    # ===== load sub data & labels
    mask_path = os.path.join(
        clearmem_dir, f"sub-{subID}", "new_mask", f"LOC_VTC_{task}_{space}_mask.nii.gz"
    )
    full_data = get_preprocessed_data(
        "clearmem",
        subID,
        task,
        space,
        mask_path,
        data_tag="",
        runs=np.arange(n_runs) + 1,
        save=True,
    )
    print("BOLD shape: ", full_data.shape)

    # label
    label_df = get_shifted_labels("clearmem", task, shift_size_TR, rest_tag)
    print("Labels shape: ", label_df.shape)

    # subsample
    X, Y_df = subsample(
        "clearmem",
        full_data,
        label_df,
        task="localizer",
        include_rest=include_rest,
    )
    Y = Y_df["category"].values
    run_ls = Y_df["run"].values
    print(f"\nShape after subsample: X: {X.shape}, Y: {Y.shape}")

    if xval:
        print("Running xval...")
        # ===== xval
        # performance eval
        scores = []
        auc_scores = []
        cms = []
        models = []
        fprs = []
        best_Cs = []

        print("\nTraining classifier...")
        logo = LeaveOneGroupOut()
        for i, (train_inds, test_inds) in enumerate(logo.split(X, Y, groups=run_ls)):
            X_train, X_test = X[train_inds], X[test_inds]
            y_train, y_test = Y[train_inds], Y[test_inds]
            train_groups = run_ls[train_inds]

            print(
                f"\nXval iter {i}, train size {X_train.shape}, test size {X_test.shape} (run {run_ls[test_inds[0]]})"
            )

            # feature selection
            fpr = SelectFpr(f_classif, alpha=0.01).fit(X_train, y_train)
            X_train_sub = fpr.transform(X_train)
            X_test_sub = fpr.transform(X_test)
            print(
                f"Shapes after feature selection: X_train_sub: {X_train_sub.shape}, X_test_sub: {X_test_sub.shape}"
            )

            # xval training
            parameters = {"C": [0.01, 0.1, 1, 10, 100, 1000]}
            logo_in = LeaveOneGroupOut()
            gscv = GridSearchCV(
                LogisticRegression(penalty="l2", solver="liblinear"),
                parameters,
                cv=logo,
                return_train_score=True,
                verbose=5,
                n_jobs=32,
            )
            gscv.fit(X_train_sub, y_train, groups=train_groups)
            model = gscv

            # test
            score = model.score(X_test_sub, y_test)
            auc_score = roc_auc_score(
                y_test, model.predict_proba(X_test_sub), multi_class="ovr"
            )
            preds = model.predict(X_test_sub)
            # confusion matrix
            true_counts = np.asarray([np.sum(y_test == i) for i in stim_labels.keys()])
            cm = (
                confusion_matrix(y_test, preds, labels=list(stim_labels.keys()))
                / true_counts[:, None]
                * 100
            )

            # save
            models.append(model)
            fprs.append(fpr)
            scores.append(score)
            auc_scores.append(auc_score)
            cms.append(cm)
            best_Cs.append(model.best_params_["C"])

            if v:
                print(
                    f"\nIter {i} score: \n"
                    f"score: {score}, auc score: {auc_score}\n"
                    f"confution matrix:\n"
                    f"{cm}"
                )

        scores = np.asarray(scores)
        auc_scores = np.asarray(auc_scores)
        cms = np.stack(cms)

        print(
            f"\nClassifier score: \n"
            f"scores: {scores.mean()} +/- {scores.std()}\n"
            f"auc scores: {auc_scores.mean()} +/- {auc_scores.std()}\n"
            f"best Cs: {best_Cs}\n"
            f"average confution matrix:\n"
            f"{cms.mean(axis=0)}"
        )

        # save
        # out_path = os.path.join(results_dir, "category_clf", f"sub-{subID}_clearmem_task-{task}_space-{space}_VTC_gscv.npz")
        # print(f"Saving results to {out_path}...")
        # np.savez_compressed(out_path, models=models, fprs=fprs, scores=scores, auc_scores=auc_scores, cms=cms)

    else:
        print("Running final model on all data...")
        # ===== use given C value to train on all data
        # feature selection
        fpr = SelectFpr(f_classif, alpha=0.01).fit(X, Y)
        X_sub = fpr.transform(X)
        print(f"Shape after feature selection: X_sub: {X_sub.shape}")

        lr = LogisticRegression(
            penalty="l2", solver="liblinear", C=C, verbose=5, n_jobs=32
        )
        lr.fit(X_sub, Y)
        model = lr

        # test
        score = model.score(X_sub, Y)
        auc_score = roc_auc_score(Y, model.predict_proba(X_sub), multi_class="ovr")
        preds = model.predict(X_sub)
        # confusion matrix
        true_counts = np.asarray([np.sum(Y == i) for i in stim_labels.keys()])
        cm = (
            confusion_matrix(Y, preds, labels=list(stim_labels.keys()))
            / true_counts[:, None]
            * 100
        )

        print(
            f"\nClassifier score: \n"
            f"score: {score}\n"
            f"auc score: {auc_score}\n"
            f"confution matrix:\n"
            f"{cm}"
        )

        # save
        out_path = os.path.join(
            results_dir,
            "category_clf",
            f"sub-{subID}_clearmem_task-{task}_space-{space}_VTC_final-lr.pkl",
        )
        print(f"Saving results to {out_path}...")
        # np.savez_compressed(out_path, model=model, fpr=fpr, score=score, auc_score=auc_score, cm=cm)
        out_dict = dict(model=model, fpr=fpr, score=score, auc_score=auc_score, cm=cm)
        with open(out_path, "wb") as f:
            pickle.dump(out_dict, f)


def organize_decode_results(data, df):
    """
    Make data (xval x TR x class) ==> (trial x xval x TR x class)
    df: design matrix df to organize trials

    Return
    dtata_out: *LIST* of trial TRs (trial x xval x TR x class). can't make ndarray bc TR size diff.
    trial_df: original df organized into one trial per row, consistent with trial dim of returned data
    """
    # make df of trial per row
    trial_df = []
    for tid, tdf in df.groupby(["trial"]):
        trial_df.append(tdf.iloc[0])
    trial_df = np.vstack(trial_df)
    trial_df = pd.DataFrame(trial_df, columns=df.columns)

    # get trial data
    data_out = []
    for ti in trial_df["trial"]:
        trial_inds = np.where(df["trial"] == ti)[0]

        # add 5 TR before stim pres
        bf_trial_inds = np.arange(trial_inds[0] - 5, trial_inds[0])
        trial_inds = np.r_[bf_trial_inds, trial_inds]

        trial_data = data[:, trial_inds, :]  # xval x TR x class
        data_out.append(trial_data)
    # # TODO: figure out way to stack without same dim
    # data_out = np.stack(data_out, axis=1)

    return data_out, trial_df


def clearmem_study_decode(subID="006"):
    # subs with study phase VTC mask: 006, 036, 077, 079
    task = "study"
    n_runs = 6
    space = "MNI"
    shift_size_TR = 10
    rest_tag = 0
    # include_rest = True
    # v = True

    # === load model
    file_path = os.path.join(
        results_dir,
        "category_clf",
        f"sub-{subID}_clearmem_task-localizer_space-{space}_VTC_final-lr.pkl",
    )
    # f = np.load(file_path, allow_pickle=True)
    # model, fpr = f["model"], f["fpr"]
    with open(file_path, "rb") as f:
        out_dict = pickle.load(f)
    model, fpr = out_dict["model"], out_dict["fpr"]

    # === load study data
    mask_path = os.path.join(
        clearmem_dir, f"sub-{subID}", "new_mask", f"LOC_VTC_{task}_{space}_mask.nii.gz"
    )
    full_data = get_preprocessed_data(
        "clearmem",
        subID,
        task,
        space,
        mask_path,
        data_tag="",
        runs=np.arange(6) + 1,
        save=True,
    )
    print("BOLD shape: ", full_data.shape)

    # === decode every TR
    data_sub = fpr.transform(full_data)
    # decode metrics
    evid = 1.0 / (1.0 + np.exp(-model.decision_function(data_sub)))  # TR x class
    prob = model.predict_proba(data_sub)

    # === organize into (xval x trial x TR x class)
    # load df
    label_df = get_shifted_labels("clearmem", task, shift_size_TR, rest_tag)
    print("Labels shape: ", label_df.shape)
    # # make df of trial per row
    # trial_df = []
    # for tid, tdf in label_df.groupby(["trial"]):
    #     trial_df.append(tdf.iloc[0])
    # trial_df = np.vstack(trial_df)
    # trial_df = pd.DataFrame(trial_df, columns=label_df.columns)

    # === save file (can ignore df; no organizing needed, all done later in organizer)
    out_fname = os.path.join(
        results_dir, "category_clf", f"sub-{subID}_final-lr-decode.pkl"
    )
    print(f"Saving to {out_fname}...")
    # np.savez_compressed(out_fname, evids=evids, probs=probs, label_df=label_df)
    out_dict = dict(evid=evid, prob=prob, label_df=label_df)
    with open(out_fname, "wb") as f:
        pickle.dump(out_dict, f)


if __name__ == "__main__":
    subID = "079"
    # clearmem_cate_classification(subID, xval=True)
    # clearmem_cate_classification(subID, C=0.01)
    clearmem_study_decode(subID)
