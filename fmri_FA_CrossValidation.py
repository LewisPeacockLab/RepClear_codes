import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
import nibabel as nib
import scipy as scipy
from scipy import stats
import scipy.io as sio
import fnmatch
from sklearn.svm import SVC
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import (
    PredefinedSplit,
    cross_validate,
    cross_val_predict,
    GridSearchCV,
    LeaveOneGroupOut,
    cross_val_score,
)
from sklearn.feature_selection import (
    VarianceThreshold,
    f_classif,
    SelectKBest,
    SelectFpr,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from nilearn.input_data import NiftiMasker, MultiNiftiMasker
from nilearn.image import clean_img
from nilearn.signal import clean
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from joblib import Parallel, delayed


import warnings

warnings.filterwarnings("ignore")

# global consts
subIDs = [
    "002",
    "003",
    "004",
    "005",
    "006",
    "007",
    "008",
    "009",
    "010",
    "011",
    "012",
    "013",
    "014",
    "015",
    "016",
    "017",
    "018",
    "020",
    "023",
    "024",
    "025",
]
phases = ["rest", "preremoval", "study", "postremoval"]
runs = np.arange(6) + 1
spaces = {"T1w": "T1w", "MNI": "MNI152NLin2009cAsym"}
descs = ["brain_mask", "preproc_bold"]
ROIs = ["VVS", "LatOcc"]
shift_sizes_TR = [5]

stim_labels = {0: "Rest", 1: "Scenes", 2: "Faces"}

workspace = "scratch"
data_dir = "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/"
param_dir = "/scratch/06873/zbretton/repclear_dataset/BIDS/params/"
results_dir = "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/model_fitting_results/"


def get_preprocessed_data(
    subID, task, space, mask_ROIS, runs=np.arange(6) + 1, save=False
):
    """
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
    """

    space_long = spaces[space]

    def load_existing_data():
        print("\n*** Attempting to load existing data if there is any...")
        preproc_data = {}
        todo_ROIs = []

        if type(mask_ROIS) == str and mask_ROIS == "wholebrain":
            if os.path.exists(
                os.path.join(bold_dir, out_fname_template.format("wholebrain"))
            ):
                print("Loading saved preprocessed data", out_fname_template, "...")
                preproc_data["wholebrain"] = np.load(
                    os.path.join(bold_dir, out_fname_template.format("wholebrain"))
                )
            else:
                print("Wholebrain data to be processed.")
                todo_ROIs = "wholebrain"

        elif type(mask_ROIS) == list:
            for ROI in mask_ROIS:
                if ROI == "VVS":
                    ROI = "vtc"  # only change when loading saved data
                if os.path.exists(
                    os.path.join(bold_dir, out_fname_template.format(ROI))
                ):
                    print(
                        "Loading saved preprocessed data",
                        out_fname_template.format(ROI),
                        "...",
                    )
                    preproc_data[ROI] = np.load(
                        os.path.join(bold_dir, out_fname_template.format(ROI))
                    )
                else:
                    if ROI == "vtc":
                        ROI = "VVS"  # change back to laod masks...
                    print(f"ROI {ROI} data to be processed.")
                    todo_ROIs.append(ROI)
        else:
            raise ValueError(f"Man! Incorrect ROI value! (Entered: {mask_ROIS})")

        return preproc_data, todo_ROIs

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
        # *** pd future warning: "A value is trying to be set on a copy of a slice from a DataFrame" ***
        confounds.loc[0, "framewise_displacement"] = confounds.loc[
            1:, "framewise_displacement"
        ].mean()
        return confounds

    def apply_mask(mask=None, target=None):
        coor = np.where(mask == 1)
        values = target[coor]
        # *** data is already vox x time when loaded ***
        # print("before transpose:", values.shape)
        # if values.ndim > 1:
        #     values = np.transpose(values) #swap axes to get feature X sample
        # print("after transpose:", values.shape)
        return values

    # ====================================================
    print(
        f"\n***** Data preprocessing for sub {subID} {task} {space} with ROIs {mask_ROIS}..."
    )

    # whole brain mask: includeing white matter. should be changed to grey matter mask later
    if mask_ROIS == "wholebrain":
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
    fname_template = (
        f"sub-{subID}_task-{task}_run-{{}}_space-{space_long}_desc-{{}}.nii.gz"
    )
    bold_fnames = [fname_template.format(i, "preproc_bold") for i in runs]
    bold_paths = [
        os.path.join(data_dir, f"sub-{subID}", "func", fname) for fname in bold_fnames
    ]

    # get mask name
    if type(mask_ROIS) == str:  # 'wholebrain'
        # whole brain masks: 1 for each run
        mask_fnames = [fname_template.format(i, "brain_mask") for i in runs]
        mask_paths = [
            os.path.join(data_dir, f"sub-{subID}", "func", fname)
            for fname in mask_fnames
        ]
    else:
        # ROI masks: 1 for each ROI
        mask_fnames = [f"{ROI}_{task}_{space}_mask.nii.gz" for ROI in mask_ROIS]
        mask_paths = [
            os.path.join(data_dir, f"sub-{subID}", "new_mask", fname)
            for fname in mask_fnames
        ]

    # get confound filenames
    confound_fnames = [f"*{task}*{run}*confounds*.tsv" for run in runs]
    confound_paths = [
        os.path.join(data_dir, f"sub-{subID}", "func", f) for f in confound_fnames
    ]  # template for each run
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
    confounds = [pd.read_csv(p, sep="\t") for p in confound_paths]
    confounds_cleaned = [confound_cleaner(c) for c in confounds]

    # ===== for each run & ROI, mask & clean
    print("\n*** Masking & cleaing bold data...")
    if type(mask_ROIS) == str:  # 'wholebrain'
        cleaned_bolds = [None for _ in range(len(runs))]
        # all files are by nruns
        for runi, (bold, mask, confound) in enumerate(
            zip(bolds, masks, confounds_cleaned)
        ):
            print(f"Processing run {runi}...")
            masked = apply_mask(mask=mask.get_data(), target=bold.get_data())
            # *** clean: confound length in time; transpose signal to (time x vox) for cleaning & model fitting ***
            cleaned_bolds[runi] = clean(
                masked.T, confounds=confound, t_r=1, detrend=False, standardize="zscore"
            )
            print("claened shape: ", cleaned_bolds[runi].shape)

        # {ROI: time x vox}
        preproc_data = {"wholebrain": np.hstack(cleaned_bolds)}

    else:  # list of specific ROIs
        cleaned_bolds = [
            [None for _ in range(len(runs))] for _ in range(len(mask_ROIS))
        ]

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
                cleaned_bolds[rowi][coli] = clean(
                    masked.T,
                    confounds=confound,
                    t_r=1,
                    detrend=False,
                    standardize="zscore",
                )
                print(f"ROI {rowi}, run {coli}")
                print(f"shape: {cleaned_bolds[rowi][coli].shape}")

        # {ROI: time x vox}
        preproc_data = {
            ROI: np.vstack(run_data) for ROI, run_data in zip(mask_ROIS, cleaned_bolds)
        }

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
        shift = pd.DataFrame(
            np.zeros((TR_shift_size, nvars)) + tag, columns=label_df.columns
        )
        shifted = pd.concat([shift, label_df])
        return shifted[: len(label_df)]  # trim time points outside of scanning time

    print("\n***** Loading labels...")

    event_path = os.path.join(param_dir, f"{task}_events.csv")
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

    stim_list = label_df["category"]
    stim_on = label_df["stim_present"]

    labels = set(stim_list)
    labels.remove(0)  # rest will be done separately afterwards

    # indices for each category
    stim_inds = {
        lab: np.where((stim_on == 1) & (stim_list == lab))[0] for lab in labels
    }
    # min_n = min([len(inds) for inds in stim_inds.values()])  # min sample size to get from each category

    min_n = 60  # so we only take 60 from Faces and Scenes to equal the 120 Rest periods we will get

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
        padded_bools = np.r_[
            False, rest_bools, False
        ]  # pad the bools at beginning and end for diff to operate
        rest_diff = np.diff(
            padded_bools
        )  # get the pairwise diff in the array --> True for start and end indices of rest periods
        rest_intervals = rest_diff.nonzero()[0].reshape(
            (-1, 2)
        )  # each pair is the interval of rest periods
        print("random sample rest_intervals: ", rest_intervals)

        # get desired time points: can be changed to be middle/end of resting periods, or just random subsample
        # current choice: get time points in the middle of rest periods for rest samples; if 0.5, round up
        rest_inds = [
            np.ceil(np.average(interval)).astype(int) for interval in rest_intervals
        ]

        # subsample to min_n
        chosen_rest_inds = np.random.choice(rest_inds, min_n * 2, replace=False)
        sampled_inds[0] = sorted(chosen_rest_inds)

    # ===== stack indices
    X = []
    Y = []
    for lab, inds in sampled_inds.items():
        X.append(full_data[inds, :])
        Y.append(np.zeros(len(inds)) + lab)

    X = np.vstack(X)
    Y = np.concatenate(Y)
    return X, Y


def subsample_by_runs(full_data, label_df, include_rest=True):
    """
    Subsample data by runs. Yield splits or all combination of 2 runs.
    Return: stacked X & Y for train/test split & model fitting
    """

    # stim_list: 1 Scenes, 2 Faces / 0 is rest
    # stim_on labels: 1 actual stim; 2 rest between stims; 0 rest between runs
    print("\n***** Subsampling data points by runs...")

    stim_list = label_df["category"]
    stim_on = label_df["stim_present"]
    run_list = label_df["run"]

    # get faces
    face_inds = np.where((stim_on == 1) & (stim_list == 2))[0]
    rest_inds = []
    groups = np.concatenate(
        [np.full(int(len(face_inds) / 2), 1), np.full(int(len(face_inds) / 2), 2)]
    )

    scenes_runs = [3, 4, 5, 6]
    for i in range(len(scenes_runs)):
        runi = scenes_runs[i]
        for j in range(i + 1, len(scenes_runs)):
            runj = scenes_runs[j]
            print(f"\nSubsampling scenes with runs {runi} & {runj}...")

            # choose scene samples based on run
            scene_inds = np.where(
                (stim_on == 1)
                & (stim_list == 1)
                & ((run_list == runi) | (run_list == runj))
            )[
                0
            ]  # actual stim; stim is scene; stim in the two runs

            if include_rest:
                print("Including resting category...")
                # get TR intervals for rest between stims (stim_on == 2)
                rest_bools = ((run_list == runi) | (run_list == runj)) & (stim_on == 2)
                padded_bools = np.r_[
                    False, rest_bools, False
                ]  # pad the bools at beginning and end for diff to operate
                rest_diff = np.diff(
                    padded_bools
                )  # get the pairwise diff in the array --> True for start and end indices of rest periods
                rest_intervals = rest_diff.nonzero()[0].reshape(
                    (-1, 2)
                )  # each pair is the interval of rest periods

                # get desired time points: can be changed to be middle/end of resting periods, or just random subsample
                # current choice: get time points in the middle of rest periods for rest samples; if 0.5, round up
                rest_intervals[:, -1] -= 1
                rest_inds = [
                    np.ceil(np.average(interval)).astype(int)
                    for interval in rest_intervals
                ] + [
                    np.ceil(np.average(interval)).astype(int) + 1
                    for interval in rest_intervals
                ]

                # should give same number of rest samples; if not, do random sample
                # rest_inds = np.random.choice(rest_inds, len(face_inds), replace=False)

            # === get X & Y
            X = []
            Y = []
            print(
                f"rest_inds: {len(rest_inds)}, scene_inds: {len(scene_inds)}, face_inds: {len(face_inds)}"
            )
            for lab, inds in zip([0, 1, 2], [rest_inds, scene_inds, face_inds]):
                print("label counts:", lab, len(inds))
                X.append(full_data[inds, :])
                Y.append(np.zeros(len(inds)) + lab)

            X = np.vstack(X)
            Y = np.concatenate(Y)
            all_groups = np.concatenate([groups, groups, groups])
            yield X, Y, all_groups

            # flip groups so even & odd groups can be paired
            all_groups = np.concatenate(
                [groups, list(reversed(groups)), list(reversed(groups))]
            )
            yield X, Y, all_groups


# subID='002'

for subID in subIDs:
    task = "preremoval"
    space = "MNI"
    ROIs = ["VVS"]
    shift_size_TR = shift_sizes_TR[0]
    rest_tag = 0

    # get data: all_ROI_vox x all_runs_time
    full_data = get_preprocessed_data(subID, task, space, ROIs, save=True)
    print(f"Full_data shape: {full_data.shape}")

    # get labels
    label_df = get_shifted_labels(task, shift_size_TR, rest_tag)
    print(f"Category label shape: {label_df.shape}")

    ###############Leave One Group Out#########################

    FA_total_std = []
    FA_total_mean = []

    PCA_total_std = []
    PCA_total_mean = []

    n_components = [
        2,
        5,
        10,
        50,
        100,
        1000,
        np.shape(full_data)[1],
    ]  # last value is # of voxels
    logo = LeaveOneGroupOut()  # Using LOGO; Sadtler et al. 2014 used K=4

    fa = FactorAnalysis(svd_method="lapack")
    fa_mean = []
    fa_std = []
    fa_predict = []

    pca = PCA(svd_solver="full")
    pca_mean = []
    pca_std = []

    fa_decoding = {}
    pca_decoding = {}

    for n in n_components:
        for X, Y, groups in subsample_by_runs(full_data, label_df):
            print("Start FA: {} components".format(n))
            fa.n_components = n  # default: #units
            fa.rotation = None  # default: None; options: ‘varimax’(orthogonal), ‘quartimax’ (oblique) ***
            # fa.max_iter     = 1e4 #default: 1e3 # currently returns NaN if used

            fa_scores = cross_val_score(
                estimator=fa, X=X, y=Y, groups=groups, cv=logo, n_jobs=-1
            )
            print("FA Score:", fa_scores)
            fa_mean.append(np.mean(fa_scores))
            fa_std.append(stats.sem(fa_scores))

            print("Start PCA: {} components".format(n))
            pca.n_components = n  # default: #units

            pca_scores = cross_val_score(
                estimator=pca, X=X, y=Y, groups=groups, cv=logo, n_jobs=-1
            )
            print("PCA Score:", pca_scores)
            pca_mean.append(np.mean(pca_scores))
            pca_std.append(stats.sem(pca_scores))

            scores = []
            auc_scores = []
            cms = []
            for train_inds, test_inds in logo.split(X, Y, groups):
                X_train, X_test, y_train, y_test = (
                    X[train_inds],
                    X[test_inds],
                    Y[train_inds],
                    Y[test_inds],
                )
                X_train = fa.fit_transform(X_train, y_train)

                X_test = fa.transform(X_test)

                lr = LogisticRegression(penalty="l2", solver="liblinear", max_iter=1000)
                lr.fit(X_train, y_train)
                # test
                score = lr.score(X_test, y_test)
                auc_score = roc_auc_score(
                    y_test, lr.predict_proba(X_test), multi_class="ovr"
                )
                preds = lr.predict(X_test)

                # confusion matrix
                cm = confusion_matrix(y_test, preds, normalize="true")

                scores.append(score)
                auc_scores.append(auc_score)
                cms.append(cm)
            itr_score = np.mean(scores)
            itr_auc = np.mean(auc_scores)
            itr_cms = np.mean(cms, axis=0)

            scores = []
            auc_scores = []
            cms = []
            for train_inds, test_inds in logo.split(X, Y, groups):
                X_train, X_test, y_train, y_test = (
                    X[train_inds],
                    X[test_inds],
                    Y[train_inds],
                    Y[test_inds],
                )
                X_train = pca.fit_transform(X_train, y_train)

                X_test = pca.transform(X_test)

                lr = LogisticRegression(penalty="l2", solver="liblinear", max_iter=1000)
                lr.fit(X_train, y_train)
                # test
                score = lr.score(X_test, y_test)
                auc_score = roc_auc_score(
                    y_test, lr.predict_proba(X_test), multi_class="ovr"
                )
                preds = lr.predict(X_test)

                # confusion matrix
                cm = confusion_matrix(y_test, preds, normalize="true")

                scores.append(score)
                auc_scores.append(auc_score)
                cms.append(cm)
            pca_itr_score = np.mean(scores)
            pca_itr_auc = np.mean(auc_scores)
            pca_itr_cms = np.mean(cms, axis=0)

        fa_decoding[n] = {
            "score": itr_score,
            "auc": itr_auc,
            "confusion_matrix": itr_cms,
        }

        pca_decoding[n] = {
            "score": pca_itr_score,
            "auc": pca_itr_auc,
            "confusion_matrix": pca_itr_cms,
        }

        FA_total_mean.append(fa_mean)
        FA_total_std.append(fa_std)
        PCA_total_mean.append(pca_mean)
        PCA_total_std.append(pca_std)

    print(np.mean(FA_total_mean, axis=0))
    print(np.mean(PCA_total_mean, axis=0))

    fa_df = pd.DataFrame(data=FA_total_mean, columns=n_components)
    pca_df = pd.DataFrame(data=PCA_total_mean, columns=n_components)

    fa_std_df = pd.DataFrame(data=FA_total_std, columns=n_components)
    pca_std_df = pd.DataFrame(data=PCA_total_std, columns=n_components)

    pd.to_pickle(
        fa_df,
        "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/sub-%s/VTC_subsample_LOGO_FA_CV_LL.pickle"
        % subID,
    )
    pd.to_pickle(
        pca_df,
        "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/sub-%s/VTC_subsample_LOGO_PCA_CV_LL.pickle"
        % subID,
    )

    del fa_df, pca_df
    ################ 4 K-fold CV ######################################

    FA_total_std = []
    FA_total_mean = []

    PCA_total_std = []
    PCA_total_mean = []

    print("Running K-Fold CV fitting...")

    n_components = [
        2,
        5,
        10,
        50,
        100,
        1000,
        np.shape(full_data)[1],
    ]  # last value is # of voxels
    k_folds = 4  #'None'/default to 5-fold CV; Sadtler et al. 2014 used K=4

    def compute_scores(X):
        fa = FactorAnalysis(svd_method="lapack")
        fa_mean = []
        fa_std = []

        pca = PCA(svd_solver="full")
        pca_mean = []
        pca_std = []

        for n in n_components:
            print("Start FA: {} components".format(n))
            fa.n_components = n  # default: #units
            fa.rotation = None  # default: None; options: ‘varimax’(orthogonal), ‘quartimax’ (oblique) ***
            fa.max_iter = 10000  # default: 1e3 # currently returns NaN if used

            fa_scores = cross_val_score(estimator=fa, X=X, cv=k_folds, n_jobs=-1)
            print("FA Score:", fa_scores)
            fa_mean.append(np.mean(fa_scores))
            fa_std.append(stats.sem(fa_scores))

            print("Start PCA: {} components".format(n))
            pca.n_components = n  # default: #units

            pca_scores = cross_val_score(estimator=pca, X=X, cv=k_folds, n_jobs=-1)
            print("PCA Score:", pca_scores)
            pca_mean.append(np.mean(pca_scores))
            pca_std.append(stats.sem(pca_scores))

        return pca_mean, pca_std, fa_mean, fa_std

    pca_mean, pca_std, fa_mean, fa_std = compute_scores(
        full_data
    )  # running the 4-fold CV with all the data, similar to previous methods

    FA_total_std = fa_std
    FA_total_mean = fa_mean

    PCA_total_std = pca_std
    PCA_total_mean = pca_mean

    fa_df = pd.DataFrame(data=FA_total_mean).T
    fa_df.columns = n_components
    pca_df = pd.DataFrame(data=PCA_total_mean).T
    pca_df.columns = n_components

    pd.to_pickle(
        fa_df,
        "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/sub-%s/VTC_all_4fold_FA_CV_LL.pickle"
        % subID,
    )
    pd.to_pickle(
        pca_df,
        "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/sub-%s/VTC_all_4fold_PCA_CV_LL.pickle"
        % subID,
    )

########PLOTS############
# plt.errorbar(x=n_components[:8], y=fa_df[[2, 5, 7, 9, 10, 50, 100, 150]].mean(), yerr=fa_std_df[[2, 5, 7, 9, 10, 50, 100, 150]].mean(), color='k', linewidth=1.5)
# maxLL_ind = np.where(fa_df[[2, 5, 7, 9, 10, 50, 100, 150]].mean() == max(fa_df[[2, 5, 7, 9, 10, 50, 100, 150]].mean()))[0][0]
# maxLL = n_components[maxLL_ind]
# plt.scatter(maxLL, fa_df[[2, 5, 7, 9, 10, 50, 100, 150]].mean()[maxLL], color='orange', s=100, edgecolor='k', zorder=3, label='Estimated Dimensionality')
# plt.axvline(maxLL, color='orange', ls='--', zorder=1)

# plt.title('LOGO Cross-Validation - sub-002 | Subsample of Localizer | Lateral Occ ROI')
# plt.ylabel('Log Likelihood')
# plt.xlabel('Dimensionality')
# plt.legend()
# plt.tight_layout()
# plt.savefig('/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/factor_analysis/figures/CV_FA_LikelihoodPlot.png')
# plt.clf()

# plt.errorbar(x=n_components[:8], y=pca_df[[2, 5, 7, 9, 10, 50, 100, 150]].mean(), yerr=pca_std_df[[2, 5, 7, 9, 10, 50, 100, 150]].mean(), color='k', linewidth=1.5)
# maxLL_ind = np.where(pca_df[[2, 5, 7, 9, 10, 50, 100, 150]].mean() == max(pca_df[[2, 5, 7, 9, 10, 50, 100, 150]].mean()))[0][0]
# maxLL = n_components[maxLL_ind]
# plt.scatter(maxLL, pca_df[[2, 5, 7, 9, 10, 50, 100, 150]].mean()[maxLL], color='orange', s=100, edgecolor='k', zorder=3, label='Estimated Dimensionality')
# plt.axvline(maxLL, color='orange', ls='--', zorder=1)

# plt.title('LOGO PCA Cross-Validation - sub-002 | Subsample of Localizer | Lateral Occ ROI')
# plt.ylabel('Log Likelihood')
# plt.xlabel('Dimensionality')
# plt.legend()
# plt.tight_layout()
# plt.savefig('/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/factor_analysis/figures/CV_PCA_LikelihoodPlot.png')


# plt.errorbar(x=n_components, y=fa_mean, yerr=fa_std, color='k', linewidth=1.5)
# maxLL_ind = np.where(fa_mean == max(fa_mean))[0][0]
# maxLL = n_components[maxLL_ind]
# plt.scatter(maxLL, fa_mean[maxLL_ind], color='orange', s=100, edgecolor='k', zorder=3, label='Estimated Dimensionality')
# plt.axvline(maxLL, color='orange', ls='--', zorder=1)

# plt.title('4-Fold Cross-Validation FA')
# plt.ylabel('Log Likelihood')
# plt.xlabel('Dimensionality')
# plt.legend()
# plt.tight_layout()
# plt.savefig('/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/factor_analysis/figures/CV_FA_alldata_LikelihoodPlot.png')
# plt.clf()

# plt.errorbar(x=n_components, y=pca_mean, yerr=pca_std, color='k', linewidth=1.5)
# maxLL_ind = np.where(pca_mean == max(pca_mean))[0][0]
# maxLL = n_components[maxLL_ind]
# plt.scatter(maxLL, pca_mean[maxLL_ind], color='orange', s=100, edgecolor='k', zorder=3, label='Estimated Dimensionality')
# plt.axvline(maxLL, color='orange', ls='--', zorder=1)

# plt.title('4-Fold Cross-Validation PCA')
# plt.ylabel('Log Likelihood')
# plt.xlabel('Dimensionality')
# plt.legend()
# plt.tight_layout()
# plt.savefig('/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/factor_analysis/figures/CV_PCA_alldata_LikelihoodPlot.png')
# plt.clf()
# #############
# plt.errorbar(x=n_components[:9], y=fa_mean[:9], yerr=fa_std[:9], color='k', linewidth=1.5)
# maxLL_ind = np.where(fa_mean[:9] == max(fa_mean[:9]))[0][0]
# maxLL = n_components[:9][maxLL_ind]
# plt.scatter(maxLL, fa_mean[:9][maxLL_ind], color='orange', s=100, edgecolor='k', zorder=3, label='Estimated Dimensionality')
# plt.axvline(maxLL, color='orange', ls='--', zorder=1)

# plt.title('4-Fold Cross-Validation FA')
# plt.ylabel('Log Likelihood')
# plt.xlabel('Dimensionality')
# plt.legend()
# plt.tight_layout()
# plt.savefig('/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/factor_analysis/figures/CV_FA_alldata_limited_LikelihoodPlot.png')
# plt.clf()

# plt.errorbar(x=n_components[:9], y=pca_mean[:9], yerr=pca_std[:9], color='k', linewidth=1.5)
# maxLL_ind = np.where(pca_mean[:9] == max(pca_mean[:9]))[0][0]
# maxLL = n_components[:9][maxLL_ind]
# plt.scatter(maxLL, pca_mean[:9][maxLL_ind], color='orange', s=100, edgecolor='k', zorder=3, label='Estimated Dimensionality')
# plt.axvline(maxLL, color='orange', ls='--', zorder=1)

# plt.title('4-Fold Cross-Validation PCA')
# plt.ylabel('Log Likelihood')
# plt.xlabel('Dimensionality')
# plt.legend()
# plt.tight_layout()
# plt.savefig('/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/factor_analysis/figures/CV_PCA_alldata_limited_LikelihoodPlot.png')
# plt.clf()
