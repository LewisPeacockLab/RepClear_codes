# category classifier with new ROIs
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
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneGroupOut,
)  # train_test_split, PredefinedSplit, cross_validate, cross_val_predict,
from sklearn.feature_selection import (
    SelectFpr,
    f_classif,
)  # VarianceThreshold, SelectKBest,
from sklearn.preprocessing import StandardScaler
from nilearn.input_data import NiftiMasker, MultiNiftiMasker
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from joblib import Parallel, delayed
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from typing import Tuple


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
    "026",
]


phases = ["rest", "preremoval", "study", "postremoval"]
runs = np.arange(6) + 1
spaces = {"T1w": "T1w", "MNI": "MNI152NLin2009cAsym"}
descs = ["brain_mask", "preproc_bold"]
# ROIs = ["Prefrontal_ROI", "Higher_Order_Visual_ROI"]
ROIs = ["Parahippocampal", "Fusiform"]
shift_sizes_TR = [5]

save = 1
shift_size_TR = shift_sizes_TR[0]


stim_labels = {0: "Rest", 1: "Scenes", 2: "Faces"}

workspace = "scratch"
data_dir = "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/"
param_dir = "/scratch/06873/zbretton/repclear_dataset/BIDS/params/"


# function to load in the confounds file for each run and then select the columns we want for cleaning
def confound_cleaner(confounds):
    COI = [
        "a_comp_cor_00",
        "a_comp_cor_01",
        "a_comp_cor_02",
        "a_comp_cor_03",
        "a_comp_cor_04",
        "a_comp_cor_05",
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


# this function takes the mask data and applies it to the bold data
def apply_mask(mask=None, target=None):
    coor = np.where(mask == 1)
    values = target[coor]
    if values.ndim > 1:
        values = np.transpose(values)  # swap axes to get feature X sample
    return values


def find(pattern, path):  # find the pattern we're looking for
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
        return result


def load_existing_data(subID, task, space, mask_ROIS, load=False):
    print("\n*** Attempting to load existing data if there is any...")
    preproc_data = {}
    todo_ROIs = []

    bold_dir = os.path.join(data_dir, f"sub-{subID}", "func")
    out_fname_template = f"sub-{subID}_{space}_{task}_{{}}_masked_cleaned.npy"

    for ROI in mask_ROIS:
        if load:
            if os.path.exists(os.path.join(bold_dir, out_fname_template.format(ROI))):
                print(
                    f"\nLoading saved preprocessed data {out_fname_template.format(ROI)}..."
                )
                preproc_data[ROI] = np.load(
                    os.path.join(bold_dir, out_fname_template.format(ROI))
                )
            else:
                print(f"\nROI {ROI} data to be processed.")
                todo_ROIs.append(ROI)
        else:
            print(f"\nROI {ROI} data to be processed.")
            todo_ROIs.append(ROI)

    return preproc_data, todo_ROIs


def load_process_data(
    subID, task, space, mask_ROIS
):  # this wraps the above function, with a process to load and save the data if there isnt an already saved .npy file
    # ========== check & load existing files
    ready_data, mask_ROIS = load_existing_data(subID, task, space, mask_ROIS)
    if type(mask_ROIS) == list and len(mask_ROIS) == 0:
        return np.vstack(list(ready_data.values()))
    else:
        print("\nPreprocessing ROIs", mask_ROIS)

    print(
        f"\n***** Data preprocessing for sub {subID} {task} {space} with ROIs {mask_ROIS}..."
    )

    space_long = spaces[space]

    if task == "study":
        runs = np.arange(3) + 1
    else:
        runs = np.arange(6) + 1

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

    # Generate mask names (specific to the new ROIs)
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

    # Load data files
    print("\n*** Loading & cleaning data...")
    bolds = [nib.load(p) for p in bold_paths]
    masks = [nib.load(p) for p in mask_paths]
    confounds = [pd.read_csv(p, sep="\t") for p in confound_paths]
    confounds_cleaned = [confound_cleaner(c) for c in confounds]

    # ===== for each run & ROI, mask & clean
    print("\n*** Masking & cleaing bold data...")
    # Mask & clean bold data
    cleaned_bolds = [[None for _ in range(len(runs))] for _ in range(len(mask_ROIS))]
    for rowi, mask in enumerate(masks):
        for coli, (bold, confound) in enumerate(zip(bolds, confounds_cleaned)):
            masked = apply_mask(mask=mask.get_fdata(), target=bold.get_fdata())
            cleaned_bolds[rowi][coli] = clean(
                masked, confounds=confound, t_r=1, detrend=False, standardize="zscore"
            )

    # Aggregate preprocessed data
    preproc_data = {
        ROI: np.vstack(run_data) for ROI, run_data in zip(mask_ROIS, cleaned_bolds)
    }

    print("processed data shape: ", [d.shape for d in preproc_data.values()])
    print("*** Done with preprocessing!")

    # Save for future use
    for ROI, run_data in preproc_data.items():
        bold_dir = os.path.join(data_dir, f"sub-{subID}", "func")
        out_fname = f"sub-{subID}_{space}_{task}_{ROI}_masked_cleaned.npy"
        np.save(os.path.join(bold_dir, out_fname), run_data)

    # Handle multiple ROIs and aggregate into one dictionary
    full_data = np.hstack(list(preproc_data.values()))
    return full_data


def get_shifted_labels(
    subID, task: str, shift_size_TR: int, rest_tag: int = 0
) -> pd.DataFrame:
    """
    Load labels and apply a temporal shift to account for the hemodynamic response delay.

    Parameters:
    - task (str): The task for which labels are being loaded.
    - shift_size_TR (int): The number of TRs by which to shift the labels.
    - rest_tag (int, optional): The tag to pad the labels with. Defaults to 0.

    Returns:
    - pd.DataFrame: The shifted labels.
    """

    def shift_timing(
        label_df: pd.DataFrame, TR_shift_size: int, tag: int = 0
    ) -> pd.DataFrame:
        nvars = len(label_df.loc[0])
        shift = pd.DataFrame(
            np.zeros((TR_shift_size, nvars)) + tag, columns=label_df.columns
        )
        shifted = pd.concat([shift, label_df])
        return shifted[: len(label_df)]

    print("\n***** Loading labels...")

    subject_design_dir = "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/subject_designs/"

    task_map = {
        "preremoval": "pre-localizer",
        "postremoval": "post-localizer",
        "study": "study",
    }
    temp_task = task_map.get(task, task)

    sub_design = f"*{subID}*{temp_task}*tr*"
    sub_design_files = find(sub_design, subject_design_dir)

    if len(sub_design_files) != 1:
        raise FileNotFoundError(
            f"Expected one design file, found {len(sub_design_files)}"
        )

    sub_design_matrix = pd.read_csv(sub_design_files[0])

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
            run_pair = [runi, runj]
            # choose scene samples based on runs in this subsample
            scene_inds = np.where(
                (stim_on == 1)
                & (stim_list == 1)
                & ((run_list == runi) | (run_list == runj))
            )[
                0
            ]  # actual stim; stim is scene; stim in the two runs

            if include_rest:
                print("Including resting category...")
                # get TR intervals for rest TRs between stims (stim_on == 2)
                rest_bools = ((run_list == runi) | (run_list == runj)) & (stim_on == 2)
                padded_bools = np.r_[
                    False, rest_bools, False
                ]  # pad the rest_bools 1 TR before and after to separate it from trial information
                rest_diff = np.diff(
                    padded_bools
                )  # get the pairwise diff in the array --> True for start and end indices of rest periods
                rest_intervals = rest_diff.nonzero()[0].reshape(
                    (-1, 2)
                )  # each pair is the interval of rest periods

                # get desired time points in the middle of rest periods for rest samples; if 0.5, round up
                rest_intervals[:, -1] -= 1
                rest_inds = [
                    np.ceil(np.average(interval)).astype(int)
                    for interval in rest_intervals
                ] + [
                    np.ceil(np.average(interval)).astype(int) + 1
                    for interval in rest_intervals
                ]

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
            yield X, Y, all_groups, run_pair

            # flip groups so even & odd groups can be paired
            all_groups = np.concatenate(
                [groups, list(reversed(groups)), list(reversed(groups))]
            )
            yield X, Y, all_groups, run_pair


def subsample_for_training(full_data, label_df, train_pairs, include_rest=True):
    """
    Subsample data by runs.
    Return: subsampled labels and bold data
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

    scenes_runs = train_pairs
    runi = scenes_runs[0]
    runj = scenes_runs[1]
    print(f"\nSubsampling scenes with runs {runi} & {runj}...")
    # choose scene samples based on runs in this subsample
    scene_inds = np.where(
        (stim_on == 1) & (stim_list == 1) & ((run_list == runi) | (run_list == runj))
    )[
        0
    ]  # actual stim; stim is scene; stim in the two runs

    if include_rest:
        print("Including resting category...")
        # get TR intervals for rest TRs between stims (stim_on == 2)
        rest_bools = ((run_list == runi) | (run_list == runj)) & (stim_on == 2)
        padded_bools = np.r_[
            False, rest_bools, False
        ]  # pad the rest_bools 1 TR before and after to separate it from trial information
        rest_diff = np.diff(
            padded_bools
        )  # get the pairwise diff in the array --> True for start and end indices of rest periods
        rest_intervals = rest_diff.nonzero()[0].reshape(
            (-1, 2)
        )  # each pair is the interval of rest periods

        # get desired time points in the middle of rest periods for rest samples; if 0.5, round up
        rest_intervals[:, -1] -= 1
        rest_inds = [
            np.ceil(np.average(interval)).astype(int) for interval in rest_intervals
        ] + [
            np.ceil(np.average(interval)).astype(int) + 1 for interval in rest_intervals
        ]

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
    return X, Y


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> Tuple:
    # Feature selection and transformation
    fpr = SelectFpr(f_classif, alpha=0.01).fit(X_train, y_train)
    X_train_sub = fpr.transform(X_train)

    # Train & hyperparam tuning
    parameters = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    gscv = GridSearchCV(
        LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000),
        parameters,
        return_train_score=True,
    )
    gscv.fit(X_train_sub, y_train)
    best_C = gscv.best_params_["C"]

    # Refit with full data and optimal penalty value
    lr = LogisticRegression(penalty="l2", solver="lbfgs", C=best_C, max_iter=1000)
    lr.fit(X_train_sub, y_train)

    return lr, fpr


def fit_model(X, Y, groups, save=False, out_fname=None, v=False):
    if v:
        print("\n***** Fitting model...")

    scores = []
    auc_scores = []
    cms = []
    trained_models = []
    fprs = []

    logo = LeaveOneGroupOut()
    for train_inds, test_inds in logo.split(X, Y, groups):
        X_train, X_test, y_train, y_test = (
            X[train_inds],
            X[test_inds],
            Y[train_inds],
            Y[test_inds],
        )

        # Train model and get feature selector
        trained_model, fpr = train_model(X_train, y_train)

        # Apply feature selection to test data
        X_test_sub = fpr.transform(X_test)

        # Test on held-out data
        score = trained_model.score(X_test_sub, y_test)
        auc_score = roc_auc_score(
            y_test, trained_model.predict_proba(X_test_sub), multi_class="ovr"
        )
        preds = trained_model.predict(X_test_sub)

        # Confusion matrix
        true_counts = np.asarray([np.sum(y_test == i) for i in stim_labels.keys()])
        cm = (
            confusion_matrix(y_test, preds, labels=list(stim_labels.keys()))
            / true_counts[:, None]
            * 100
        )

        scores.append(score)
        auc_scores.append(auc_score)
        cms.append(cm)
        trained_models.append(trained_model)
        fprs.append(fpr)

    # Package results
    results = {
        "scores": np.array(scores),
        "auc_scores": np.array(auc_scores),
        "cms": np.stack(cms),
        "trained_models": trained_models,
        "fprs": fprs,
    }

    if v:
        print(
            f"\nClassifier score: \n"
            f"scores: {np.mean(scores)} +/- {np.std(scores)}\n"
            f"auc scores: {np.mean(auc_scores)} +/- {np.std(auc_scores)}\n"
            f"average confusion matrix:\n"
            f"{np.mean(np.stack(cms), axis=0)}"
        )

    return results


def decode(training_runs, train_data, train_labels, test_data, test_labels):
    print(f"Running model fitting...")

    X_train, y_train = subsample_for_training(
        train_data, train_labels, training_runs, include_rest=True
    )
    X_test, y_test = test_data, test_labels["category"].values

    # Train model and get feature selector
    trained_model, fpr = train_model(X_train, y_train)

    # Apply feature selection to test data
    X_test_sub = fpr.transform(X_test)

    # Test on held-out data
    predictions = trained_model.predict(X_test_sub)
    evidence = 1.0 / (1.0 + np.exp(-trained_model.decision_function(X_test_sub)))

    return predictions, evidence, y_test


def classification(subID):
    # Initialize variables
    task = "preremoval"
    task2 = "study"
    space = "T1w"
    ROIs = ["Parahippocampal", "Fusiform"]
    n_iters = 1
    shift_size_TR = shift_sizes_TR[0]
    rest_tag = 0

    for roi in ROIs:
        print(
            f"\n***** Running category level classification for sub {subID} {task} {space} with ROI {roi}..."
        )

        # Load data and labels for the first task
        full_data = load_process_data(subID, task, space, [roi])
        print(f"Full_data shape: {full_data.shape}")

        label_df = get_shifted_labels(subID, task, shift_size_TR, rest_tag)
        print(f"Category label shape: {label_df.shape}")

        # Consistency check
        assert len(full_data) == len(
            label_df
        ), f"Length of data ({len(full_data)}) does not match Length of labels ({len(label_df)})!"

        # Initialize lists to store metrics
        scores = []
        auc_scores = []
        cms = []
        trained_models = []
        fprs = []
        run_pairs = []

        # Cross-run cross-validation
        for X, Y, groups, run_pair in subsample_by_runs(full_data, label_df):
            print(f"Running model fitting...")
            print("shape of X & Y:", X.shape, Y.shape)
            assert len(X) == len(
                Y
            ), f"Length of X ({len(X)}) doesn't match length of Y({len(Y)})"

            results = fit_model(X, Y, groups, save=False, v=True)
            score = results["scores"]
            auc_score = results["auc_scores"]
            cm = results["cms"]
            trained_model = results["trained_models"]
            fpr = results["fprs"]

            scores.append(score)
            auc_scores.append(auc_score)
            cms.append(cm)
            trained_models.append(trained_model)
            fprs.append(fpr)
            run_pairs.append(run_pair)

        # Average scores and AUCs
        scores = np.mean(scores, axis=1)
        auc_scores = np.mean(auc_scores, axis=1)
        cms = np.stack(cms)

        # Select the best model
        best_model = np.where(auc_scores == max(auc_scores))[0][0]
        train_pairs = run_pairs[best_model]

        print(
            f"\n***** Running category level classification for sub {subID} {task2} {space} with ROI {roi}..."
        )

        # Load data and labels for the second task
        full_data2 = load_process_data(subID, task2, space, [roi])
        print(f"Full_data shape: {full_data2.shape}")

        label_df2 = get_shifted_labels(subID, task2, shift_size_TR, rest_tag)
        print(f"Category label shape: {label_df2.shape}")

        # Decode the second task
        predicts, evidence, true = decode(
            train_pairs, full_data, label_df, full_data2, label_df2
        )

        # Save evidence
        evidence_df = pd.DataFrame(data=label_df2)
        evidence_df.drop(columns=evidence_df.columns[0], axis=1, inplace=True)
        evidence_df["rest_evi"] = evidence[:, 0]
        evidence_df["scene_evi"] = evidence[:, 1]
        evidence_df["face_evi"] = evidence[:, 2]

        sub_dir = os.path.join(data_dir, f"sub-{subID}")
        out_fname_template = (
            f"sub-{subID}_{space}_trained-{task}_tested-{task2}_{roi}_evidence.csv"
        )
        print("\n *** Saving evidence values with subject dataframe ***")
        evidence_df.to_csv(os.path.join(sub_dir, out_fname_template))


def organize_evidence(subID, save=True):
    task = "preremoval"
    task2 = "study"
    space = "T1w"
    ROIs = ["Parahippocampal", "Fusiform"]

    for roi in ROIs:
        print("\n *** loading evidence values from subject dataframe ***")

        sub_dir = os.path.join(data_dir, f"sub-{subID}")
        in_fname_template = (
            f"sub-{subID}_{space}_trained-{task}_tested-{task2}_{roi}_evidence.csv"
        )

        sub_df = pd.read_csv(os.path.join(sub_dir, in_fname_template))
        sub_df.drop(
            columns=sub_df.columns[0], axis=1, inplace=True
        )  # now drop the extra index column

        sub_images, sub_index = np.unique(
            sub_df["image_id"], return_index=True
        )  # this searches through the dataframe to find each occurance of the image_id. This allows me to find the start of each trial, and linked to the image_ID #

        # now to sort the trials, we need to figure out what the operation performed is:
        sub_condition_list = sub_df["condition"][sub_index].values.astype(
            int
        )  # so using the above indices, we will now grab what the condition is of each image

        counter = 0
        maintain_trials = {}
        replace_trials = {}
        suppress_trials = {}

        for i in sub_condition_list:
            if (
                i == 0
            ):  # this is the first rest period (because we had to shift of hemodynamics. So this "0" condition is nothing)
                print("i==0")
                counter += 1
                continue
            elif i == 1:
                temp_image = sub_images[counter]
                maintain_trials[temp_image] = sub_df[
                    ["rest_evi", "scene_evi", "face_evi"]
                ][sub_index[counter] - 5 : sub_index[counter] + 9].values
                counter += 1

            elif i == 2:
                temp_image = sub_images[counter]
                replace_trials[temp_image] = sub_df[
                    ["rest_evi", "scene_evi", "face_evi"]
                ][sub_index[counter] - 5 : sub_index[counter] + 9].values
                counter += 1

            elif i == 3:
                temp_image = sub_images[counter]
                suppress_trials[temp_image] = sub_df[
                    ["rest_evi", "scene_evi", "face_evi"]
                ][sub_index[counter] - 5 : sub_index[counter] + 9].values
                counter += 1

        # now that the trials are sorted, we need to get the subject average for each condition:
        avg_maintain = pd.DataFrame(
            data=np.dstack(maintain_trials.values()).mean(axis=2)
        )
        avg_replace = pd.DataFrame(data=np.dstack(replace_trials.values()).mean(axis=2))
        avg_suppress = pd.DataFrame(
            data=np.dstack(suppress_trials.values()).mean(axis=2)
        )

        # now I will have to change the structure to be able to plot in seaborn:
        avg_maintain = (
            avg_maintain.T.melt()
        )  # now you get 2 columns: variable (TR) and value (evidence)
        avg_maintain["sub"] = np.repeat(
            subID, len(avg_maintain)
        )  # input the subject so I can stack melted dfs
        avg_maintain["evidence_class"] = np.tile(
            ["rest", "scenes", "faces"], 14
        )  # add in the labels so we know what each data point is refering to
        avg_maintain.rename(
            columns={"variable": "TR", "value": "evidence"}, inplace=True
        )  # renamed the melted column names
        avg_maintain[
            "condition"
        ] = "maintain"  # now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

        avg_replace = (
            avg_replace.T.melt()
        )  # now you get 2 columns: variable (TR) and value (evidence)
        avg_replace["sub"] = np.repeat(
            subID, len(avg_replace)
        )  # input the subject so I can stack melted dfs
        avg_replace["evidence_class"] = np.tile(
            ["rest", "scenes", "faces"], 14
        )  # add in the labels so we know what each data point is refering to
        avg_replace.rename(
            columns={"variable": "TR", "value": "evidence"}, inplace=True
        )  # renamed the melted column names
        avg_replace[
            "condition"
        ] = "replace"  # now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

        avg_suppress = (
            avg_suppress.T.melt()
        )  # now you get 2 columns: variable (TR) and value (evidence)
        avg_suppress["sub"] = np.repeat(
            subID, len(avg_suppress)
        )  # input the subject so I can stack melted dfs
        avg_suppress["evidence_class"] = np.tile(
            ["rest", "scenes", "faces"], 14
        )  # add in the labels so we know what each data point is refering to
        avg_suppress.rename(
            columns={"variable": "TR", "value": "evidence"}, inplace=True
        )  # renamed the melted column names
        avg_suppress[
            "condition"
        ] = "suppress"  # now I want to add in a condition label, since I can then stack all 3 conditions into 1 array per subject

        avg_subject_df = pd.concat(
            [avg_maintain, avg_replace, avg_suppress], ignore_index=True, sort=False
        )

        # save for future use
        if save:
            sub_dir = os.path.join(data_dir, f"sub-{subID}")
            out_fname_template = (
                f"sub-{subID}_{space}_{task2}_{roi}_evidence_dataframe.csv"
            )
            print(
                f"\n Saving the sorted {roi} evidence dataframe for {subID} - phase: {task2} - as {out_fname_template}"
            )
            avg_subject_df.to_csv(os.path.join(sub_dir, out_fname_template))


for sub in subIDs:
    classification(sub)

for sub in subIDs:
    organize_evidence(sub)


####### analysis and visualize #######
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
    "026",
]
task = "study"
space = "T1w"
rois = ["Parahippocampal", "Fusiform"]
data_dir = "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/"


def aggregate_data(subIDs, task, space, rois, data_dir):
    group_data = pd.DataFrame()
    for subID in subIDs:
        for roi in rois:
            file_path = os.path.join(
                data_dir,
                f"sub-{subID}",
                f"sub-{subID}_{space}_{task}_{roi}_evidence_dataframe.csv",
            )
            sub_df = pd.read_csv(file_path)
            sub_df["subID"] = subID  # Add subject ID column
            sub_df["ROI"] = roi  # Add ROI column
            group_data = pd.concat([group_data, sub_df], ignore_index=True)
    return group_data


def normalize_evidence(group_data, baseline_condition="maintain", baseline_TRs=(0, 1)):
    # Calculate baseline (mean evidence of the first two TRs in maintain condition)
    baseline = (
        group_data[
            (group_data["condition"] == baseline_condition)
            & (group_data["TR"].isin(baseline_TRs))
        ]
        .groupby(["subID", "ROI", "evidence_class"])["evidence"]
        .mean()
        .reset_index()
    )

    # Normalize evidence by subtracting the baseline
    normalized_data = pd.merge(
        group_data,
        baseline,
        how="left",
        left_on=["subID", "ROI", "evidence_class"],
        right_on=["subID", "ROI", "evidence_class"],
    )
    normalized_data["normalized_evidence"] = (
        normalized_data["evidence_x"] - normalized_data["evidence_y"]
    )
    return normalized_data


def visualize_normalized_evidence(updated_data, rois):
    for roi in rois:
        roi_data = updated_data[
            (updated_data["ROI"] == roi) & (updated_data["evidence_class"] != "rest")
        ]

        # Setting up the plot
        plt.figure(figsize=(10, 6))

        # Plotting scene evidence for all conditions
        sns.lineplot(
            data=roi_data[roi_data["evidence_class"] == "scenes"],
            x="TR",
            y="normalized_evidence",
            hue="condition",
            ci="sd",
            palette={"maintain": "green", "replace": "blue", "suppress": "red"},
        )

        # Plotting face evidence only for replace trials
        replace_face_data = roi_data[
            (roi_data["condition"] == "replace")
            & (roi_data["evidence_class"] == "faces")
        ]
        sns.lineplot(
            data=replace_face_data,
            x="TR",
            y="normalized_evidence",
            color="black",
            linestyle="--",
            label="replace (face evidence)",
        )

        plt.title(f"Normalized Evidence Trajectory in {roi}")
        plt.xlabel("Time (TR)")
        plt.ylabel("Normalized Evidence")
        plt.legend(
            title="Condition & Evidence Class",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )
        plt.tight_layout()
        plt.savefig(f"normalized_evidence_trajectory_{roi}.png", dpi=300)
        plt.show()


def visualize_evidence(group_data, rois):
    for roi in rois:
        # Filter data for the current ROI and exclude 'rest' class
        roi_data = group_data[
            (group_data["ROI"] == roi) & (group_data["evidence_class"] == "scenes")
        ]

        # Calculate mean rest evidence across conditions for the current ROI
        mean_rest_evidence = (
            group_data[
                (group_data["ROI"] == roi) & (group_data["evidence_class"] == "rest")
            ]
            .groupby(["TR", "subID"])["evidence"]
            .mean()
            .reset_index()
        )

        # Setting up the plot
        plt.figure(figsize=(10, 6))

        sns.lineplot(
            data=mean_rest_evidence,
            x="TR",
            y="evidence",
            color="gray",
            linestyle="--",
            ci=95,
            label="rest",
        )

        # Plotting face evidence only for replace trials
        replace_face_data = group_data[
            (group_data["ROI"] == roi)
            & (group_data["condition"] == "replace")
            & (group_data["evidence_class"] == "faces")
        ]

        sns.lineplot(
            data=replace_face_data,
            x="TR",
            y="evidence",
            color="black",
            linestyle="--",
            ci=95,
            label="replace (face evidence)",
        )

        # Plotting scene evidence for all conditions
        sns.lineplot(
            data=roi_data,
            x="TR",
            y="evidence",
            hue="condition",
            ci=95,
            palette={"maintain": "green", "replace": "blue", "suppress": "red"},
        )

        plt.title(f"Evidence Trajectory in {roi}")
        plt.xlabel("Time (TR)")
        plt.ylabel("Evidence")
        plt.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"evidence_trajectory_{roi}.png", dpi=300)
        plt.show()


# Call the function with the group_data DataFrame and the list of ROIs you're interested in

group_data = aggregate_data(subIDs, task, space, rois, data_dir)

# Normalize evidence
normalized_data = normalize_evidence(group_data)

# Visualize normalized evidence trajectories
visualize_normalized_evidence(normalized_data, rois)

visualize_evidence(group_data, ["Parahippocampal", "Fusiform"])
