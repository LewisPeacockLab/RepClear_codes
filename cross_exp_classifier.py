import warnings
import sys

if not sys.warnoptions:
    warnings.simplefilter("ignore")
import os
import glob

import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt

import nibabel as nib
from nilearn.image import clean_img
from nilearn.signal import clean

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.feature_selection import SelectFpr, f_classif
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay


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

clearmem_dir = "/scratch/06873/zbretton/clearmem/"
workspace = "scratch"
data_dir = "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/"
param_dir = "/scratch/06873/zbretton/repclear_dataset/BIDS/params/"
results_dir = "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/model_fitting_results/"


def get_preprocessed_data(
    projID,
    subID,
    task,
    space,
    mask_path,
    data_tag="group-GM",
    runs=np.arange(6) + 1,
    save=False,
):
    """
    Generic data loading & cleaning for both repclear & clearmem
    Inputs:
    mask_path: paths to masks shared between project bold, assuming all ROIs are included
    """

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

    print(
        f"\n***** Data preprocessing for proj {projID} sub {subID} ({max(runs)} runs) {task} {space} with mask {mask_path}..."
    )

    if projID == "repclear":
        data_dir = repclear_dir
        desc = "preproc_bold"

    elif projID == "clearmem":
        data_dir = clearmem_dir
        desc = "preproc_resized_bold"

    # ===== generate output name & load existing
    out_fname = f"sub-{subID}_{space}_{task}_{data_tag}_masked_cleaned.npy"
    out_dir = os.path.join(data_dir, f"sub-{subID}", "func", "masked_cleaned")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_fname)
    if os.path.exists(out_path):
        print(f"Loading from existing {out_path}")
        return np.load(out_path)

    # ===== start from scratch
    # load bold
    print("Loading BOLD data...")
    fname_template = (
        f"sub-{subID}_task-{task}_run-{{}}_space-{spaces[space]}_desc-{{}}.nii.gz"
    )
    try:
        bold_fnames = [fname_template.format(i, desc) for i in runs]
        bold_paths = [
            os.path.join(data_dir, f"sub-{subID}", "func", fname)
            for fname in bold_fnames
        ]
        # shape: ((x/y/z) x time)
        bolds = [nib.load(p) for p in bold_paths]
    except:
        print("No resized data. Using original voxel size for clearmem...")
        desc = "preproc_bold"
        bold_fnames = [fname_template.format(i, desc) for i in runs]
        bold_paths = [
            os.path.join(data_dir, f"sub-{subID}", "func", fname)
            for fname in bold_fnames
        ]
        # shape: ((x/y/z) x time)
        bolds = [nib.load(p) for p in bold_paths]

    # load mask
    print("Loading mask...")
    mask = nib.load(mask_path)

    # load confounds
    print("Loading confound files...")
    confound_fnames = [f"*{task}*{run}*confounds*.tsv" for run in runs]
    confound_paths = [
        os.path.join(data_dir, f"sub-{subID}", "func", f) for f in confound_fnames
    ]  # template for each run
    confound_paths = [glob.glob(p)[0] for p in confound_paths]  # actual file names
    print(f"glob check: {len(confound_paths)} files to be loaded")
    confounds = [pd.read_csv(p, sep="\t") for p in confound_paths]
    confounds_cleaned = [confound_cleaner(c) for c in confounds]

    # mask & clean
    cleaned_bolds = [None for _ in range(len(runs))]
    # all files are by nruns
    for runi, (bold, confound) in enumerate(zip(bolds, confounds_cleaned)):
        print(f"Processing run {runi}...")
        masked = apply_mask(mask=mask.get_data(), target=bold.get_data())
        # *** clean: confound length in time; transpose signal to (time x vox) for cleaning & model fitting ***
        cleaned_bolds[runi] = clean(
            masked.T, confounds=confound, t_r=1, detrend=False, standardize="zscore"
        )
        print("cleaned shape: ", cleaned_bolds[runi].shape)

    # put together
    # ROI: time x vox
    preproc_data = np.vstack(cleaned_bolds)
    print("processed data shape: ", preproc_data.shape)
    print("*** Done with preprocessing!")

    # save for future use
    if save:
        print(f"Saving to file {out_path}...")
        np.save(f"{out_path}", preproc_data)

    return preproc_data


def get_shifted_labels(projID, task, shift_size_TR, rest_tag=0):
    # load labels, & add hemodynamic shift to all vars

    def shift_timing(label_df, TR_shift_size, tag=0):
        # Shift 2D df labels by given TRs with paddings of tag
        # Input label_df must be time x nvars
        nvars = len(label_df.loc[0])
        shift = pd.DataFrame(
            (np.zeros((TR_shift_size, nvars)) + tag).astype(int),
            columns=label_df.columns,
        )
        shifted = pd.concat([shift, label_df])
        return shifted[: len(label_df)]  # trim time points outside of scanning time

    print(f"\n***** Loading labels for {task} with {shift_size_TR} shift...")

    if projID == "repclear":
        event_path = os.path.join(param_dir, f"{task}_events.csv")
    elif projID == "clearmem":
        event_path = os.path.join(clearmem_dir, f"{task}_volume_design_matrix.csv")

    events_df = pd.read_csv(event_path)

    shifted_df = shift_timing(events_df, shift_size_TR, rest_tag)

    return shifted_df


def subsample(
    projID, full_data, label_df, include_iti=False, task="study", include_rest=False
):
    print(f"\n***** Subsampling data points with include_iti {include_iti}...")

    if task == "study":
        if include_rest:
            print(
                "\n***WARNING: does not support include_rest == True for opeation decoding yet***\n"
            )
        # "condition":
        # repclear: 1. maintain, 2. replace_category, 3. suppress
        # clearmem: 1 - maintain, 2 - replace_category, 4 - suppress   # ignore: 0 - rest, 3 - replace_subcategory, 5 - clear
        # "presentation / stim_present":
        # repclear: 1 - stim, 2 - operation, 3 - ITI
        # clearmem: 1 - stim, 2 - operation, 0 - ITI/rest
        condition = label_df["condition"]

        # proj diff in stim_on & condition
        if projID == "repclear":
            stim_on = label_df["stim_present"]
            op_filter = condition != 0
        elif projID == "clearmem":
            stim_on = label_df["presentation"]
            op_filter = (condition == 1) | (condition == 2) | (condition == 4)

        # whether to include ITIs
        if include_iti:
            stim_on_filter = stim_on != 1  # exclude stim pres
        else:
            stim_on_filter = stim_on == 2  # include only operation

        final_filter = op_filter & stim_on_filter
        X = full_data[final_filter, :]
        Y_all_df = label_df[final_filter]  # condition[final_filter].values
        # subID_list = label_df["subID"][final_filter].values

        # make sure labels are the same
        if projID == "clearmem":
            conds = Y_all_df["condition"]
            conds[conds == 4] = 3
            Y_all_df["condition"] = conds

        print("Category counts after subsampling:")
        for opi, op in op_labels.items():
            op_bools = Y_all_df["condition"] == opi
            print(f"{op}: {sum(op_bools)}")

    elif task == "localizer":
        assert (
            projID == "clearmem"
        ), "Use classification_replicate.subsample_by_runs for repclear cate data loading"
        # cateory: 0 - rest, 1 - face, 2 - fruit, 3 - scene
        category = label_df["category"]
        stim_on = label_df["stimulus"]

        cate_filter = category != 0
        stim_on_filter = stim_on == 1
        rest_filter = np.array([False for _ in range(len(label_df))])  # place holder

        if include_rest:
            run = label_df["run"]
            runs = sorted(set(run))
            runs.remove(0)
            rest_inds = []
            # for each run, subsample to same number of categories
            for runi in runs:
                cate_n = sum((run == runi) & (category == 1) & (stim_on == 1))
                run_rest_inds = np.where((run == runi) & (category == 0))[0]
                chosen_inds = np.random.choice(run_rest_inds, cate_n, replace=False)
                rest_inds.append(sorted(chosen_inds))
            rest_inds = np.concatenate(rest_inds).astype(int)
            rest_filter[rest_inds] = True

        final_filter = (cate_filter & stim_on_filter) | rest_filter
        X = full_data[final_filter, :]
        Y_all_df = label_df[final_filter]

        print("Category counts after subsampling:")
        for catei, cate in stim_labels.items():
            cate_bools = Y_all_df["category"] == catei
            print(f"{cate}: {sum(cate_bools)}")

    return X, Y_all_df


def cross_exp_classification():
    # consts
    clearmem_subIDs = [
        "004",
        "006",
        "011",
        "015",
        "027",
        "034",
        "036",
        "042",
        "044",
        "045",
        #    "050", "061", "069", "077", "079",
    ]
    repclear_subIDs = subIDs = [
        "002",
        "003",
        "004",
        "005",
        "006",
        "007",
        "008",
        "009",
        "010",
    ]
    mask_path = os.path.join(repclear_dir, "group_MNI_GM_mask.nii.gz")
    clearmem_shift_size_TR = 10
    repclear_shift_size_TR = 5

    task = "study"
    space = "MNI"
    rest_tag = 0
    include_iti = True

    # ===== load clearmem data & labels
    # load training data: cleaned & masked
    # study phase: 6 runs each sub
    X_train = []
    clearmem_label_df = []
    for subID in clearmem_subIDs:
        # bold
        X_train.append(
            get_preprocessed_data(
                "clearmem",
                subID,
                task,
                space,
                mask_path,
                runs=np.arange(6) + 1,
                save=True,
            )
        )
        print(X_train[-1].shape)

        # labels
        sub_label_df = get_shifted_labels(
            "clearmem", task, clearmem_shift_size_TR, rest_tag
        )
        sub_label_df["subID"] = np.array([subID for _ in range(len(sub_label_df))])
        clearmem_label_df.append(sub_label_df)
        print(clearmem_label_df[-1].shape)

    X_train = np.vstack(X_train)
    clearmem_label_df = pd.concat(clearmem_label_df)

    print(f"\nX_train shape: {X_train.shape}")
    print(f"Category label shape: {clearmem_label_df.shape}")

    # subsample clearmem data
    #    to match repclear operations
    #    to balance operation counts
    X_train, y_train_df = subsample(
        "clearmem", X_train, clearmem_label_df, include_iti=include_iti
    )
    y_train = y_train_df["condition"].values
    train_subID_list = y_train_df["subID"]
    print(
        "\nShape after subsample: \n"
        f"X_train: {X_train.shape}, y_train: {y_train.shape}, train_subID_list: {train_subID_list.shape}"
    )

    # ===== load repclear data & labels
    X_test = []
    repclear_label_df = []
    for subID in repclear_subIDs:
        X_test.append(
            get_preprocessed_data(
                "repclear",
                subID,
                task,
                space,
                mask_path,
                runs=np.arange(3) + 1,
                save=True,
            )
        )
        print(X_test[-1].shape)

        sub_label_df = get_shifted_labels(
            "repclear", task, repclear_shift_size_TR, rest_tag
        )
        sub_label_df["subID"] = np.array([subID for _ in range(len(sub_label_df))])
        repclear_label_df.append(sub_label_df)
        print(repclear_label_df[-1].shape)

    X_test = np.vstack(X_test)
    repclear_label_df = pd.concat(repclear_label_df)

    # subsample
    X_test, y_test_df = subsample(
        "repclear", X_test, repclear_label_df, include_iti=include_iti
    )
    y_test = y_test_df["condition"]
    test_subID_list = y_test_df["subID"]
    print(
        "\nShape after subsample: \n"
        f"X_test: {X_test.shape}, y_test: {y_test.shape}, test_subID_list: {test_subID_list.shape}"
    )

    # ===== train: put them here for now
    print("\nFeature selection...")
    # feature selection
    fpr = SelectFpr(f_classif, alpha=0.01).fit(X_train, y_train)
    X_train_sub = fpr.transform(X_train)
    X_test_sub = fpr.transform(X_test)
    print(
        f"Shapes after feature selection: X_train_sub: {X_train_sub.shape}, X_test_sub: {X_test_sub.shape}"
    )

    print("\nTraining model...")
    print(f"Data shapes: X_train_sub: {X_train_sub.shape}, y_train: {y_train.shape}")
    lr = LogisticRegression(penalty="l2", solver="liblinear", C=1, n_jobs=32, verbose=5)
    lr.fit(X_train_sub, y_train)

    # parameters ={'C':[0.01,0.1,1,10,100,1000]}
    # logo = LeaveOneGroupOut()
    # gscv = GridSearchCV(
    #     LogisticRegression(penalty='l2', solver='liblinear'),
    #     parameters,
    #     cv=logo, return_train_score=True, verbose=5, n_jobs=8)
    # gscv.fit(X_train_sub, y_train, groups=train_subID_list)
    # lr = gscv

    # lr = np.load('/scratch1/07365/sguo19/model_fitting_results/operation_clf/cross_expt_lr.npz', allow_pickle=True)["lr"][0]

    # test
    print("Testing model accuracy...")
    scores = []
    auc_scores = []
    cms = []
    for test_subID in repclear_subIDs:
        print(f"Testing sub{test_subID}...")
        X_test_sub_data = X_test_sub[test_subID_list == test_subID]
        y_test_sub_data = y_test[test_subID_list == test_subID]
        print(X_test_sub_data)
        print(y_test_sub_data)

        score = lr.score(X_test_sub_data, y_test_sub_data)
        auc_score = roc_auc_score(
            y_test_sub_data, lr.predict_proba(X_test_sub_data), multi_class="ovr"
        )
        preds = lr.predict(X_test_sub_data)

        # confusion matrix
        true_counts = np.asarray(
            [np.sum(y_test_sub_data == i) for i in op_labels.keys()]
        )
        cm = (
            confusion_matrix(y_test_sub_data, preds, labels=list(op_labels.keys()))
            / true_counts[:, None]
            * 100
        )

        print(
            f"\nSub{test_subID} result: \n"
            f"score: {score}, auc score: {auc_score}\n"
            f"confution matrix:\n"
            f"{cm}"
        )

        scores.append(score)
        auc_scores.append(auc_score)
        cms.append(cm)

    scores = np.asarray(scores)
    auc_scores = np.asarray(auc_scores)
    cms = np.stack(cms)

    print(
        f"\nClassifier score: \n"
        f"scores: {scores.mean()} +/- {scores.std()}\n"
        f"auc scores: {auc_scores.mean()} +/- {auc_scores.std()}\n"
        f"average confution matrix:\n"
        f"{cms.mean(axis=0)}"
    )

    # save
    out_path = os.path.join(results_dir, "operation_clf", f"cross_expt_lr_include-ITI")
    print(f"Saving results to {out_path}...")
    np.savez_compressed(out_path, lr=lr, scores=scores, auc_scores=auc_scores, cms=cms)


if __name__ == "__main__":
    cross_exp_classification()
