# This code is to load in the representations and then either weight them (category vs. item) and then perform RSA
# This will also handle performing this and then comparing Pre-Localizer to Study, so that we can see same vs. other

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
import glob
import fnmatch
import pandas as pd
import pickle
import re
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import (
    PredefinedSplit,
    cross_validate,
    cross_val_predict,
    GridSearchCV,
    LeaveOneGroupOut,
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
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
from statannot import add_stat_annotation
from statsmodels.stats.anova import AnovaRM


subs = [
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "20",
    "23",
    "24",
    "25",
    "26",
]
brain_flag = "MNI"
stim_labels = {0: "Rest", 1: "Scenes", 2: "Faces"}
sub_cates = {
    "scene": ["manmade", "natural"],  # 120
}


def mkdir(path, local=False):
    if not os.path.exists(path):
        os.makedirs(path)


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


container_path = "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep"
param_dir = (
    "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/subject_designs"
)

# the subject's list of image number to trial numbers are in the "subject_designs" folder

for subID in subs:
    print("Running sub-0%s..." % subID)
    # define the subject
    sub = "sub-0%s" % subID

    # lets pull out the pre-localizer data here:
    tim_path = os.path.join(param_dir, f"sub-0{subID}_trial_image_match.csv")

    tim_df = pd.read_csv(tim_path)
    tim_df = tim_df[tim_df["phase"] == 2]  # phase 2 is pre-localizer
    tim_df = tim_df.sort_values(by=["category", "subcategory", "trial_id"])

    pre_scene_order = tim_df[tim_df["category"] == 1][
        ["trial_id", "image_id", "condition", "subcategory"]
    ]
    pre_face_order = tim_df[tim_df["category"] == 2][
        ["trial_id", "image_id", "condition"]
    ]

    # lets pull out the study data here:
    tim_df2 = pd.read_csv(tim_path)
    tim_df2 = tim_df2[tim_df2["phase"] == 3]  # phase 3 is study
    tim_df2 = tim_df2.sort_values(by=["category", "subcategory", "trial_id"])

    study_scene_order = tim_df2[tim_df2["category"] == 1][
        ["trial_id", "image_id", "condition", "subcategory"]
    ]

    print(f"Running RSA for sub-0{subID}...")

    # ===== load mask for BOLD
    if brain_flag == "MNI":
        mask_path = os.path.join(container_path, "group_MNI_VTC_mask.nii.gz")
    else:
        mask_path = os.path.join(
            "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/",
            sub,
            "new_mask",
            "VVS_preremoval_%s_mask.nii.gz" % brain_flag,
        )

    mask = nib.load(mask_path)

    print("mask shape: ", mask.shape)

    # ===== load ready BOLD for each trial of prelocalizer
    print(f"Loading preprocessed BOLDs for pre-localizer...")
    bold_dir_1 = os.path.join(
        container_path, f"sub-0{subID}", "item_representations_%s" % brain_flag
    )

    all_bolds_1 = {}  # {cateID: {trialID: bold}}
    bolds_arr_1 = []  # sample x vox
    for cateID in sub_cates.keys():
        cate_bolds_fnames_1 = glob.glob(f"{bold_dir_1}/*pre*{cateID}*")
        cate_bolds_1 = {}

        for fname in cate_bolds_fnames_1:
            trialID = fname.split("/")[-1].split("_")[-2]  # "trial1"
            trialID = int(trialID[5:])
            cate_bolds_1[trialID] = nib.load(fname).get_fdata()  # .flatten()
        cate_bolds_1 = {i: cate_bolds_1[i] for i in sorted(cate_bolds_1.keys())}
        all_bolds_1[cateID] = cate_bolds_1

        bolds_arr_1.append(
            np.stack([cate_bolds_1[i] for i in sorted(cate_bolds_1.keys())])
        )

    bolds_arr_1 = np.vstack(bolds_arr_1)
    print("bolds for prelocalizer - shape: ", bolds_arr_1.shape)

    # ===== load ready BOLD for each trial of study
    print(f"Loading preprocessed BOLDs for the study operation...")
    bold_dir_2 = os.path.join(
        container_path, f"sub-0{subID}", "item_representations_%s" % brain_flag
    )

    all_bolds_2 = {}  # {cateID: {trialID: bold}}
    bolds_arr_2 = []  # sample x vox
    for cateID in sub_cates.keys():
        cate_bolds_fnames_2 = glob.glob(f"{bold_dir_2}/*study_{cateID}*")
        cate_bolds_2 = {}
        try:
            for fname in cate_bolds_fnames_2:
                trialID = fname.split("/")[-1].split("_")[-2]  # "trial1"
                trialID = int(trialID[5:])
                cate_bolds_2[trialID] = nib.load(fname).get_fdata()  # .flatten()
            cate_bolds_2 = {i: cate_bolds_2[i] for i in sorted(cate_bolds_2.keys())}
            all_bolds_2[cateID] = cate_bolds_2

            bolds_arr_2.append(
                np.stack([cate_bolds_2[i] for i in sorted(cate_bolds_2.keys())])
            )
        except:
            print("no %s trials" % cateID)
    bolds_arr_2 = np.vstack(bolds_arr_2)
    print("bolds for study - shape: ", bolds_arr_2.shape)

    # apply VTC mask on prelocalizer BOLD
    masked_bolds_arr_1 = []
    for bold in bolds_arr_1:
        masked_bolds_arr_1.append(
            apply_mask(mask=mask.get_fdata(), target=bold).flatten()
        )
    masked_bolds_arr_1 = np.vstack(masked_bolds_arr_1)
    print("masked prelocalizer bold array shape: ", masked_bolds_arr_1.shape)

    # apply mask on study BOLD
    masked_bolds_arr_2 = []
    for bold in bolds_arr_2:
        masked_bolds_arr_2.append(
            apply_mask(mask=mask.get_fdata(), target=bold).flatten()
        )
    masked_bolds_arr_2 = np.vstack(masked_bolds_arr_2)
    print("masked study bold array shape: ", masked_bolds_arr_2.shape)

    # ===== load weights
    print(f"Loading weights...")
    # prelocalizer
    if brain_flag == "MNI":
        cate_weights_dir = os.path.join(
            f"/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/sub-0{subID}",
            "preremoval_lvl1_%s/scene_stimuli_MNI_zmap.nii.gz" % brain_flag,
        )
        item_weights_dir = os.path.join(
            container_path, f"sub-0{subID}", "preremoval_item_level_MNI"
        )
    else:
        cate_weights_dir = os.path.join(
            f"/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/sub-0{subID}",
            "preremoval_lvl1_%s/scene_stimuli_T1w_zmap.nii.gz" % brain_flag,
        )
        item_weights_dir = os.path.join(
            container_path, f"sub-0{subID}", "preremoval_item_level_T1w"
        )

    # prelocalizer weights (category and item) get applied to study/post representations

    all_weights = {}
    weights_arr = []

    # load in all the item specific weights, which come from the LSA contrasts per subject
    for cateID in sub_cates.keys():
        item_weights_fnames = glob.glob(f"{item_weights_dir}/{cateID}*full*zmap*")
        print(cateID, len(item_weights_fnames))
        item_weights = {}

        for fname in item_weights_fnames:
            trialID = fname.split("/")[-1].split("_")[1]  # "trial1"
            trialID = int(trialID[5:])
            item_weights[trialID] = nib.load(fname).get_fdata()
        item_weights = {i: item_weights[i] for i in sorted(item_weights.keys())}
        all_weights[cateID] = item_weights
        weights_arr.append(
            np.stack([item_weights[i] for i in sorted(item_weights.keys())])
        )

    # this now masks the item weights to ensure that they are all in the same ROI (group VTC):
    weights_arr = np.vstack(weights_arr)
    print("weights shape: ", weights_arr.shape)
    # apply mask on BOLD
    masked_weights_arr = []
    for weight in weights_arr:
        masked_weights_arr.append(
            apply_mask(mask=mask.get_fdata(), target=weight).flatten()
        )
    masked_weights_arr = np.vstack(masked_weights_arr)
    print("masked item weights arr shape: ", masked_weights_arr.shape)

    # ===== multiply
    # prelocalizer patterns and prelocalizer item weights
    item_repress_pre = np.multiply(
        masked_bolds_arr_1, masked_weights_arr
    )  # these are lined up since the trials goes to correct trials

    print("item representations pre shape: ", item_repress_pre.shape)

    # these are used to hold the fidelity changes from pre to study (item-weighted)
    iw_dict = {}

    # these are used to hold the fidelity changes from pre to study (scene-weighted)
    cw_dict = {}

    counter = 0

    item_repress_study_comp = np.zeros_like(item_repress_pre[:90, :])
    item_repress_pre_comp = np.zeros_like(item_repress_pre[:90, :])
    item_repress_removal_comp = {}

    for trial in study_scene_order["trial_id"].values:
        study_trial_index = study_scene_order.index[
            study_scene_order["trial_id"] == trial
        ].tolist()[
            0
        ]  # find the order
        study_image_id = study_scene_order.loc[
            study_trial_index, "image_id"
        ]  # this now uses the index of the dataframe to find the image_id

        pre_trial_index = pre_scene_order.index[
            pre_scene_order["image_id"] == study_image_id
        ].tolist()[
            0
        ]  # find the index in the pre for this study trial
        pre_trial_num = pre_scene_order.loc[pre_trial_index, "trial_id"]

        image_condition = pre_scene_order.loc[
            pre_scene_order["trial_id"] == pre_trial_num
        ]["condition"].values[0]
        pre_trial_subcat = pre_scene_order.loc[pre_trial_index, "subcategory"]

        item_repress_study_comp[counter] = np.multiply(
            masked_bolds_arr_2[trial - 1, :],
            masked_weights_arr[pre_trial_num - 1, :],
        )

        item_repress_pre_comp[counter] = item_repress_pre[pre_trial_num - 1, :]
        counter = counter + 1

    item_pre_study_comp = np.corrcoef(item_repress_pre_comp, item_repress_study_comp)

    item_weighted_pre = np.zeros_like(item_repress_pre[:90, :])
    item_weighted_study = np.zeros_like(item_repress_pre[:90, :])

    counter = 0
    # this loop is limited by the smaller index, so thats the study condition (only 90 stims)
    for trial in study_scene_order["trial_id"].values:
        study_trial_index = study_scene_order.index[
            study_scene_order["trial_id"] == trial
        ].tolist()[
            0
        ]  # find the order
        study_image_id = study_scene_order.loc[
            study_trial_index, "image_id"
        ]  # this now uses the index of the dataframe to find the image_id

        pre_trial_index = pre_scene_order.index[
            pre_scene_order["image_id"] == study_image_id
        ].tolist()[
            0
        ]  # find the index in the pre for this study trial
        pre_trial_num = pre_scene_order.loc[pre_trial_index, "trial_id"]

        # now that I have the link between prelocalizer, study, and postlocalizer I can get that representation weighted with the item weight
        item_weighted_study[counter] = np.multiply(
            masked_bolds_arr_2[trial - 1, :], masked_weights_arr[pre_trial_num - 1, :]
        )

        item_weighted_pre[counter] = item_repress_pre[pre_trial_num - 1, :]

        item_others = []

        for j in range(len(masked_bolds_arr_2)):
            if (trial - 1) != j:
                weighted_other = np.multiply(
                    masked_bolds_arr_2[j - 1, :],
                    masked_weights_arr[pre_trial_num - 1, :],
                )

                temp_fidelity = np.corrcoef(
                    item_weighted_pre[counter, :], weighted_other
                )

                item_others.append(temp_fidelity[1][0])

        # This is to get the fidelity of the current item/trial from pre to study (item_weighted)
        pre_study_trial_iw_fidelity = np.corrcoef(
            item_weighted_pre[counter, :], item_weighted_study[counter, :]
        )

        other_fidelity = np.mean(item_others)

        iw_dict["image ID: %s" % study_image_id] = [
            pre_study_trial_iw_fidelity[1][0],
            other_fidelity,
        ]

        counter = counter + 1

    temp_df = pd.DataFrame(data=iw_dict).T
    temp_df.to_csv(
        os.path.join(
            container_path,
            "sub-0%s" % subID,
            "Representational_Changes_%s" % brain_flag,
            "item_weight_pre_study_RSA_w_other.csv",
        )
    )

    temp_df2 = temp_df.mean().to_frame().T
    temp_df2.columns = ["Fidelity-Same", "Fidelity-Other"]
    temp_df2.to_csv(
        os.path.join(
            container_path,
            "sub-0%s" % subID,
            "Representational_Changes_%s" % brain_flag,
            "average_item_weight_pre_study_RSA_w_other.csv",
        )
    )

    del temp_df

    print("Subject is done... saving everything")
    print(
        "==============================================================================="
    )


def load_all_data(container_path, brain_flag, subject_ids):
    dataframes = []

    for subID in subject_ids:
        file_path = os.path.join(
            container_path,
            "sub-0%s" % subID,
            "Representational_Changes_%s" % brain_flag,
            "item_weight_pre_study_RSA_w_other.csv",
        )
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
            df["subject_id"] = subID  # Add subject_id column for reference
            dataframes.append(df)

    # concatenate all dataframes along the row axis
    all_data = pd.concat(dataframes, axis=0)

    return all_data


def load_average_data(container_path, brain_flag, subject_ids):
    dataframes = []

    for subID in subject_ids:
        file_path = os.path.join(
            container_path,
            "sub-0%s" % subID,
            "Representational_Changes_%s" % brain_flag,
            "average_item_weight_pre_study_RSA_w_other.csv",
        )
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
            df["subject_id"] = subID  # Add subject_id column for reference
            dataframes.append(df)

    # concatenate all dataframes along the row axis
    average_data = pd.concat(dataframes, axis=0, ignore_index=True)

    return average_data


all_data = load_all_data(container_path, brain_flag, subs)
average_data = load_average_data(container_path, brain_flag, subs)

t_value, p_value = stats.ttest_rel(
    average_data["Fidelity-Same"], average_data["Fidelity-Other"]
)

print(f"Results of Subject level analysis: T(21)={t_value}    p = {p_value}!")
