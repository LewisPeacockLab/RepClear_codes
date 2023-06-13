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
from nilearn.image import clean_img, concat_imgs
from nilearn.signal import clean
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed


subs = ["61", "69", "77"]
brain_flag = "MNI"

# code for the item level voxel activity for faces and scenes


def mkdir(path, local=False):
    if not os.path.exists(path):
        os.makedirs(path)


def pad_contrast(contrast_, n_columns):
    """A small routine to append zeros in contrast vectors"""
    return np.hstack((contrast_, np.zeros(n_columns - len(contrast_))))


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

    # find the proper nii.gz files


def find(pattern, path):  # find the pattern we're looking for
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
        return result


def item_representation_extract(subID):
    print("Running sub-0%s..." % subID)
    # define the subject
    sub = "sub-0%s" % subID
    container_path = "/scratch1/06873/zbretton/clearmem/"

    bold_path = os.path.join(container_path, sub, "func/")
    os.chdir(bold_path)

    # set up the path to the files and then moved into that directory

    localizer_files = find("*localizer*bold*.nii.gz", bold_path)
    wholebrain_mask_path = find("*localizer*mask*.nii.gz", bold_path)

    if brain_flag == "MNI":
        pattern = "*MNI*"
        pattern2 = "*MNI152NLin2009cAsym*preproc*resized*"
        brain_mask_path = fnmatch.filter(wholebrain_mask_path, pattern)
        localizer_files = fnmatch.filter(localizer_files, pattern2)

    brain_mask_path.sort()
    localizer_files.sort()
    vtc_mask_path = os.path.join(
        "/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/group_MNI_VTC_mask.nii.gz"
    )
    vtc_mask = nib.load(vtc_mask_path)

    # load in category mask that was created from the first GLM

    img = concat_imgs(localizer_files, memory="/scratch1/06873/zbretton/nilearn_cache")

    # to be used to filter the data
    # First we are removing the confounds
    # get all the folders within the bold path
    # confound_folders=[x[0] for x in os.walk(bold_path)]
    localizer_confounds_1 = find("*localizer*1*confounds*.tsv", bold_path)
    localizer_confounds_2 = find("*localizer*2*confounds*.tsv", bold_path)
    localizer_confounds_3 = find("*localizer*3*confounds*.tsv", bold_path)
    localizer_confounds_4 = find("*localizer*4*confounds*.tsv", bold_path)
    localizer_confounds_5 = find("*localizer*5*confounds*.tsv", bold_path)

    confound_run1 = pd.read_csv(localizer_confounds_1[0], sep="\t")
    confound_run2 = pd.read_csv(localizer_confounds_2[0], sep="\t")
    confound_run3 = pd.read_csv(localizer_confounds_3[0], sep="\t")
    confound_run4 = pd.read_csv(localizer_confounds_4[0], sep="\t")
    confound_run5 = pd.read_csv(localizer_confounds_5[0], sep="\t")

    confound_run1 = confound_cleaner(confound_run1)
    confound_run2 = confound_cleaner(confound_run2)
    confound_run3 = confound_cleaner(confound_run3)
    confound_run4 = confound_cleaner(confound_run4)
    confound_run5 = confound_cleaner(confound_run5)

    localizer_confounds = pd.concat(
        [confound_run1, confound_run2, confound_run3, confound_run4, confound_run5],
        ignore_index=False,
    )

    # get run list so I can clean the data across each of the runs
    run1_length = int((img.get_fdata().shape[3]) / 5)
    run2_length = int((img.get_fdata().shape[3]) / 5)
    run3_length = int((img.get_fdata().shape[3]) / 5)
    run4_length = int((img.get_fdata().shape[3]) / 5)
    run5_length = int((img.get_fdata().shape[3]) / 5)

    run1 = np.full(run1_length, 1)
    run2 = np.full(run2_length, 2)
    run3 = np.full(run3_length, 3)
    run4 = np.full(run4_length, 4)
    run5 = np.full(run5_length, 5)

    run_list = np.concatenate((run1, run2, run3, run4, run5))
    # clean data ahead of the GLM
    img_clean = clean_img(
        img,
        sessions=run_list,
        t_r=1,
        detrend=False,
        standardize="zscore",
        mask_img=vtc_mask,
    )
    """load in the denoised bold data and events file"""
    events = pd.read_csv(
        "/scratch1/06873/zbretton/clearmem/localizer_events_item_sampled.csv", sep=","
    )
    # now will need to create a loop where I iterate over the face & scene indexes
    # I then relabel that trial of the face or scene as "face_trial#" or "scene_trial#" and then label rest and all other trials as "other"
    # I can either do this in one loop, or two consecutive

    # this has too much info so we need to only take the important columns
    events = events[["onset", "duration", "trial_type", "onsets_volume"]]

    events["onset"] = (
        events["onsets_volume"] - 1
    )  # we need to be in TR timing and not seconds
    events["duration"] = events["duration"] / 0.46

    events = events[["onset", "duration", "trial_type"]]

    temp_events = (
        events.copy()
    )  # copy the original events list, so that we can convert the "faces" and "scenes" to include the trial # (which corresponds to a unique image)

    for image_id in np.unique(events["trial_type"].values):
        img_indx = np.where(temp_events["trial_type"] == image_id)[0]
        temp_events["trial_type"][img_indx[1:]] = 0

    print("data is loaded, and events file is sorted...")
    for image_id in np.unique(events["trial_type"].values):
        if image_id > 0:
            print("running image %s" % (image_id))
            # get the onset of the trial so that I can average the time points:
            idx = temp_events[temp_events["trial_type"] == image_id].index.values[0]
            onset = temp_events.loc[idx, "onset"] + 10

            affine_mat = img_clean.affine
            dimsize = img_clean.header.get_zooms()

            """point to and if necessary create the output folder"""
            out_folder = os.path.join(container_path, sub, "item_representations_MDS")
            if not os.path.exists(out_folder):
                os.makedirs(out_folder, exist_ok=True)

            trial_pattern = np.mean(
                img_clean.get_fdata()[:, :, :, (onset) : (onset + 3)], axis=3
            )  # this is getting the 3 TRs for that trial's onset and then taking the average of it across the 4th dimension (time)

            output_name = os.path.join(
                out_folder,
                ("Sub-0%s_localizer_image%s_result.nii.gz" % (subID, (image_id))),
            )
            trial_pattern = trial_pattern.astype(
                "double"
            )  # Convert the output into a precision format that can be used by other applications
            trial_pattern[
                np.isnan(trial_pattern)
            ] = 0  # Exchange nans with zero to ensure compatibility with other applications
            trial_pattern_nii = nib.Nifti1Image(
                trial_pattern, affine_mat
            )  # create the volume image
            hdr = trial_pattern_nii.header  # get a handle of the .nii file's header
            hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2]))
            nib.save(trial_pattern_nii, output_name)  # Save the volume

            print("image %s saved" % image_id)

            del (
                trial_pattern,
                trial_pattern_nii,
                affine_mat,
                onset,
                out_folder,
                output_name,
                hdr,
            )

    print("Sub-0%s is complete" % subID)


Parallel(n_jobs=len(subs))(delayed(item_representation_extract)(i) for i in subs)
