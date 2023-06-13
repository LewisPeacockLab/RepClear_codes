# This is a cleaned version of the localizer data script, it's now being optimized for the values we already know we want and simplify the data that we output
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
from nilearn.image import clean_img
from nilearn.signal import clean
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score


# Making a subcategory version of this classifier
subs = ["02", "03", "04"]

TR_shifts = [5, 6]
brain_flag = "T1w"

# masks=['wholebrain','vtc'] #wholebrain/vtc
masks = ["vtc"]  # added: PHG and FG
# masks=['vtc','PHG','FG']

clear_data = 1  # 0 off / 1 on
force_clean = 1  # 0 off / 1 on

rest = "on"  # off


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


for TR_shift in TR_shifts:
    for mask_flag in masks:
        for num in range(len(subs)):
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

            localizer_files = find("*preremoval*bold*.nii.gz", bold_path)
            wholebrain_mask_path = find("*preremoval*mask*.nii.gz", bold_path)

            if brain_flag == "MNI":
                pattern = "*MNI*"
                pattern2 = "*MNI152NLin2009cAsym*preproc*"
                brain_mask_path = fnmatch.filter(wholebrain_mask_path, pattern)
                localizer_files = fnmatch.filter(localizer_files, pattern2)
            elif brain_flag == "T1w":
                pattern = "*T1w*"
                pattern2 = "*T1w*preproc*"
                brain_mask_path = fnmatch.filter(wholebrain_mask_path, pattern)
                localizer_files = fnmatch.filter(localizer_files, pattern2)

            brain_mask_path.sort()
            localizer_files.sort()
            if mask_flag == "vtc":
                vtc_mask_path = os.path.join(
                    "/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/",
                    sub,
                    "new_mask",
                    "VVS_preremoval_%s_mask.nii.gz" % brain_flag,
                )
                # vtc_mask_path=os.path.join('/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/','MNI_VTC_mask.nii.gz')

                vtc_mask = nib.load(vtc_mask_path)
            elif mask_flag == "PHG":
                PHG_mask_path = os.path.join(
                    "/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/",
                    sub,
                    "new_mask",
                    "PHG_preremoval_%s_mask.nii.gz" % brain_flag,
                )

                PHG_mask = nib.load(PHG_mask_path)
            elif mask_flag == "FG":
                FG_mask_path = os.path.join(
                    "/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/",
                    sub,
                    "new_mask",
                    "FG_preremoval_%s_mask.nii.gz" % brain_flag,
                )

                FG_mask = nib.load(FG_mask_path)

            if clear_data == 0:
                if os.path.exists(
                    os.path.join(
                        bold_path,
                        "sub-0%s_%s_preremoval_%s_masked.npy"
                        % (sub_num, brain_flag, mask_flag),
                    )
                ):
                    localizer_bold = np.load(
                        os.path.join(
                            bold_path,
                            "sub-0%s_%s_preremoval_%s_masked.npy"
                            % (sub_num, brain_flag, mask_flag),
                        )
                    )
                    print("%s %s Localizer Loaded..." % (brain_flag, mask_flag))
                    run1_length = int((len(localizer_bold) / 6))
                    run2_length = int((len(localizer_bold) / 6))
                    run3_length = int((len(localizer_bold) / 6))
                    run4_length = int((len(localizer_bold) / 6))
                    run5_length = int((len(localizer_bold) / 6))
                    run6_length = int((len(localizer_bold) / 6))
                else:
                    # select the specific file
                    localizer_run1 = nib.load(localizer_files[0])
                    localizer_run2 = nib.load(localizer_files[1])
                    localizer_run3 = nib.load(localizer_files[2])
                    localizer_run4 = nib.load(localizer_files[3])
                    localizer_run5 = nib.load(localizer_files[4])
                    localizer_run6 = nib.load(localizer_files[5])

                    # to be used to filter the data
                    # First we are removing the confounds
                    # get all the folders within the bold path
                    # confound_folders=[x[0] for x in os.walk(bold_path)]
                    localizer_confounds_1 = find(
                        "*preremoval*1*confounds*.tsv", bold_path
                    )
                    localizer_confounds_2 = find(
                        "*preremoval*2*confounds*.tsv", bold_path
                    )
                    localizer_confounds_3 = find(
                        "*preremoval*3*confounds*.tsv", bold_path
                    )
                    localizer_confounds_4 = find(
                        "*preremoval*4*confounds*.tsv", bold_path
                    )
                    localizer_confounds_5 = find(
                        "*preremoval*5*confounds*.tsv", bold_path
                    )
                    localizer_confounds_6 = find(
                        "*preremoval*6*confounds*.tsv", bold_path
                    )

                    confound_run1 = pd.read_csv(localizer_confounds_1[0], sep="\t")
                    confound_run2 = pd.read_csv(localizer_confounds_2[0], sep="\t")
                    confound_run3 = pd.read_csv(localizer_confounds_3[0], sep="\t")
                    confound_run4 = pd.read_csv(localizer_confounds_4[0], sep="\t")
                    confound_run5 = pd.read_csv(localizer_confounds_5[0], sep="\t")
                    confound_run6 = pd.read_csv(localizer_confounds_6[0], sep="\t")

                    confound_run1 = confound_cleaner(confound_run1)
                    confound_run2 = confound_cleaner(confound_run2)
                    confound_run3 = confound_cleaner(confound_run3)
                    confound_run4 = confound_cleaner(confound_run4)
                    confound_run5 = confound_cleaner(confound_run5)
                    confound_run6 = confound_cleaner(confound_run6)

                    wholebrain_mask1 = nib.load(brain_mask_path[0])
                    wholebrain_mask2 = nib.load(brain_mask_path[1])
                    wholebrain_mask3 = nib.load(brain_mask_path[2])
                    wholebrain_mask4 = nib.load(brain_mask_path[3])
                    wholebrain_mask5 = nib.load(brain_mask_path[4])
                    wholebrain_mask6 = nib.load(brain_mask_path[5])

                    def apply_mask(mask=None, target=None):
                        coor = np.where(mask == 1)
                        values = target[coor]
                        if values.ndim > 1:
                            values = np.transpose(
                                values
                            )  # swap axes to get feature X sample
                        return values

                    if mask_flag == "wholebrain":
                        localizer_run1 = apply_mask(
                            mask=(wholebrain_mask1.get_data()),
                            target=(localizer_run1.get_data()),
                        )
                        localizer_run2 = apply_mask(
                            mask=(wholebrain_mask1.get_data()),
                            target=(localizer_run2.get_data()),
                        )
                        localizer_run3 = apply_mask(
                            mask=(wholebrain_mask1.get_data()),
                            target=(localizer_run3.get_data()),
                        )
                        localizer_run4 = apply_mask(
                            mask=(wholebrain_mask1.get_data()),
                            target=(localizer_run4.get_data()),
                        )
                        localizer_run5 = apply_mask(
                            mask=(wholebrain_mask1.get_data()),
                            target=(localizer_run5.get_data()),
                        )
                        localizer_run6 = apply_mask(
                            mask=(wholebrain_mask1.get_data()),
                            target=(localizer_run6.get_data()),
                        )

                    elif mask_flag == "vtc":
                        localizer_run1 = apply_mask(
                            mask=(vtc_mask.get_data()),
                            target=(localizer_run1.get_data()),
                        )
                        localizer_run2 = apply_mask(
                            mask=(vtc_mask.get_data()),
                            target=(localizer_run2.get_data()),
                        )
                        localizer_run3 = apply_mask(
                            mask=(vtc_mask.get_data()),
                            target=(localizer_run3.get_data()),
                        )
                        localizer_run4 = apply_mask(
                            mask=(vtc_mask.get_data()),
                            target=(localizer_run4.get_data()),
                        )
                        localizer_run5 = apply_mask(
                            mask=(vtc_mask.get_data()),
                            target=(localizer_run5.get_data()),
                        )
                        localizer_run6 = apply_mask(
                            mask=(vtc_mask.get_data()),
                            target=(localizer_run6.get_data()),
                        )

                    elif mask_flag == "PHG":
                        localizer_run1 = apply_mask(
                            mask=(PHG_mask.get_data()),
                            target=(localizer_run1.get_data()),
                        )
                        localizer_run2 = apply_mask(
                            mask=(PHG_mask.get_data()),
                            target=(localizer_run2.get_data()),
                        )
                        localizer_run3 = apply_mask(
                            mask=(PHG_mask.get_data()),
                            target=(localizer_run3.get_data()),
                        )
                        localizer_run4 = apply_mask(
                            mask=(PHG_mask.get_data()),
                            target=(localizer_run4.get_data()),
                        )
                        localizer_run5 = apply_mask(
                            mask=(PHG_mask.get_data()),
                            target=(localizer_run5.get_data()),
                        )
                        localizer_run6 = apply_mask(
                            mask=(PHG_mask.get_data()),
                            target=(localizer_run6.get_data()),
                        )

                    elif mask_flag == "FG":
                        localizer_run1 = apply_mask(
                            mask=(FG_mask.get_data()),
                            target=(localizer_run1.get_data()),
                        )
                        localizer_run2 = apply_mask(
                            mask=(FG_mask.get_data()),
                            target=(localizer_run2.get_data()),
                        )
                        localizer_run3 = apply_mask(
                            mask=(FG_mask.get_data()),
                            target=(localizer_run3.get_data()),
                        )
                        localizer_run4 = apply_mask(
                            mask=(FG_mask.get_data()),
                            target=(localizer_run4.get_data()),
                        )
                        localizer_run5 = apply_mask(
                            mask=(FG_mask.get_data()),
                            target=(localizer_run5.get_data()),
                        )
                        localizer_run6 = apply_mask(
                            mask=(FG_mask.get_data()),
                            target=(localizer_run6.get_data()),
                        )

                    preproc_1 = clean(
                        localizer_run1, t_r=1, detrend=False, standardize="zscore"
                    )
                    preproc_2 = clean(
                        localizer_run2, t_r=1, detrend=False, standardize="zscore"
                    )
                    preproc_3 = clean(
                        localizer_run3, t_r=1, detrend=False, standardize="zscore"
                    )
                    preproc_4 = clean(
                        localizer_run4, t_r=1, detrend=False, standardize="zscore"
                    )
                    preproc_5 = clean(
                        localizer_run5, t_r=1, detrend=False, standardize="zscore"
                    )
                    preproc_6 = clean(
                        localizer_run6, t_r=1, detrend=False, standardize="zscore"
                    )

                    run1_length = int((len(localizer_run1)))
                    run2_length = int((len(localizer_run2)))
                    run3_length = int((len(localizer_run3)))
                    run4_length = int((len(localizer_run4)))
                    run5_length = int((len(localizer_run5)))
                    run6_length = int((len(localizer_run6)))

                    localizer_bold = np.concatenate(
                        (
                            preproc_1,
                            preproc_2,
                            preproc_3,
                            preproc_4,
                            preproc_5,
                            preproc_6,
                        )
                    )
                    # save this data if we didn't have it saved before
                    os.chdir(bold_path)
                    np.save(
                        "sub-0%s_%s_preremoval_%s_masked"
                        % (sub_num, brain_flag, mask_flag),
                        localizer_bold,
                    )
                    print("%s %s masked data...saved" % (mask_flag, brain_flag))

            else:
                if (
                    os.path.exists(
                        os.path.join(
                            bold_path,
                            "sub-0%s_%s_preremoval_%s_masked_cleaned.npy"
                            % (sub_num, brain_flag, mask_flag),
                        )
                    )
                ) & (force_clean == 0):
                    localizer_bold = np.load(
                        os.path.join(
                            bold_path,
                            "sub-0%s_%s_preremoval_%s_masked_cleaned.npy"
                            % (sub_num, brain_flag, mask_flag),
                        )
                    )
                    print("%s %s Cleaned Localizer Loaded..." % (brain_flag, mask_flag))
                    run1_length = int((len(localizer_bold) / 6))
                    run2_length = int((len(localizer_bold) / 6))
                    run3_length = int((len(localizer_bold) / 6))
                    run4_length = int((len(localizer_bold) / 6))
                    run5_length = int((len(localizer_bold) / 6))
                    run6_length = int((len(localizer_bold) / 6))
                else:
                    # select the specific file
                    localizer_run1 = nib.load(localizer_files[0])
                    localizer_run2 = nib.load(localizer_files[1])
                    localizer_run3 = nib.load(localizer_files[2])
                    localizer_run4 = nib.load(localizer_files[3])
                    localizer_run5 = nib.load(localizer_files[4])
                    localizer_run6 = nib.load(localizer_files[5])

                    # to be used to filter the data
                    # First we are removing the confounds
                    # get all the folders within the bold path
                    # confound_folders=[x[0] for x in os.walk(bold_path)]
                    localizer_confounds_1 = find(
                        "*preremoval*1*confounds*.tsv", bold_path
                    )
                    localizer_confounds_2 = find(
                        "*preremoval*2*confounds*.tsv", bold_path
                    )
                    localizer_confounds_3 = find(
                        "*preremoval*3*confounds*.tsv", bold_path
                    )
                    localizer_confounds_4 = find(
                        "*preremoval*4*confounds*.tsv", bold_path
                    )
                    localizer_confounds_5 = find(
                        "*preremoval*5*confounds*.tsv", bold_path
                    )
                    localizer_confounds_6 = find(
                        "*preremoval*6*confounds*.tsv", bold_path
                    )

                    confound_run1 = pd.read_csv(localizer_confounds_1[0], sep="\t")
                    confound_run2 = pd.read_csv(localizer_confounds_2[0], sep="\t")
                    confound_run3 = pd.read_csv(localizer_confounds_3[0], sep="\t")
                    confound_run4 = pd.read_csv(localizer_confounds_4[0], sep="\t")
                    confound_run5 = pd.read_csv(localizer_confounds_5[0], sep="\t")
                    confound_run6 = pd.read_csv(localizer_confounds_6[0], sep="\t")

                    confound_run1 = confound_cleaner(confound_run1)
                    confound_run2 = confound_cleaner(confound_run2)
                    confound_run3 = confound_cleaner(confound_run3)
                    confound_run4 = confound_cleaner(confound_run4)
                    confound_run5 = confound_cleaner(confound_run5)
                    confound_run6 = confound_cleaner(confound_run6)

                    wholebrain_mask1 = nib.load(brain_mask_path[0])
                    wholebrain_mask2 = nib.load(brain_mask_path[1])
                    wholebrain_mask3 = nib.load(brain_mask_path[2])
                    wholebrain_mask4 = nib.load(brain_mask_path[3])
                    wholebrain_mask5 = nib.load(brain_mask_path[4])
                    wholebrain_mask6 = nib.load(brain_mask_path[5])

                    def apply_mask(mask=None, target=None):
                        coor = np.where(mask == 1)
                        values = target[coor]
                        if values.ndim > 1:
                            values = np.transpose(
                                values
                            )  # swap axes to get feature X sample
                        return values

                    if mask_flag == "wholebrain":
                        localizer_run1 = apply_mask(
                            mask=(wholebrain_mask1.get_data()),
                            target=(localizer_run1.get_data()),
                        )
                        localizer_run2 = apply_mask(
                            mask=(wholebrain_mask1.get_data()),
                            target=(localizer_run2.get_data()),
                        )
                        localizer_run3 = apply_mask(
                            mask=(wholebrain_mask1.get_data()),
                            target=(localizer_run3.get_data()),
                        )
                        localizer_run4 = apply_mask(
                            mask=(wholebrain_mask1.get_data()),
                            target=(localizer_run4.get_data()),
                        )
                        localizer_run5 = apply_mask(
                            mask=(wholebrain_mask1.get_data()),
                            target=(localizer_run5.get_data()),
                        )
                        localizer_run6 = apply_mask(
                            mask=(wholebrain_mask1.get_data()),
                            target=(localizer_run6.get_data()),
                        )

                    elif mask_flag == "vtc":
                        localizer_run1 = apply_mask(
                            mask=(vtc_mask.get_data()),
                            target=(localizer_run1.get_data()),
                        )
                        localizer_run2 = apply_mask(
                            mask=(vtc_mask.get_data()),
                            target=(localizer_run2.get_data()),
                        )
                        localizer_run3 = apply_mask(
                            mask=(vtc_mask.get_data()),
                            target=(localizer_run3.get_data()),
                        )
                        localizer_run4 = apply_mask(
                            mask=(vtc_mask.get_data()),
                            target=(localizer_run4.get_data()),
                        )
                        localizer_run5 = apply_mask(
                            mask=(vtc_mask.get_data()),
                            target=(localizer_run5.get_data()),
                        )
                        localizer_run6 = apply_mask(
                            mask=(vtc_mask.get_data()),
                            target=(localizer_run6.get_data()),
                        )

                    elif mask_flag == "PHG":
                        localizer_run1 = apply_mask(
                            mask=(PHG_mask.get_data()),
                            target=(localizer_run1.get_data()),
                        )
                        localizer_run2 = apply_mask(
                            mask=(PHG_mask.get_data()),
                            target=(localizer_run2.get_data()),
                        )
                        localizer_run3 = apply_mask(
                            mask=(PHG_mask.get_data()),
                            target=(localizer_run3.get_data()),
                        )
                        localizer_run4 = apply_mask(
                            mask=(PHG_mask.get_data()),
                            target=(localizer_run4.get_data()),
                        )
                        localizer_run5 = apply_mask(
                            mask=(PHG_mask.get_data()),
                            target=(localizer_run5.get_data()),
                        )
                        localizer_run6 = apply_mask(
                            mask=(PHG_mask.get_data()),
                            target=(localizer_run6.get_data()),
                        )

                    elif mask_flag == "FG":
                        localizer_run1 = apply_mask(
                            mask=(FG_mask.get_data()),
                            target=(localizer_run1.get_data()),
                        )
                        localizer_run2 = apply_mask(
                            mask=(FG_mask.get_data()),
                            target=(localizer_run2.get_data()),
                        )
                        localizer_run3 = apply_mask(
                            mask=(FG_mask.get_data()),
                            target=(localizer_run3.get_data()),
                        )
                        localizer_run4 = apply_mask(
                            mask=(FG_mask.get_data()),
                            target=(localizer_run4.get_data()),
                        )
                        localizer_run5 = apply_mask(
                            mask=(FG_mask.get_data()),
                            target=(localizer_run5.get_data()),
                        )
                        localizer_run6 = apply_mask(
                            mask=(FG_mask.get_data()),
                            target=(localizer_run6.get_data()),
                        )

                    preproc_1 = clean(
                        localizer_run1,
                        confounds=(confound_run1),
                        t_r=1,
                        detrend=False,
                        standardize="zscore",
                    )
                    preproc_2 = clean(
                        localizer_run2,
                        confounds=(confound_run2),
                        t_r=1,
                        detrend=False,
                        standardize="zscore",
                    )
                    preproc_3 = clean(
                        localizer_run3,
                        confounds=(confound_run3),
                        t_r=1,
                        detrend=False,
                        standardize="zscore",
                    )
                    preproc_4 = clean(
                        localizer_run4,
                        confounds=(confound_run4),
                        t_r=1,
                        detrend=False,
                        standardize="zscore",
                    )
                    preproc_5 = clean(
                        localizer_run5,
                        confounds=(confound_run5),
                        t_r=1,
                        detrend=False,
                        standardize="zscore",
                    )
                    preproc_6 = clean(
                        localizer_run6,
                        confounds=(confound_run6),
                        t_r=1,
                        detrend=False,
                        standardize="zscore",
                    )

                    run1_length = int((len(localizer_run1)))
                    run2_length = int((len(localizer_run2)))
                    run3_length = int((len(localizer_run3)))
                    run4_length = int((len(localizer_run4)))
                    run5_length = int((len(localizer_run5)))
                    run6_length = int((len(localizer_run6)))

                    localizer_bold = np.concatenate(
                        (
                            preproc_1,
                            preproc_2,
                            preproc_3,
                            preproc_4,
                            preproc_5,
                            preproc_6,
                        )
                    )
                    # save this data if we didn't have it saved before
                    os.chdir(bold_path)
                    np.save(
                        "sub-0%s_%s_preremoval_%s_masked_cleaned"
                        % (sub_num, brain_flag, mask_flag),
                        localizer_bold,
                    )
                    print(
                        "%s %s masked and cleaned data...saved"
                        % (mask_flag, brain_flag)
                    )

            # fill in the run array with run number
            run1 = np.full(run1_length, 1)
            run2 = np.full(run2_length, 2)
            run3 = np.full(run3_length, 3)
            run4 = np.full(run4_length, 4)
            run5 = np.full(run5_length, 5)
            run6 = np.full(run6_length, 6)

            run_list = np.concatenate(
                (run1, run2, run3, run4, run5, run6)
            )  # now combine

            # load regs / labels

            # Categories: 1 Scenes, 2 Faces / 0 is rest

            params_dir = "/scratch1/06873/zbretton/repclear_dataset/BIDS/params"
            # find the mat file, want to change this to fit "sub"
            param_search = "preremoval*events*.csv"
            param_file = find(param_search, params_dir)

            reg_matrix = pd.read_csv(param_file[0])
            reg_category = reg_matrix["category"].values
            reg_subcategory = reg_matrix["subcategory"].values
            reg_stim_on = reg_matrix["stim_present"].values
            reg_run = reg_matrix["run"].values

            run1_index = np.where(reg_run == 1)
            run2_index = np.where(reg_run == 2)
            run3_index = np.where(reg_run == 3)
            run4_index = np.where(reg_run == 4)
            run5_index = np.where(reg_run == 5)
            run6_index = np.where(reg_run == 6)

            # extract times where stimuli is on for both categories:
            # stim_on=np.where((reg_stim_on==1) & ((reg_category==1) | (reg_category==2)))
            stim_on = reg_stim_on
            # need to convert this list to 1-d
            stim_list = np.empty(len(localizer_bold))
            stim_list = reg_category
            # this list is now 1d list, need to add a dimentsionality to it
            stim_list = stim_list[:, None]
            stim_on = stim_on[:, None]

            stim_subcat_list = np.empty(len(localizer_bold))
            stim_subcat_list = reg_subcategory
            stim_subcat_list = stim_subcat_list[:, None]

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
            shift_size = TR_shift  # this is shifting by 5TR
            tag = 0  # rest label is 0
            stim_list_shift = shift_timing(
                stim_list, shift_size, tag
            )  # rest is label 0
            stim_on_shift = shift_timing(stim_on, shift_size, tag)
            stim_subcat_shift = shift_timing(stim_subcat_list, shift_size, tag)

            # Here I need to balance the trials of the categories / rest. I will be removing rest, but Scenes have twice the data of faces, so need to decide how to handle
            rest_times = np.where(stim_list_shift == 0)
            rest_times_on = np.where(stim_on_shift == 0)
            # these should be the same, but doing it just in case

            # this condition is going to be rebuild to include the resting times, so that the two versions being tested are stimuli on alone and then stimuli on + balanced amount of rest

            temp_shift = np.full(len(stim_list_shift), 0)
            temp_shift[
                np.where(
                    (stim_on_shift == 1)
                    & (stim_list_shift == 1)
                    & (stim_subcat_shift == 1)
                )
            ] = 1  # label manmade scenes as 1
            temp_shift[
                np.where(
                    (stim_on_shift == 1)
                    & (stim_list_shift == 1)
                    & (stim_subcat_shift == 2)
                )
            ] = 2  # label natural scenes as 2
            temp_shift[
                np.where(
                    (stim_on_shift == 1)
                    & (stim_list_shift == 2)
                    & (stim_subcat_shift == 1)
                )
            ] = 3  # label female faces as 3
            temp_shift[
                np.where(
                    (stim_on_shift == 1)
                    & (stim_list_shift == 2)
                    & (stim_subcat_shift == 2)
                )
            ] = 4  # label male faces as 4

            stims_on = np.where(
                (stim_on_shift == 1) & ((stim_list_shift == 1) | (stim_list_shift == 2))
            )  # get times where stims are on

            rest_btwn_stims = np.where(
                (stim_on_shift == 2) & ((stim_list_shift == 1) | (stim_list_shift == 2))
            )
            rest_btwn_stims_filt = rest_btwn_stims[0][
                2::12
            ]  # start at the 3rd index and then take each 12th item from this list of indicies
            # this above line results in 90 samples, we need to bring it down to 60 to match the samples of faces (minimum samples)
            # this works well with the current situation since we are not using the last 2 runs in the x-validation because of uneven samples
            temp_shift[
                rest_btwn_stims_filt
            ] = 0  # using the targeted rest times, I am setting them to 0 (rest's label) since for now they are still labeled as the category of that trial

            stims_and_rest = np.concatenate(
                (rest_btwn_stims_filt, stims_on[0])
            )  # these are all the events we need, faces(120), scenes(240) and rest(180)
            stims_and_rest.sort()  # put this in order so we can sample properly
            stim_on_rest = temp_shift[stims_and_rest]
            localizer_bold_stims_and_rest = localizer_bold[stims_and_rest]
            run_list_stims_and_rest = run_list[stims_and_rest]

            # for now I have the Classifier taking the first 4 runs and doing a 50/50 split. I could add another wrinkle here where the runs 3 & 4 get swapped out for 5 & 6 and add another iteration
            # but it doesnt seem to change anything, and then when I use this I am training on all the data anyhow
            def L2_xval(data, labels, run_labels, groups):
                scores = []
                predicts = []
                trues = []
                evidences = []
                chosenvoxels = []
                sig_scores = []
                predict_probs = []  # adding this to test against the decision function
                ps = LeaveOneGroupOut()
                C_best = []
                for train_index, test_index in ps.split(data, labels, groups):
                    X_train, X_test = data[train_index], data[test_index]
                    y_train, y_test = labels[train_index], labels[test_index]

                    # selectedvoxels=SelectKBest(f_classif,k=7500).fit(X_train,y_train)
                    selectedvoxels = SelectFpr(f_classif, alpha=0.01).fit(
                        X_train, y_train
                    )  # I compared this method to taking ALL k items in the F-test and filtering by p-value, so I assume this is a better feature selection method

                    X_train = selectedvoxels.transform(X_train)
                    X_test = selectedvoxels.transform(X_test)

                    # fit model
                    if rest == "on":
                        L2 = LogisticRegression(penalty="l2", solver="liblinear", C=1)
                        L2_out = L2.fit(X_train, y_train)
                        # calculate evidence values
                        evidence = 1.0 / (
                            1.0 + np.exp(-L2_out.decision_function(X_test))
                        )
                        evidences.append(evidence)
                        predict_prob = L2_out.predict_proba(X_test)
                        predict_probs.append(predict_prob)
                        score = L2_out.score(X_test, y_test)  # score the model
                        predict = L2_out.predict(
                            X_test
                        )  # what are the predicted classes
                        predicts.append(predict)  # append the list over the x-val runs
                        true = y_test  # what is the known labels
                        scores.append(score)  # append the scores
                        trues.append(true)  # append the known labels
                        chosenvoxels.append(selectedvoxels.get_support())
                        sig_score = roc_auc_score(
                            true, L2_out.predict_proba(X_test), multi_class="ovr"
                        )
                        sig_scores.append(sig_score)
                    elif rest == "off":
                        L2 = LinearSVC()
                        L2_out = L2.fit(X_train, y_train)
                        # calculate evidence values
                        evidence = 1.0 / (
                            1.0 + np.exp(-L2_out.decision_function(X_test))
                        )
                        evidences.append(evidence)
                        predict_prob = L2_out.predict_proba(X_test)
                        predict_probs.append(predict_prob)
                        score = L2_out.score(X_test, y_test)  # score the model
                        predict = L2_out.predict(
                            X_test
                        )  # what are the predicted classes
                        predicts.append(predict)  # append the list over the x-val runs
                        true = y_test  # what is the known labels
                        scores.append(score)  # append the scores
                        trues.append(true)  # append the known labels
                        chosenvoxels.append(selectedvoxels.get_support())
                        sig_score = roc_auc_score(
                            true, L2_out.decision_function(X_test)
                        )
                        sig_scores.append(sig_score)
                return (
                    L2_out,
                    scores,
                    predicts,
                    trues,
                    evidences,
                    C_best,
                    chosenvoxels,
                    sig_scores,
                    predict_probs,
                )

            # test if limiting to the first 4 runs changes anything, so the samples should be balanced
            # stim_on_rest=stim_on_rest[run_list_stims_and_rest<=4] # now all the samples are even
            # an alternative of this is:
            stim_on_rest = stim_on_rest[
                (run_list_stims_and_rest <= 2) | (run_list_stims_and_rest >= 5)
            ]

            # stim_on_filt=stim_on_filt[run_list_on_filt<=4]

            # localizer_bold_stims_and_rest=localizer_bold_stims_and_rest[run_list_stims_and_rest<=4]
            # localizer_bold_on_filt=localizer_bold_on_filt[run_list_on_filt<=4]

            # run_list_stims_and_rest=run_list_stims_and_rest[run_list_stims_and_rest<=4]
            # run_list_on_filt=run_list_on_filt[run_list_on_filt<=4]

            localizer_bold_stims_and_rest = localizer_bold_stims_and_rest[
                (run_list_stims_and_rest <= 2) | (run_list_stims_and_rest >= 5)
            ]
            # localizer_bold_on_filt=localizer_bold_on_filt[(run_list_stims_and_rest<=2)|(run_list_stims_and_rest>=5)]

            run_list_stims_and_rest = run_list_stims_and_rest[
                (run_list_stims_and_rest <= 2) | (run_list_stims_and_rest >= 5)
            ]
            # run_list_on_filt=run_list_on_filt[(run_list_stims_and_rest<=2)|(run_list_stims_and_rest>=5)]

            # split the data in half, we will use the groupings:
            group_even_stims_and_rest = np.full(
                int(run_list_stims_and_rest.size / 4), 2
            )
            group_odd_stims_and_rest = np.full(int(run_list_stims_and_rest.size / 4), 1)
            group_end_stims_and_rest = np.full(int(run_list_stims_and_rest.size / 4), 3)

            groups_stims_and_rest = np.hstack(
                (
                    group_even_stims_and_rest,
                    group_odd_stims_and_rest,
                    group_even_stims_and_rest,
                    group_odd_stims_and_rest,
                )
            )

            group_even_on = np.full(int(run_list_on_filt.size / 4), 2)
            group_odd_on = np.full(int(run_list_on_filt.size / 4), 1)
            group_end_on = np.full(int(run_list_on_filt.size / 4), 3)

            groups_on = np.hstack(
                (group_even_on, group_odd_on, group_even_on, group_odd_on)
            )

            print("Now running classifier...")

            if rest == "off":
                (
                    L2_models,
                    L2_scores,
                    L2_predicts,
                    L2_trues,
                    L2_evidence,
                    L2_costs,
                    L2_chosenvoxels,
                    L2_sig_scores,
                    L2_predict_probs,
                ) = L2_xval(
                    localizer_bold_on_filt, stim_on_filt, run_list_on_filt, groups_on
                )
                L2_subject_score_mean = np.mean(L2_scores)
            elif rest == "on":
                (
                    L2_models_on_rest,
                    L2_scores_on_rest,
                    L2_predicts_on_rest,
                    L2_trues_on_rest,
                    L2_evidence_on_rest,
                    L2_costs_on_rest,
                    L2_chosenvoxels_on_rest,
                    L2_sig_scores_on_rest,
                    L2_predict_probs_on_rest,
                ) = L2_xval(
                    localizer_bold_stims_and_rest,
                    stim_on_rest,
                    run_list_stims_and_rest,
                    groups_stims_and_rest,
                )
                L2_subject_score_mean_on_rest = np.mean(L2_scores_on_rest)
            print("Classifier done!")
            print("Now running random Classifier to find chance level...")
            # Need to run a randomized classifier to test for validity
            n_iters = 1  # How many different permutations
            sp = LeaveOneGroupOut()
            clf_score = np.array([])

            for i in range(n_iters):
                clf_score_i = np.array([])
                permuted_labels = np.random.permutation(stim_on_rest)
                for train, test in sp.split(
                    localizer_bold_stims_and_rest, stim_on_rest, groups_stims_and_rest
                ):
                    # Pull out the sample data
                    train_data = localizer_bold_stims_and_rest[train, :]
                    test_data = localizer_bold_stims_and_rest[test, :]

                    # Do voxel selection on all voxels
                    # selected_voxels=SelectKBest(f_classif,k=1500).fit(train_data,permuted_labels[train])
                    selected_voxels = SelectFpr(f_classif, alpha=0.05).fit(
                        train_data, permuted_labels[train]
                    )  # I compared this method to taking ALL k items in the F-test and filtering by p-value, so I assume this is a better feature selection method

                    # Train and test the classifier
                    classifier = LogisticRegression(
                        penalty="l2", solver="liblinear", C=1
                    )
                    clf = classifier.fit(
                        selected_voxels.transform(train_data), permuted_labels[train]
                    )
                    score = clf.score(
                        selected_voxels.transform(test_data), permuted_labels[test]
                    )
                    clf_score_i = np.hstack((clf_score_i, score))
                clf_score = np.hstack((clf_score, clf_score_i.mean()))
            # the output of this should line up to "Chance"
            print("done!")
            print("----------------------------------")
            print("Data summary: Sub-0%s" % sub_num)
            print("TR Shift: %s" % TR_shift)
            print("Random Label score: %s" % clf_score)
            if rest == "off":
                print(
                    "L2 score - Only stimuli on: %s - Cost: %s - ROC/AUC: %s"
                    % (L2_scores, L2_costs, L2_sig_scores)
                )
            elif rest == "on":
                print(
                    "L2 With Rest Score: %s - Cost: %s - ROC/AUC: %s"
                    % (L2_scores_on_rest, L2_costs_on_rest, L2_sig_scores_on_rest)
                )
            print("----------------------------------")
            print("saving data...")
            # need to save an output per subject here
            if rest == "off":
                output_table = {
                    "subject": sub,
                    "Shuffled Data": clf_score,
                    "L2 Average Scores (only On)": L2_subject_score_mean,
                    "L2 Model (only On)": L2_models,
                    "L2 Raw Scores (only On)": L2_scores,
                    "L2 Predictions (only On)": L2_predicts,
                    "L2 True (only On)": L2_trues,
                    "L2 Costs (only On)": L2_costs,
                    "Class List (only On)": stim_on_filt,
                    "L2 Evidence (only On)": L2_evidence,
                    "Run List": run_list_on_filt,
                    "ROC/AUC (only On)": L2_sig_scores,
                    "Localizer Path": localizer_files,
                }
            elif rest == "on":
                output_table = {
                    "subject": sub,
                    "Shuffled Data": clf_score,
                    "L2 Average Scores (With Rest)": L2_subject_score_mean_on_rest,
                    "L2 Model (With Rest)": L2_models_on_rest,
                    "L2 Raw Scores (With Rest)": L2_scores_on_rest,
                    "L2 Predictions (With Rest)": L2_predicts_on_rest,
                    "L2 True (With Rest)": L2_trues_on_rest,
                    "L2 Costs (With Rest)": L2_costs_on_rest,
                    "Class List (With Rest)": stim_on_rest,
                    "L2 Evidence (With Rest)": L2_evidence_on_rest,
                    "Run List w/ Rest": run_list_stims_and_rest,
                    "ROC/AUC (With Rest)": L2_sig_scores_on_rest,
                    "Localizer Path": localizer_files,
                }

            import pickle

            os.chdir(os.path.join(container_path, sub))
            if clear_data == 0:
                f = open(
                    "%s-preremoval_subcategory_rest_%s_%s_%s_%sTR lag_data.pkl"
                    % (sub, rest, brain_flag, mask_flag, TR_shift),
                    "wb",
                )
                pickle.dump(output_table, f)
                f.close()
            else:
                f = open(
                    "%s-preremoval_subcategory_rest_%s_%s_%s_%sTR lag_data_cleaned.pkl"
                    % (sub, rest, brain_flag, mask_flag, TR_shift),
                    "wb",
                )
                pickle.dump(output_table, f)
                f.close()

            print("Sub-0%s is now complete" % sub_num)
