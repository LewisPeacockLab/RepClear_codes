# Outline of the searchlight code:
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
import brainiak
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import (
    PredefinedSplit,
    cross_validate,
    cross_val_predict,
    GridSearchCV,
    cross_val_score,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.feature_selection import (
    VarianceThreshold,
    f_classif,
    SelectKBest,
    SelectFpr,
)
from sklearn.preprocessing import StandardScaler
from nilearn.input_data import NiftiMasker, MultiNiftiMasker
from nilearn.image import clean_img, load_img, get_data, concat_imgs, resample_img
from nilearn.signal import clean
from scipy import stats
from sklearn import preprocessing
import time
from sklearn.metrics import roc_auc_score
from mpi4py import MPI
from brainiak.searchlight.searchlight import Ball, Searchlight
import argparse

parser = argparse.ArgumentParser(description="Subject input")
parser.add_argument(
    "--subject",
    dest="subject",
    type=str,
    help="subject number",
    default=["02", "03", "04"],
)
args = parser.parse_args()


# subs=['02','03','04']
subs = [
    args.subject
]  # allows me to input per subject so I can run the three subjects in parallel

TR_shifts = [5]  # 5,6
brain_flag = "MNI"  # MNI/T1w

masks = ["GM"]  # this is masked to the group GM mask

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


comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

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

                preproc_1 = clean_img(
                    study_run1,
                    t_r=1,
                    detrend=False,
                    standardize="zscore",
                    mask_img=gm_mask,
                )
                preproc_2 = clean_img(
                    study_run2,
                    t_r=1,
                    detrend=False,
                    standardize="zscore",
                    mask_img=gm_mask,
                )
                preproc_3 = clean_img(
                    study_run3,
                    t_r=1,
                    detrend=False,
                    standardize="zscore",
                    mask_img=gm_mask,
                )

                study_bold = concat_imgs((preproc_1, preproc_2, preproc_3))

                # create run array
                run1_length = int((len(study_run1)))
                run2_length = int((len(study_run2)))
                run3_length = int((len(study_run3)))
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

                preproc_1 = clean_img(
                    study_run1,
                    confounds=confound_run1,
                    t_r=1,
                    detrend=False,
                    standardize="zscore",
                    mask_img=gm_mask,
                )
                preproc_2 = clean_img(
                    study_run2,
                    confounds=confound_run2,
                    t_r=1,
                    detrend=False,
                    standardize="zscore",
                    mask_img=gm_mask,
                )
                preproc_3 = clean_img(
                    study_run3,
                    confounds=confound_run3,
                    t_r=1,
                    detrend=False,
                    standardize="zscore",
                    mask_img=gm_mask,
                )

                study_bold = concat_imgs((preproc_1, preproc_2, preproc_3))

                # create run array
                run1_length = int(study_run1.shape[3])  # get the TR length of this run
                run2_length = int(study_run2.shape[3])
                run3_length = int(study_run3.shape[3])

            # fill in the run array with run number
            run1 = np.full(run1_length, 1)
            run2 = np.full(run2_length, 2)
            run3 = np.full(run3_length, 3)

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
            stim_list = np.full(study_bold.shape[3], 0)
            maintain_list = np.where(
                (reg_operation == 1) & ((reg_present == 2) | (reg_present == 3))
            )
            suppress_list = np.where(
                (reg_operation == 3) & ((reg_present == 2) | (reg_present == 3))
            )
            replace_list = np.where(
                (reg_operation == 2) & ((reg_present == 2) | (reg_present == 3))
            )

            # this is KEY, if you dont use the .copy() ending, then it links these together such that the changes below occur across ALL three lists
            maintain_labels = stim_list.copy()
            suppress_labels = stim_list.copy()
            replace_labels = stim_list.copy()

            maintain_labels[maintain_list] = 1
            suppress_labels[suppress_list] = 1
            replace_labels[replace_list] = 1

            maintain_labels = maintain_labels[:, None]
            suppress_labels = suppress_labels[:, None]
            replace_labels = replace_labels[:, None]

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
            maintain_labels_shift = shift_timing(
                maintain_labels, shift_size, tag
            )  # rest is label 0
            suppress_labels_shift = shift_timing(
                suppress_labels, shift_size, tag
            )  # rest is label 0
            replace_labels_shift = shift_timing(
                replace_labels, shift_size, tag
            )  # rest is label 0

            operation_labels = {
                "maintain": maintain_labels_shift,
                "suppress": suppress_labels_shift,
                "replace": replace_labels_shift,
            }
            import random

            # parameters for the searchlight:
            # data = The brain data as a 4D volume.
            # mask = A binary mask specifying the "center" voxels in the brain around which you want to perform searchlight analyses. A searchlight will be drawn around every voxel with the value of 1. Hence, if you chose to use the wholebrain mask as the mask for the searchlight procedure, the searchlight may include voxels outside of your mask when the "center" voxel is at the border of the mask. It is up to you to decide whether then to include these results.
            # bcvar = An additional variable which can be a list, numpy array, dictionary, etc. you want to use in your searchlight kernel. For instance you might want the condition labels so that you can determine to which condition each 3D volume corresponds. If you don't need to broadcast anything, e.g, when doing RSA, set this to 'None'.
            # sl_rad = The size of the searchlight's radius, excluding the center voxel. This means the total volume size of the searchlight, if using a cube, is defined as: ((2 * sl_rad) + 1) ^ 3.
            # max_blk_edge = When the searchlight function carves the data up into chunks, it doesn't distribute only a single searchlight's worth of data. Instead, it creates a block of data, with the edge length specified by this variable, which determines the number of searchlights to run within a job.
            # pool_size = Maximum number of cores running on a block (typically 1).

            # make a small mask to debug!
            small_mask = np.zeros(gm_mask.shape)
            small_mask[18:23, 22:27, 32:37] = 1

            # Preset the variables to be used in the searchlight
            data = study_bold.get_fdata()  # need this as 4D data
            mask = gm_mask.get_fdata()  # this is the group GM mask
            # to debug set the mask to the small_mask

            sl_rad = 3  # Searchlight radius
            max_blk_edge = 30  # blocks of data
            pool_size = 1  # Cores running on a block

            affine_mat = study_bold.affine
            dimsize = study_bold.header.get_zooms()

            # Create the searchlight object
            sl = Searchlight(
                sl_rad=sl_rad, max_blk_edge=max_blk_edge, shape=Ball
            )  # this runs the SL as a ball and not cube
            print("Setup searchlight inputs")
            print("Input data shape: " + str(data.shape))
            print("Input mask shape: " + str(mask.shape) + "\n")

            # Set up the kernel to be used in Searchlight
            def calc_L2(data, sl_mask, myrad, bcvar):
                # Pull out the data
                data4D = data[0]
                labels = bcvar

                bolddata_sl = data4D.reshape(
                    sl_mask.shape[0] * sl_mask.shape[1] * sl_mask.shape[2],
                    data[0].shape[3],
                ).T

                # # Check if the number of voxels is what you expect.
                # print("Searchlight data shape: " + str(data[0].shape))
                # print("Searchlight data shape after reshaping: " + str(bolddata_sl.shape))
                # print("Searchlight mask shape:" + str(sl_mask.shape) + "\n")
                # print("Searchlight mask (note that the center equals 1):\n" + str(sl_mask) + "\n")

                t1 = time.time()
                clf = LogisticRegression(solver="liblinear")

                # this actually should work as intended, since I am running the searchlight for maintain, replace and suppress separately (e.g., a regressor list where maintain is labeled with a 1 and all other conditions as a 0, since we run the LogisticRegression as a 1vs.the-rest)
                scores = cross_val_score(
                    clf, bolddata_sl, labels, scoring="roc_auc", cv=3
                )  # Study session has three runs, so this should split the data properly... but will need to decide on that
                # adjusted the scoring system to roc_auc, since just using accuracy was a wrong approach since the samples are not balanced

                accuracy = scores.mean()
                t2 = time.time()

                print("Kernel duration: %.2f\n\n" % (t2 - t1))

                return accuracy

            # Distribute the information to the searchlights (preparing it to run)
            # This also knows how to handle the MPI information to split this up depending on the HPC parameters
            print("distributing information to searchlights")
            sl.distribute([data], mask)
            for key in operation_labels:
                start_time = time.time()
                bcvar = operation_labels[
                    key
                ]  # this is the labels to determine which condition each 3D volume corresponds to (Operation decoding)

                # Data that is needed for all searchlights is sent to all cores via the sl.broadcast function. In this, we are sending the labels for classification to all searchlights.
                sl.broadcast(bcvar)
                # time to execute the searchlight
                print("Begin Searchlight\n")
                sl_result = sl.run_searchlight(calc_L2, pool_size=pool_size)
                print("End Searchlight\n")
                # Print outputs
                print("Summarize searchlight results")
                print("Number of searchlights run: " + str(len(sl_result[mask == 1])))
                print(
                    "Accuracy for each kernel function: "
                    + str(sl_result[mask == 1].astype("double"))
                )
                end_time = time.time()
                print("Total searchlight duration: %.2f" % (end_time - start_time))

                # Save the results to a .nii file
                output_dir = os.path.join(container_path, sub, "searchlight")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_name = os.path.join(
                    output_dir, ("Sub-0%s_SL_%s_result.nii.gz" % (sub_num, key))
                )
                sl_result = sl_result.astype(
                    "double"
                )  # Convert the output into a precision format that can be used by other applications
                sl_result[
                    np.isnan(sl_result)
                ] = 0  # Exchange nans with zero to ensure compatibility with other applications
                sl_nii = nib.Nifti1Image(
                    sl_result, affine_mat
                )  # create the volume image
                hdr = sl_nii.header  # get a handle of the .nii file's header
                hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2]))
                nib.save(sl_nii, output_name)  # Save the volume
