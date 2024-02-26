# This code is to load in the representations and then either weight them (category vs. item) and then perform RSA
# This will also handle performing this and then comparing Pre-Localizer to Study, so that we can see same vs. other

# Imports
import glob
import nibabel as nib
import numpy as np
import os
import pandas as pd
import scipy
from scipy import stats
import sys
import warnings
import fnmatch

from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    PredefinedSplit,
    cross_validate,
    cross_val_predict,
    GridSearchCV,
    LeaveOneGroupOut,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


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


def get_pre_and_study_scene_order(subID, param_dir):
    """
    Retrieves pre-scene and study-scene order based on the subject ID and parameter directory.

    Parameters:
        subID (str): Subject ID
        param_dir (str): Directory path to the parameter files

    Returns:
        pre_scene_order (DataFrame): DataFrame containing the order of scenes in the pre-localizer phase
        study_scene_order (DataFrame): DataFrame containing the order of scenes in the study phase
    """

    print(f"Running sub-0{subID}...")

    # Define the path to the trial image match CSV file based on the subject ID and parameter directory
    tim_path = os.path.join(param_dir, f"sub-0{subID}_trial_image_match.csv")

    # Read the CSV file into a DataFrame and filter it to include only the pre-localizer phase (phase 2)
    tim_df = pd.read_csv(tim_path)
    tim_df = tim_df[tim_df["phase"] == 2]  # Phase 2 is the pre-localizer
    tim_df = tim_df.sort_values(by=["category", "subcategory", "trial_id"])

    # Extract pre_scene_order from the filtered DataFrame
    pre_scene_order = tim_df[tim_df["category"] == 1][
        ["trial_id", "image_id", "condition", "subcategory"]
    ]

    # Read the CSV file into another DataFrame and filter it to include only the study phase (phase 3)
    tim_df2 = pd.read_csv(tim_path)
    tim_df2 = tim_df2[tim_df2["phase"] == 3]  # Phase 3 is the study
    tim_df2 = tim_df2.sort_values(by=["category", "subcategory", "trial_id"])

    # Extract study_scene_order from the second filtered DataFrame
    study_scene_order = tim_df2[tim_df2["category"] == 1][
        ["trial_id", "image_id", "condition", "subcategory"]
    ]

    return pre_scene_order, study_scene_order


def load_and_mask_bold_data(
    roi,
    subID,
    sub_cates,
    container_path="/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep",
):
    """
    Load and mask BOLD data for prelocalizer and study phases.

    Parameters:
        roi (str): Region of Interest
        subID (str): Subject ID
        sub_cates (dict): Subject categories
        container_path (str, optional): Container path for fMRI data

    Returns:
        masked_bolds_arr_1 (ndarray): Masked BOLD data for prelocalizer
        masked_bolds_arr_2 (ndarray): Masked BOLD data for study
    """

    # Load mask for BOLD
    mask_path = os.path.join(f"{container_path}/group_MNI_{roi}.nii.gz")
    mask = nib.load(mask_path)
    mask_data = mask.get_fdata()

    # Load ready BOLD for each trial of prelocalizer
    bold_dir_1 = os.path.join(
        container_path, f"sub-0{subID}", f"item_representations_{roi}_MNI"
    )
    bolds_arr_1 = []
    for cateID in sub_cates.keys():
        cate_bolds_fnames_1 = glob.glob(f"{bold_dir_1}/*pre*{cateID}*")
        cate_bolds_fnames_1.sort(key=lambda fname: int(fname.split("_")[-2][5:]))
        cate_bolds_1 = [nib.load(fname).get_fdata() for fname in cate_bolds_fnames_1]
        bolds_arr_1.extend(cate_bolds_1)

    bolds_arr_1 = np.stack(bolds_arr_1, axis=0)

    # Load ready BOLD for each trial of study
    bold_dir_2 = os.path.join(
        container_path, f"sub-0{subID}", f"item_representations_{roi}_MNI"
    )
    bolds_arr_2 = []
    for cateID in sub_cates.keys():
        cate_bolds_fnames_2 = glob.glob(f"{bold_dir_2}/*study_{cateID}*")
        cate_bolds_fnames_2.sort(key=lambda fname: int(fname.split("_")[-2][5:]))
        cate_bolds_2 = [nib.load(fname).get_fdata() for fname in cate_bolds_fnames_2]
        bolds_arr_2.extend(cate_bolds_2)

    bolds_arr_2 = np.stack(bolds_arr_2, axis=0)

    # Apply mask on prelocalizer BOLD
    masked_bolds_arr_1 = [
        apply_mask(mask=mask_data, target=bold) for bold in bolds_arr_1
    ]
    masked_bolds_arr_1 = np.vstack(masked_bolds_arr_1)

    # Apply mask on study BOLD
    masked_bolds_arr_2 = [
        apply_mask(mask=mask_data, target=bold) for bold in bolds_arr_2
    ]
    masked_bolds_arr_2 = np.vstack(masked_bolds_arr_2)

    return masked_bolds_arr_1, masked_bolds_arr_2


def compute_fidelity(
    masked_bolds_arr_1,
    masked_bolds_arr_2,
    study_scene_order,
    pre_scene_order,
    sub_cates,
    roi,
    container_path,
):
    """
    Compute the fidelity between prelocalizer and study phases.

    Parameters:
        masked_bolds_arr_1 (ndarray): Masked BOLD data for prelocalizer
        masked_bolds_arr_2 (ndarray): Masked BOLD data for study
        study_scene_order (DataFrame): Study scene order
        pre_scene_order (DataFrame): Prelocalizer scene order
        sub_cates (dict): Subject categories
        roi (str): Region of Interest
        container_path (str): Container path for fMRI data

    Returns:
        iw_dict (dict): Dictionary containing fidelity values
    """

    # Load weights
    item_weights_dir = os.path.join(
        container_path, f"sub-0{subID}", f"preremoval_item_level_MNI_{roi}"
    )
    all_weights = {}
    weights_arr = []

    for cateID in sub_cates.keys():
        item_weights_fnames = glob.glob(f"{item_weights_dir}/{cateID}*full*zmap*")
        item_weights = {
            int(fname.split("/")[-1].split("_")[1][5:]): nib.load(fname).get_fdata()
            for fname in item_weights_fnames
        }
        all_weights[cateID] = item_weights
        weights_arr.extend([item_weights[i] for i in sorted(item_weights.keys())])

    weights_arr = np.vstack(weights_arr)

    # Apply mask on weights
    masked_weights_arr = [
        apply_mask(mask=mask.get_fdata(), target=weight).flatten()
        for weight in weights_arr
    ]
    masked_weights_arr = np.vstack(masked_weights_arr)

    # Multiply prelocalizer patterns and prelocalizer item weights
    item_repress_pre = np.multiply(masked_bolds_arr_1, masked_weights_arr)

    # Initialize dictionaries and arrays for storing results
    iw_dict = {}
    counter = 0
    item_repress_study_comp = np.zeros_like(item_repress_pre[:90, :])
    item_repress_pre_comp = np.zeros_like(item_repress_pre[:90, :])

    for trial in study_scene_order["trial_id"].values:
        study_trial_index = study_scene_order.index[
            study_scene_order["trial_id"] == trial
        ].tolist()[0]
        study_image_id = study_scene_order.loc[study_trial_index, "image_id"]

        pre_trial_index = pre_scene_order.index[
            pre_scene_order["image_id"] == study_image_id
        ].tolist()[0]
        pre_trial_num = pre_scene_order.loc[pre_trial_index, "trial_id"]

        item_repress_study_comp[counter] = np.multiply(
            masked_bolds_arr_2[trial - 1, :], masked_weights_arr[pre_trial_num - 1, :]
        )
        item_repress_pre_comp[counter] = item_repress_pre[pre_trial_num - 1, :]

        item_others = []
        for j in range(len(masked_bolds_arr_2)):
            if (trial - 1) != j:
                weighted_other = np.multiply(
                    masked_bolds_arr_2[j - 1, :],
                    masked_weights_arr[pre_trial_num - 1, :],
                )
                temp_fidelity = np.corrcoef(
                    item_repress_pre_comp[counter, :], weighted_other
                )
                item_others.append(temp_fidelity[1][0])

        pre_study_trial_iw_fidelity = np.corrcoef(
            item_repress_pre_comp[counter, :], item_repress_study_comp[counter, :]
        )
        other_fidelity = np.mean(item_others)

        iw_dict[f"image ID: {study_image_id}"] = [
            pre_study_trial_iw_fidelity[1][0],
            other_fidelity,
        ]
        counter += 1

    return iw_dict


def save_results(iw_dict, subID, roi, container_path):
    # Step 1: Convert Dictionary to DataFrame
    temp_df = pd.DataFrame(data=iw_dict).T

    # Construct the directory path where the file will be saved
    save_dir = os.path.join(
        container_path, f"sub-0{subID}", f"Representational_Changes_MNI_{roi}"
    )

    # Save Detailed Results
    temp_df.to_csv(os.path.join(save_dir, "item_weight_pre_study_RSA_w_other.csv"))

    # Compute and Save Averages
    temp_df2 = temp_df.mean().to_frame().T
    temp_df2.columns = ["Fidelity-Same", "Fidelity-Other"]
    temp_df2.to_csv(
        os.path.join(save_dir, "average_item_weight_pre_study_RSA_w_other.csv")
    )

    # Cleanup
    del temp_df, temp_df2

    # Print Completion Message
    print("Subject is done... saving everything")
    print(
        "==============================================================================="
    )


def load_all_data(container_path, roi, subject_ids):
    dataframes = []
    for subID in subject_ids:
        file_path = os.path.join(
            container_path,
            f"sub-0{subID}",
            f"Representational_Changes_MNI_{roi}",
            "item_weight_pre_study_RSA_w_other.csv",
        )
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
            df["subject_id"] = subID
            dataframes.append(df)
    return pd.concat(dataframes, axis=0)


def load_average_data(container_path, roi, subject_ids):
    dataframes = []
    for subID in subject_ids:
        file_path = os.path.join(
            container_path,
            f"sub-0{subID}",
            f"Representational_Changes_MNI_{roi}",
            "average_item_weight_pre_study_RSA_w_other.csv",
        )
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
            df["subject_id"] = subID
            dataframes.append(df)
    return pd.concat(dataframes, axis=0, ignore_index=True)


def perform_subject_level_analysis(container_path, roi, subject_ids):
    all_data = load_all_data(container_path, roi, subject_ids)
    average_data = load_average_data(container_path, roi, subject_ids)

    t_value, p_value = stats.ttest_rel(
        average_data["Fidelity-Same"], average_data["Fidelity-Other"]
    )
    print(
        f"Results of Subject level analysis: T({len(subject_ids) - 1})={t_value}, p = {p_value}!"
    )


def main():
    # Define the ROIs and subject IDs to loop over
    # rois = ["Prefrontal_ROI", "Higher_Order_Visual_ROI", "hippocampus_ROI", "VTC_mask"]
    rois = ["hippocampus_ROI"]
    subject_ids = [
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

    # Define the container path and parameter directory
    container_path = (
        "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep"
    )
    param_dir = "/scratch/06873/zbretton/repclear_dataset/BIDS/params"

    # Loop through each ROI
    for roi in rois:
        print(f"Running for ROI: {roi}")

        # Loop through each subject within the ROI
        for subID in subject_ids:
            print(f"Processing subject: {subID} for ROI: {roi}")

            # Step 1: Get pre and study scene order
            pre_scene_order, study_scene_order = get_pre_and_study_scene_order(
                subID, param_dir
            )

            # Step 2: Load and mask the BOLD data
            masked_bolds_arr_1, masked_bolds_arr_2 = load_and_mask_bold_data(
                roi, subID, sub_cates, container_path
            )

            # Step 3: Compute the fidelity measures
            iw_dict = compute_fidelity(
                masked_bolds_arr_1,
                masked_bolds_arr_2,
                study_scene_order,
                pre_scene_order,
                sub_cates,
                roi,
                container_path,
            )

            # Step 4: Save the results
            save_results(iw_dict, subID, roi, container_path)

        # Step 5: Perform subject-level analysis for the ROI
        perform_subject_level_analysis(container_path, roi, subject_ids)

        print(f"Completed processing for ROI: {roi}")


if __name__ == "__main__":
    main()

# ===========#
# OLD VERSION OF THE CODE:
# for roi in rois:
#     for subID in subs:
#         print("Running sub-0%s..." % subID)
#         # define the subject
#         sub = "sub-0%s" % subID

#         # lets pull out the pre-localizer data here:
#         tim_path = os.path.join(param_dir, f"sub-0{subID}_trial_image_match.csv")

#         tim_df = pd.read_csv(tim_path)
#         tim_df = tim_df[tim_df["phase"] == 2]  # phase 2 is pre-localizer
#         tim_df = tim_df.sort_values(by=["category", "subcategory", "trial_id"])

#         pre_scene_order = tim_df[tim_df["category"] == 1][
#             ["trial_id", "image_id", "condition", "subcategory"]
#         ]
#         pre_face_order = tim_df[tim_df["category"] == 2][
#             ["trial_id", "image_id", "condition"]
#         ]

#         # lets pull out the study data here:
#         tim_df2 = pd.read_csv(tim_path)
#         tim_df2 = tim_df2[tim_df2["phase"] == 3]  # phase 3 is study
#         tim_df2 = tim_df2.sort_values(by=["category", "subcategory", "trial_id"])

#         study_scene_order = tim_df2[tim_df2["category"] == 1][
#             ["trial_id", "image_id", "condition", "subcategory"]
#         ]

#         print(f"Running RSA for sub-0{subID}...")

#         # ===== load mask for BOLD
#         mask_path = os.path.join(
#             f"/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/group_MNI_{roi}.nii.gz"
#         )

#         mask = nib.load(mask_path)

#         print("mask shape: ", mask.shape)

#         # ===== load ready BOLD for each trial of prelocalizer
#         print(f"Loading preprocessed BOLDs for pre-localizer...")
#         bold_dir_1 = os.path.join(
#             container_path, f"sub-0{subID}", f"item_representations_MNI_{roi}"
#         )

#         all_bolds_1 = {}  # {cateID: {trialID: bold}}
#         bolds_arr_1 = []  # sample x vox
#         for cateID in sub_cates.keys():
#             cate_bolds_fnames_1 = glob.glob(f"{bold_dir_1}/*pre*{cateID}*")
#             cate_bolds_1 = {}

#             for fname in cate_bolds_fnames_1:
#                 trialID = fname.split("/")[-1].split("_")[-2]  # "trial1"
#                 trialID = int(trialID[5:])
#                 cate_bolds_1[trialID] = nib.load(fname).get_fdata()  # .flatten()
#             cate_bolds_1 = {i: cate_bolds_1[i] for i in sorted(cate_bolds_1.keys())}
#             all_bolds_1[cateID] = cate_bolds_1

#             bolds_arr_1.append(
#                 np.stack([cate_bolds_1[i] for i in sorted(cate_bolds_1.keys())])
#             )

#         bolds_arr_1 = np.vstack(bolds_arr_1)
#         print("bolds for prelocalizer - shape: ", bolds_arr_1.shape)

#         # ===== load ready BOLD for each trial of study
#         print(f"Loading preprocessed BOLDs for the study operation...")
#         bold_dir_2 = os.path.join(
#             container_path, f"sub-0{subID}", f"item_representations_MNI_{roi}"
#         )

#         all_bolds_2 = {}  # {cateID: {trialID: bold}}
#         bolds_arr_2 = []  # sample x vox
#         for cateID in sub_cates.keys():
#             cate_bolds_fnames_2 = glob.glob(f"{bold_dir_2}/*study_{cateID}*")
#             cate_bolds_2 = {}
#             try:
#                 for fname in cate_bolds_fnames_2:
#                     trialID = fname.split("/")[-1].split("_")[-2]  # "trial1"
#                     trialID = int(trialID[5:])
#                     cate_bolds_2[trialID] = nib.load(fname).get_fdata()  # .flatten()
#                 cate_bolds_2 = {i: cate_bolds_2[i] for i in sorted(cate_bolds_2.keys())}
#                 all_bolds_2[cateID] = cate_bolds_2

#                 bolds_arr_2.append(
#                     np.stack([cate_bolds_2[i] for i in sorted(cate_bolds_2.keys())])
#                 )
#             except:
#                 print("no %s trials" % cateID)
#         bolds_arr_2 = np.vstack(bolds_arr_2)
#         print("bolds for study - shape: ", bolds_arr_2.shape)

#         # apply mask on prelocalizer BOLD
#         masked_bolds_arr_1 = []
#         for bold in bolds_arr_1:
#             masked_bolds_arr_1.append(
#                 apply_mask(mask=mask.get_fdata(), target=bold).flatten()
#             )
#         masked_bolds_arr_1 = np.vstack(masked_bolds_arr_1)
#         print("masked prelocalizer bold array shape: ", masked_bolds_arr_1.shape)

#         # apply mask on study BOLD
#         masked_bolds_arr_2 = []
#         for bold in bolds_arr_2:
#             masked_bolds_arr_2.append(
#                 apply_mask(mask=mask.get_fdata(), target=bold).flatten()
#             )
#         masked_bolds_arr_2 = np.vstack(masked_bolds_arr_2)
#         print("masked study bold array shape: ", masked_bolds_arr_2.shape)

#         # ===== load weights
#         print(f"Loading weights...")
#         # prelocalizer
#         item_weights_dir = os.path.join(
#             container_path, f"sub-0{subID}", f"preremoval_item_level_MNI_{roi}"
#         )

#         # prelocalizer weights (category and item) get applied to study/post representations

#         all_weights = {}
#         weights_arr = []

#         # load in all the item specific weights, which come from the LSA contrasts per subject
#         for cateID in sub_cates.keys():
#             item_weights_fnames = glob.glob(f"{item_weights_dir}/{cateID}*full*zmap*")
#             print(cateID, len(item_weights_fnames))
#             item_weights = {}

#             for fname in item_weights_fnames:
#                 trialID = fname.split("/")[-1].split("_")[1]  # "trial1"
#                 trialID = int(trialID[5:])
#                 item_weights[trialID] = nib.load(fname).get_fdata()
#             item_weights = {i: item_weights[i] for i in sorted(item_weights.keys())}
#             all_weights[cateID] = item_weights
#             weights_arr.append(
#                 np.stack([item_weights[i] for i in sorted(item_weights.keys())])
#             )

#         # this now masks the item weights to ensure that they are all in the same ROI (group VTC):
#         weights_arr = np.vstack(weights_arr)
#         print("weights shape: ", weights_arr.shape)
#         # apply mask on BOLD
#         masked_weights_arr = []
#         for weight in weights_arr:
#             masked_weights_arr.append(
#                 apply_mask(mask=mask.get_fdata(), target=weight).flatten()
#             )
#         masked_weights_arr = np.vstack(masked_weights_arr)
#         print("masked item weights arr shape: ", masked_weights_arr.shape)

#         # ===== multiply
#         # prelocalizer patterns and prelocalizer item weights
#         item_repress_pre = np.multiply(
#             masked_bolds_arr_1, masked_weights_arr
#         )  # these are lined up since the trials goes to correct trials

#         print("item representations pre shape: ", item_repress_pre.shape)

#         # these are used to hold the fidelity changes from pre to study (item-weighted)
#         iw_dict = {}

#         # these are used to hold the fidelity changes from pre to study (scene-weighted)
#         cw_dict = {}

#         counter = 0

#         item_repress_study_comp = np.zeros_like(item_repress_pre[:90, :])
#         item_repress_pre_comp = np.zeros_like(item_repress_pre[:90, :])
#         item_repress_removal_comp = {}

#         for trial in study_scene_order["trial_id"].values:
#             study_trial_index = study_scene_order.index[
#                 study_scene_order["trial_id"] == trial
#             ].tolist()[
#                 0
#             ]  # find the order
#             study_image_id = study_scene_order.loc[
#                 study_trial_index, "image_id"
#             ]  # this now uses the index of the dataframe to find the image_id

#             pre_trial_index = pre_scene_order.index[
#                 pre_scene_order["image_id"] == study_image_id
#             ].tolist()[
#                 0
#             ]  # find the index in the pre for this study trial
#             pre_trial_num = pre_scene_order.loc[pre_trial_index, "trial_id"]

#             image_condition = pre_scene_order.loc[
#                 pre_scene_order["trial_id"] == pre_trial_num
#             ]["condition"].values[0]
#             pre_trial_subcat = pre_scene_order.loc[pre_trial_index, "subcategory"]

#             item_repress_study_comp[counter] = np.multiply(
#                 masked_bolds_arr_2[trial - 1, :],
#                 masked_weights_arr[pre_trial_num - 1, :],
#             )

#             item_repress_pre_comp[counter] = item_repress_pre[pre_trial_num - 1, :]
#             counter = counter + 1

#         item_pre_study_comp = np.corrcoef(
#             item_repress_pre_comp, item_repress_study_comp
#         )

#         item_weighted_pre = np.zeros_like(item_repress_pre[:90, :])
#         item_weighted_study = np.zeros_like(item_repress_pre[:90, :])

#         counter = 0
#         # this loop is limited by the smaller index, so thats the study condition (only 90 stims)
#         for trial in study_scene_order["trial_id"].values:
#             study_trial_index = study_scene_order.index[
#                 study_scene_order["trial_id"] == trial
#             ].tolist()[
#                 0
#             ]  # find the order
#             study_image_id = study_scene_order.loc[
#                 study_trial_index, "image_id"
#             ]  # this now uses the index of the dataframe to find the image_id

#             pre_trial_index = pre_scene_order.index[
#                 pre_scene_order["image_id"] == study_image_id
#             ].tolist()[
#                 0
#             ]  # find the index in the pre for this study trial
#             pre_trial_num = pre_scene_order.loc[pre_trial_index, "trial_id"]

#             # now that I have the link between prelocalizer, study, and postlocalizer I can get that representation weighted with the item weight
#             item_weighted_study[counter] = np.multiply(
#                 masked_bolds_arr_2[trial - 1, :],
#                 masked_weights_arr[pre_trial_num - 1, :],
#             )

#             item_weighted_pre[counter] = item_repress_pre[pre_trial_num - 1, :]

#             item_others = []

#             for j in range(len(masked_bolds_arr_2)):
#                 if (trial - 1) != j:
#                     weighted_other = np.multiply(
#                         masked_bolds_arr_2[j - 1, :],
#                         masked_weights_arr[pre_trial_num - 1, :],
#                     )

#                     temp_fidelity = np.corrcoef(
#                         item_weighted_pre[counter, :], weighted_other
#                     )

#                     item_others.append(temp_fidelity[1][0])

#             # This is to get the fidelity of the current item/trial from pre to study (item_weighted)
#             pre_study_trial_iw_fidelity = np.corrcoef(
#                 item_weighted_pre[counter, :], item_weighted_study[counter, :]
#             )

#             other_fidelity = np.mean(item_others)

#             iw_dict["image ID: %s" % study_image_id] = [
#                 pre_study_trial_iw_fidelity[1][0],
#                 other_fidelity,
#             ]

#             counter = counter + 1

#         temp_df = pd.DataFrame(data=iw_dict).T
#         temp_df.to_csv(
#             os.path.join(
#                 container_path,
#                 f"sub-0{subID}",
#                 f"Representational_Changes_MNI_{roi}",
#                 "item_weight_pre_study_RSA_w_other.csv",
#             )
#         )

#         temp_df2 = temp_df.mean().to_frame().T
#         temp_df2.columns = ["Fidelity-Same", "Fidelity-Other"]
#         temp_df2.to_csv(
#             os.path.join(
#                 container_path,
#                 f"sub-0{subID}",
#                 f"Representational_Changes_MNI_{roi}",
#                 "average_item_weight_pre_study_RSA_w_other.csv",
#             )
#         )

#         del temp_df

#         print("Subject is done... saving everything")
#         print(
#             "==============================================================================="
#         )
