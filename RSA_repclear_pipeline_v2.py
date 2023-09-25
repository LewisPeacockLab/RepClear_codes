import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import re
from joblib import Parallel, delayed

subs = [
    # "02",
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
sub_cates = {"scene": ["manmade", "natural"]}  # 120
rois = ["Prefrontal_ROI", "Higher_Order_Visual_ROI"]


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


def get_scene_orders(param_dir, subID, sub_cates):
    """
    Retrieves the scene orders for pre-localizer, study, and post-localizer phases for a given subject.

    Parameters:
    param_dir (str): The directory where the trial-image match CSV files are located.
    subID (str): Subject ID.
    sub_cates (dict): A dictionary containing the subcategories.

    Returns:
    pre_scene_order (DataFrame): Scene order for the pre-localizer phase.
    study_scene_order (DataFrame): Scene order for the study phase.
    post_scene_order (DataFrame): Scene order for the post-localizer phase.
    """

    # Define the path to the trial-image match CSV file
    tim_path = os.path.join(param_dir, f"sub-0{subID}_trial_image_match.csv")

    # Load the trial-image match data
    tim_df = pd.read_csv(tim_path)

    # Filter and sort the data for the pre-localizer phase
    tim_df_pre = tim_df[tim_df["phase"] == 2].sort_values(
        by=["category", "subcategory", "trial_id"]
    )
    pre_scene_order = tim_df_pre[tim_df_pre["category"] == 1][
        ["trial_id", "image_id", "condition", "subcategory"]
    ]

    # Filter and sort the data for the study phase
    tim_df_study = tim_df[tim_df["phase"] == 3].sort_values(
        by=["category", "subcategory", "trial_id"]
    )
    study_scene_order = tim_df_study[tim_df_study["category"] == 1][
        ["trial_id", "image_id", "condition", "subcategory"]
    ]

    # Filter and sort the data for the post-localizer phase
    tim_df_post = tim_df[tim_df["phase"] == 4].sort_values(
        by=["category", "subcategory", "trial_id"]
    )
    post_scene_order = tim_df_post[tim_df_post["category"] == 1][
        ["trial_id", "image_id", "condition"]
    ]

    return pre_scene_order, study_scene_order, post_scene_order


def load_pre_post_localizer_data(
    container_path, subID, brain_flag, param_dir, sub_cates, roi
):
    """
    Load preprocessed BOLDs for pre-localizer and post-localizer phases.

    Parameters:
    container_path (str): The path to the data container
    subID (str): The subject ID
    brain_flag (str): The flag indicating the brain region
    param_dir (str): The directory where parameter files like trial_image_match.csv are stored
    sub_cates (dict): A dictionary containing the subcategories

    Returns:
    masked_bolds_arr_1 (array): The array of masked BOLDs for prelocalizer
    masked_bolds_arr_3 (array): The array of masked BOLDs for postlocalizer
    """

    # Load the trial-image match data from CSV
    tim_path = os.path.join(param_dir, f"sub-0{subID}_trial_image_match.csv")
    tim_df = pd.read_csv(tim_path)

    # Extract prelocalizer and postlocalizer data
    tim_df_pre = tim_df[tim_df["phase"] == 2]
    tim_df_post = tim_df[tim_df["phase"] == 4]

    # Determine the mask path based on the brain_flag and roi
    if brain_flag == "MNI":
        if roi == "Prefrontal_ROI":
            mask_path = "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/group_MNI_Prefrontal_ROI.nii.gz"
        elif roi == "Higher_Order_Visual_ROI":
            mask_path = "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/group_MNI_Higher_Order_Visual_ROI.nii.gz"
        else:
            mask_path = os.path.join(container_path, "group_MNI_VTC_mask.nii.gz")

    # Load the mask
    mask = nib.load(mask_path)

    # Logic for loading prelocalizer BOLDs
    bold_dir_1 = os.path.join(
        container_path, f"sub-0{subID}", f"item_representations_{roi}_{brain_flag}"
    )
    print(f"Loading preprocessed BOLDs for pre-localizer...")
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

    # Logic for loading postlocalizer BOLDs
    bold_dir_3 = os.path.join(
        container_path, f"sub-0{subID}", f"item_representations_{roi}_{brain_flag}"
    )
    print(f"Loading preprocessed BOLDs for post-localizer...")
    all_bolds_3 = {}  # {cateID: {trialID: bold}}
    bolds_arr_3 = []  # sample x vox
    for cateID in sub_cates.keys():
        cate_bolds_fnames_3 = glob.glob(f"{bold_dir_1}/*post*{cateID}*")
        cate_bolds_3 = {}

        for fname in cate_bolds_fnames_3:
            trialID = fname.split("/")[-1].split("_")[-2]  # "trial1"
            trialID = int(trialID[5:])
            cate_bolds_3[trialID] = nib.load(fname).get_fdata()  # .flatten()
        cate_bolds_3 = {i: cate_bolds_3[i] for i in sorted(cate_bolds_3.keys())}
        all_bolds_3[cateID] = cate_bolds_3

        bolds_arr_3.append(
            np.stack([cate_bolds_3[i] for i in sorted(cate_bolds_3.keys())])
        )

    bolds_arr_3 = np.vstack(bolds_arr_3)
    print("bolds for prelocalizer - shape: ", bolds_arr_3.shape)

    # Apply the mask on the BOLD data
    masked_bolds_arr_1 = []
    for bold in bolds_arr_1:
        masked_bolds_arr_1.append(
            apply_mask(mask=mask.get_fdata(), target=bold).flatten()
        )
    masked_bolds_arr_1 = np.vstack(masked_bolds_arr_1)
    print("masked prelocalizer bold array shape: ", masked_bolds_arr_1.shape)

    masked_bolds_arr_3 = []
    for bold in bolds_arr_3:
        masked_bolds_arr_3.append(
            apply_mask(mask=mask.get_fdata(), target=bold).flatten()
        )
    masked_bolds_arr_3 = np.vstack(masked_bolds_arr_3)
    print("masked postlocalizer bold array shape: ", masked_bolds_arr_3.shape)

    return masked_bolds_arr_1, masked_bolds_arr_3


def load_weights(brain_flag, subID, ROI, mask, sub_cates, container_path):
    # Initialize the directories for the weights based on the brain_flag, subID, and ROI
    if brain_flag == "MNI":
        cate_weights_dir = os.path.join(
            f"/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/sub-0{subID}",
            f"preremoval_lvl1_{brain_flag}_{ROI}/scene_stimuli_MNI_zmap.nii.gz",
        )
        item_weights_dir_suffix = "MNI" if ROI == "VTC" else f"MNI_{ROI}"
        item_weights_dir = os.path.join(
            container_path,
            f"sub-0{subID}",
            f"preremoval_item_level_{item_weights_dir_suffix}",
        )
    else:
        cate_weights_dir = os.path.join(
            f"/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/sub-0{subID}",
            f"preremoval_lvl1_{brain_flag}_{ROI}/scene_stimuli_T1w_zmap.nii.gz",
        )
        item_weights_dir = os.path.join(
            container_path, f"sub-0{subID}", f"preremoval_item_level_T1w_{ROI}"
        )

    # Load category weights and mask
    cate_weights_arr = nib.load(cate_weights_dir).get_fdata()
    masked_cate_weights_arr = apply_mask(
        mask=mask.get_fdata(), target=cate_weights_arr
    ).flatten()

    # Load item weights and mask
    all_weights = {}
    weights_arr = []
    for cateID in sub_cates.keys():
        item_weights_fnames = glob.glob(f"{item_weights_dir}/{cateID}*full*zmap*")
        item_weights = {}
        for fname in item_weights_fnames:
            trialID = fname.split("/")[-1].split("_")[1]
            trialID = int(trialID[5:])
            item_weights[trialID] = nib.load(fname).get_fdata()

        # Sort and stack the weights
        sorted_item_weights = [item_weights[i] for i in sorted(item_weights.keys())]
        weights_arr.append(np.stack(sorted_item_weights))

    # Apply mask on the item weights
    weights_arr = np.vstack(weights_arr)
    masked_weights_arr = []
    for weight in weights_arr:
        masked_weights_arr.append(
            apply_mask(mask=mask.get_fdata(), target=weight).flatten()
        )
    masked_weights_arr = np.vstack(masked_weights_arr)

    return masked_cate_weights_arr, masked_weights_arr


def apply_weighting_to_bold(
    pre_scene_order,
    study_scene_order,
    post_scene_order,
    masked_bolds_arr_1,
    masked_bolds_arr_2,
    masked_bolds_arr_3,
    masked_item_weights_arr,
    masked_cate_weights_arr,
):
    # Initialize dictionaries to hold item-weighted and category-weighted BOLD data
    item_weighted_data = []
    cate_weighted_data = []
    trial_info = []

    # Loop over each 'study' trial to find matching 'pre' and 'post' trials
    for trial in study_scene_order["trial_id"].values:
        study_trial_index = study_scene_order.index[
            study_scene_order["trial_id"] == trial
        ].tolist()[0]
        study_image_id = study_scene_order.loc[study_trial_index, "image_id"]
        image_condition = study_scene_order.loc[study_trial_index, "condition"]

        # Find corresponding 'pre' trial
        pre_trial_index = pre_scene_order.index[
            pre_scene_order["image_id"] == study_image_id
        ].tolist()[0]
        pre_trial_num = pre_scene_order.loc[pre_trial_index, "trial_id"]

        # Find corresponding 'post' trial
        post_trial_index = post_scene_order.index[
            post_scene_order["image_id"] == study_image_id
        ].tolist()[0]
        post_trial_num = post_scene_order.loc[post_trial_index, "trial_id"]

        # Debugging print statements
        print(f"For study trial with image ID {study_image_id}:")
        print(f"  Corresponding pre-trial number is {pre_trial_num}")
        print(f"  Corresponding post-trial number is {post_trial_num}")

        # Confirm that the pre and post image IDs are the same as the study image ID
        pre_image_id_check = pre_scene_order.loc[pre_trial_index, "image_id"]
        post_image_id_check = post_scene_order.loc[post_trial_index, "image_id"]

        # Additional debugging print statements
        print(f"  Confirming image ID for pre-trial: {pre_image_id_check}")
        print(f"  Confirming image ID for post-trial: {post_image_id_check}")

        # Apply item-based weights
        item_weighted_pre = np.multiply(
            masked_bolds_arr_1[pre_trial_num - 1, :],
            masked_item_weights_arr[pre_trial_num - 1, :],
        )
        item_weighted_post = np.multiply(
            masked_bolds_arr_3[post_trial_num - 1, :],
            masked_item_weights_arr[pre_trial_num - 1, :],
        )

        # Apply category-based weights
        cate_weighted_pre = np.multiply(
            masked_bolds_arr_1[pre_trial_num - 1, :], masked_cate_weights_arr
        )
        cate_weighted_post = np.multiply(
            masked_bolds_arr_3[post_trial_num - 1, :], masked_cate_weights_arr
        )

        # Store in list
        item_weighted_data.append(
            {"trial_id": trial, "pre": item_weighted_pre, "post": item_weighted_post}
        )
        cate_weighted_data.append(
            {"trial_id": trial, "pre": cate_weighted_pre, "post": cate_weighted_post}
        )
        trial_info.append(
            {
                "trial_id": trial,
                "image_id": study_image_id,
                "image_condition": image_condition,
            }
        )

    # Convert lists to DataFrames for easier manipulation later
    item_weighted_df = pd.DataFrame(item_weighted_data)
    cate_weighted_df = pd.DataFrame(cate_weighted_data)
    trial_info_df = pd.DataFrame(trial_info)

    return item_weighted_df, cate_weighted_df, trial_info_df


def calculate_and_save_fidelities(
    item_weighted_df, cate_weighted_df, trial_info, save_path, roi_name, brain_flag
):
    # Initialize lists to hold fidelities
    iw_fidelity_list = []
    cw_fidelity_list = []

    # Counter for debugging purposes
    counter = 0

    pre_weighted_iw = item_weighted_df["pre"]
    post_weighted_iw = item_weighted_df["post"]
    pre_weighted_cw = cate_weighted_df["pre"]
    post_weighted_cw = cate_weighted_df["post"]

    for trial_id, image_id, image_condition in zip(
        trial_info["trial_id"], trial_info["image_id"], trial_info["image_condition"]
    ):
        # Calculate the fidelity from pre to post (item_weighted)
        pre_post_iw_fidelity = np.corrcoef(
            pre_weighted_iw.loc[trial_id], post_weighted_iw.loc[trial_id]
        )[0, 1]

        # Calculate the fidelity from pre to post (category_weighted)
        pre_post_cw_fidelity = np.corrcoef(
            pre_weighted_cw.loc[trial_id], post_weighted_cw.loc[trial_id]
        )[0, 1]

        # Map the image_condition to its textual representation
        condition_map = {1: "Maintain", 2: "Replace", 3: "Suppress"}
        operation = condition_map[image_condition]

        # Store fidelities in lists of dictionaries
        iw_fidelity_list.append(
            {
                "Image_ID": image_id,
                "Operation": operation,
                "Fidelity": pre_post_iw_fidelity,
            }
        )

        # Store CW fidelities in list
        cw_fidelity_list.append(
            {
                "Image_ID": image_id,
                "Operation": operation,
                "Fidelity": pre_post_cw_fidelity,
            }
        )

        counter += 1  # Increment counter

    # Convert lists to DataFrames
    iw_fidelity_df = pd.DataFrame(iw_fidelity_list)
    cw_fidelity_df = pd.DataFrame(cw_fidelity_list)

    # Save as CSV files
    iw_fidelity_df.to_csv(f"{save_path}/{roi_name}_IW_Fidelity.csv", index=False)
    cw_fidelity_df.to_csv(f"{save_path}/{roi_name}_CW_Fidelity.csv", index=False)

    return iw_fidelity_df, cw_fidelity_df


def segregate_by_memory_outcome(
    item_weighted_df,
    cate_weighted_df,
    trial_info_df,
    container_path,
    brain_flag,
    subID,
    roi_name,
):
    memory_csv_path = f"/scratch/06873/zbretton/repclear_dataset/BIDS/params/memory_and_familiar_sub-0{subID}.csv"
    # Load memory outcome data
    memory_csv = pd.read_csv(memory_csv_path)

    # Initialize empty DataFrames to hold segregated data
    itemw_remembered_df = pd.DataFrame(columns=["Image_ID", "Operation", "Fidelity"])
    itemw_forgot_df = pd.DataFrame(columns=["Image_ID", "Operation", "Fidelity"])

    catew_remembered_df = pd.DataFrame(columns=["Image_ID", "Operation", "Fidelity"])
    catew_forgot_df = pd.DataFrame(columns=["Image_ID", "Operation", "Fidelity"])

    # Loop through each trial
    for index, row in trial_info_df.iterrows():
        # Extract trial information
        trial_id = row["trial_id"]
        image_id = row["image_id"]
        image_condition = row["image_condition"]

        # Find corresponding memory outcome
        memory_outcome = memory_csv.loc[
            memory_csv["image_num"] == image_id, "memory"
        ].values[0]

        # Identify the operation for this trial
        condition_map = {1: "Maintain", 2: "Replace", 3: "Suppress"}
        operation = condition_map[image_condition]

        # Segregate item-weighted data
        itemw_data = item_weighted_df[
            (item_weighted_df["Image_ID"] == image_id)
            & (item_weighted_df["Operation"] == operation)
        ]
        if memory_outcome == 1:
            itemw_remembered_df = pd.concat(
                [itemw_remembered_df, itemw_data], ignore_index=True
            )
        else:
            itemw_forgot_df = pd.concat(
                [itemw_forgot_df, itemw_data], ignore_index=True
            )

        # Segregate category-weighted data
        catew_data = cate_weighted_df[
            (cate_weighted_df["Image_ID"] == image_id)
            & (cate_weighted_df["Operation"] == operation)
        ]
        if memory_outcome == 1:
            catew_remembered_df = pd.concat(
                [catew_remembered_df, catew_data], ignore_index=True
            )
        else:
            catew_forgot_df = pd.concat(
                [catew_forgot_df, catew_data], ignore_index=True
            )

    # Save the segregated DataFrames to CSV files
    itemw_remembered_df.to_csv(
        os.path.join(
            container_path,
            f"sub-0{subID}",
            f"Representational_Changes_{brain_flag}_{roi_name}",
            "itemweighted_remembered_fidelity.csv",
        )
    )
    itemw_forgot_df.to_csv(
        os.path.join(
            container_path,
            f"sub-0{subID}",
            f"Representational_Changes_{brain_flag}_{roi_name}",
            "itemweighted_forgot_fidelity.csv",
        )
    )

    catew_remembered_df.to_csv(
        os.path.join(
            container_path,
            f"sub-0{subID}",
            f"Representational_Changes_{brain_flag}_{roi_name}",
            "cateweighted_remembered_fidelity.csv",
        )
    )
    catew_forgot_df.to_csv(
        os.path.join(
            container_path,
            f"sub-0{subID}",
            f"Representational_Changes_{brain_flag}_{roi_name}",
            "cateweighted_forgot_fidelity.csv",
        )
    )

    return itemw_remembered_df, itemw_forgot_df, catew_remembered_df, catew_forgot_df


def rsa_pipeline_for_new_ROIs(subID, roi, brain_flag="MNI"):
    # Initial configurations
    container_path = (
        "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep"
    )
    param_dir = "/scratch/06873/zbretton/repclear_dataset/BIDS/params"

    save_path = os.path.join(
        container_path, f"sub-0{subID}", f"Representational_Changes_{brain_flag}_{roi}"
    )
    # if os.path.exists(f"{save_path}/cateweighted_forgot_fidelity.csv"):
    #     print(
    #         f"Skipping RSA pipeline for sub-0{subID} and ROI {roi} as it has already been run."
    #     )
    #     return
    mkdir(save_path)

    # Get the scene orders for pre, study, and post phases
    pre_scene_order, study_scene_order, post_scene_order = get_scene_orders(
        param_dir=param_dir, subID=subID, sub_cates=sub_cates
    )

    # Load the preprocessed BOLDs for pre-localizer and post-localizer phases
    masked_bolds_arr_1, masked_bolds_arr_3 = load_pre_post_localizer_data(
        container_path=container_path,
        subID=subID,
        brain_flag=brain_flag,
        param_dir=param_dir,
        sub_cates=sub_cates,
        roi=roi,
    )

    # Load the weights
    mask_path = os.path.join(
        container_path,
        f"group_MNI_{roi}.nii.gz",
    )
    mask = nib.load(mask_path)
    masked_cate_weights_arr, masked_item_weights_arr = load_weights(
        brain_flag=brain_flag,
        subID=subID,
        ROI=roi,
        mask=mask,
        sub_cates=sub_cates,
        container_path=container_path,
    )

    # Apply weighting to BOLD signals
    item_weighted_df, cate_weighted_df, trial_info_df = apply_weighting_to_bold(
        pre_scene_order=pre_scene_order,
        study_scene_order=study_scene_order,
        post_scene_order=post_scene_order,
        masked_bolds_arr_1=masked_bolds_arr_1,
        masked_bolds_arr_2=None,  # We don't have masked_bolds_arr_2 in this version
        masked_bolds_arr_3=masked_bolds_arr_3,
        masked_item_weights_arr=masked_item_weights_arr,
        masked_cate_weights_arr=masked_cate_weights_arr,
    )

    item_weighted_df.set_index("trial_id", inplace=True)
    cate_weighted_df.set_index("trial_id", inplace=True)

    # Calculate and save fidelities
    iw_fidelity_df, cw_fidelity_df = calculate_and_save_fidelities(
        item_weighted_df=item_weighted_df,
        cate_weighted_df=cate_weighted_df,
        trial_info=trial_info_df,
        save_path=save_path,
        roi_name=roi,
        brain_flag=brain_flag,
    )

    # Segregate by memory outcome
    (
        itemw_remembered_df,
        itemw_forgot_df,
        catew_remembered_df,
        catew_forgot_df,
    ) = segregate_by_memory_outcome(
        item_weighted_df=iw_fidelity_df,
        cate_weighted_df=cw_fidelity_df,
        trial_info_df=trial_info_df,
        container_path=container_path,
        brain_flag=brain_flag,
        subID=subID,
        roi_name=roi,
    )

    print(f"RSA pipeline for sub-0{subID} and ROI {roi} completed.")


# Parallel execution
Parallel(n_jobs=-1, verbose=1)(
    delayed(rsa_pipeline_for_new_ROIs)(sub_num, roi, brain_flag="MNI")
    for sub_num in subs
    for roi in rois
)
