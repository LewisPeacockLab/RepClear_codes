# RSA item-level timecourse decoding
import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import pearsonr
from joblib import Parallel, delayed
import glob
import re

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

param_dir = "/scratch/06873/zbretton/repclear_dataset/BIDS/params"
container_path = "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep"


def fisher_z_transform(r):
    # Avoid division by zero or taking arctanh of 1
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r)


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
        container_path,
        f"sub-0{subID}",
        f"item_representations_MNI",  # to load timecourses
    )
    masked_bolds_arr_2 = {}
    for cateID in sub_cates.keys():
        cate_bolds_fnames_2 = glob.glob(f"{bold_dir_2}/*study*timecourse*{cateID}*")
        cate_bolds_fnames_2.sort(key=lambda fname: int(fname.split("_")[-2][5:]))
        # Load each timecourse file and apply the mask
        for fname in cate_bolds_fnames_2:
            trial_number = int(re.search(r"trial(\d+)", fname).group(1))
            timecourse_data = nib.load(fname).get_fdata()
            masked_timecourse_data = apply_mask(mask=mask_data, target=timecourse_data)
            masked_bolds_arr_2[trial_number] = masked_timecourse_data

    # Apply mask on prelocalizer BOLD
    masked_bolds_arr_1 = [
        apply_mask(mask=mask_data, target=bold) for bold in bolds_arr_1
    ]
    masked_bolds_arr_1 = np.vstack(masked_bolds_arr_1)

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
    Compute the fidelity between prelocalizer and study phase timecourses.

    Parameters:
        masked_bolds_arr_1 (ndarray): Masked BOLD data for prelocalizer
        masked_bolds_arr_2 (dictionary): Masked BOLD data for study timecourses
        study_scene_order (DataFrame): Study scene order
        pre_scene_order (DataFrame): Prelocalizer scene order
        sub_cates (dict): Subject categories
        roi (str): Region of Interest
        container_path (str): Container path for fMRI data
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

    # weights_arr = np.vstack(weights_arr)

    # Apply mask on weights
    masked_weights_arr = [
        apply_mask(mask=mask.get_fdata(), target=weight).flatten()
        for weight in weights_arr
    ]
    masked_weights_arr = np.vstack(masked_weights_arr)

    # Multiply prelocalizer patterns and prelocalizer item weights
    item_repress_pre = np.multiply(masked_bolds_arr_1, masked_weights_arr)

    # Initialize dictionaries to store RSA and Fisher Z transformed RSA values
    rsa_results = {"maintain": [], "replace": [], "suppress": []}
    rsa_results_z = {"maintain": [], "replace": [], "suppress": []}

    for index, study_row in study_scene_order.iterrows():
        study_trial_id = study_row["trial_id"]
        study_image_id = study_row["image_id"]
        study_condition = study_row["condition"]

        # Find the corresponding prelocalizer trial
        pre_row = pre_scene_order[pre_scene_order["image_id"] == study_image_id].iloc[0]
        pre_trial_id = pre_row["trial_id"]

        # Get the prelocalizer data for this image
        pre_data = item_repress_pre[pre_trial_id - 1]

        # Get and weight the study timecourse
        study_timecourse = masked_bolds_arr_2[study_trial_id]
        study_timecourse_weighted = np.array(
            [tr * masked_weights_arr[pre_trial_id - 1] for tr in study_timecourse]
        )

        # Compute RSA values for each TR in the timecourse
        rsa_values = [
            pearsonr(pre_data, tr_weighted)[0]
            for tr_weighted in study_timecourse_weighted
        ]
        rsa_values_z = [
            fisher_z_transform(r) for r in rsa_values
        ]  # Apply Fisher Z transformation

        # Map condition to string for readability
        condition_str = {1: "maintain", 2: "replace", 3: "suppress"}.get(
            study_condition, "unknown"
        )

        # Store RSA values by condition
        rsa_results[condition_str].append(rsa_values)
        rsa_results_z[condition_str].append(rsa_values_z)

    # Calculate mean timecourses for normal RSA values
    mean_timecourses = {
        condition: np.mean(rsa_values, axis=0)
        for condition, rsa_values in rsa_results.items()
    }

    # Calculate mean timecourses for Fisher Z transformed RSA values
    mean_timecourses_z = {
        condition: np.mean(rsa_values_z, axis=0)
        for condition, rsa_values_z in rsa_results_z.items()
    }

    return mean_timecourses, mean_timecourses_z


# Initialize an empty DataFrame to collect data from all subjects
all_subjects_data = pd.DataFrame()

# Loop through each subject
for subID in subs:
    # Load pre and study scene orders
    pre_scene_order, study_scene_order = get_pre_and_study_scene_order(subID, param_dir)

    # Load and mask BOLD data
    masked_bolds_arr_1, masked_bolds_arr_2 = load_and_mask_bold_data(
        roi="VTC_mask",  # or any other ROI you're interested in
        subID=subID,
        sub_cates=sub_cates,
        container_path=container_path,
    )

    # Compute fidelity (RSA) and get mean timecourses
    mean_timecourses, mean_timecourses_z = compute_fidelity(
        masked_bolds_arr_1,
        masked_bolds_arr_2,
        study_scene_order,
        pre_scene_order,
        sub_cates,
        roi="VTC_mask",  # or any other ROI
        container_path=container_path,
    )

    # Loop through each condition to append data
    for condition, timecourse in mean_timecourses_z.items():
        # Create a DataFrame for this condition and subject
        temp_df = pd.DataFrame(
            {
                "TR": np.arange(
                    1, len(timecourse) + 1
                ),  # Directly use numerical TR values
                "RSA_Value": timecourse,
                "Condition": condition,
                "Subject": subID,
            }
        )

        # Append to the collective DataFrame
        all_subjects_data = pd.concat([all_subjects_data, temp_df], ignore_index=True)

# After the loop
all_subjects_data.to_csv(
    os.path.join(container_path, "grouplevel_RSAstudy_timecourse_fisherz.csv"),
    index=False,
)

# Ensure the data types are correct
all_subjects_data["TR"] = all_subjects_data["TR"].astype(int)
all_subjects_data["RSA_Value"] = all_subjects_data["RSA_Value"].astype(float)

# Basic Line Plot
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=all_subjects_data,
    x="TR",
    y="RSA_Value",
    hue="Condition",
    palette={"maintain": "green", "replace": "blue", "suppress": "red"},
    ci="sd",
)  # ci='sd' will plot standard deviation as the confidence interval
plt.title("Group Level Timecourse by Condition")
plt.xlabel("Timepoint (TR)")
plt.ylabel("RSA Value")
plt.legend(title="Condition")
plt.tight_layout()
