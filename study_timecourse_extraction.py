# Essential imports
import os
import fnmatch
import numpy as np
import nibabel as nib
import pandas as pd
from joblib import Parallel, delayed

# List of subjects
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
brain_flag = "MNI"  # or "T1w"


# Function to create directories if they don't exist
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Function to find files matching a pattern
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


# Main function to extract removal timecourse data
def item_representation_study(subID):
    print(f"Running sub-0{subID}...")
    sub = f"sub-0{subID}"
    container_path = (
        "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep"
    )
    bold_path = os.path.join(container_path, sub, "func/")
    bold_files = find("*study*bold*.nii.gz", bold_path)

    # Mask path determination based on brain_flag
    if brain_flag == "MNI":
        mask_path = os.path.join(container_path, "group_MNI_VTC_mask.nii.gz")
    elif brain_flag == "T1w":
        mask_path = os.path.join(
            container_path, sub, "new_mask", f"VVS_study_{brain_flag}_mask.nii.gz"
        )

    mask = nib.load(mask_path)
    events = pd.read_csv(
        "/scratch/06873/zbretton/repclear_dataset/BIDS/task-study_events.tsv", sep="\t"
    )
    scene_index = [i for i, n in enumerate(events["trial_type"]) if n == "stim"]

    for trial, index in enumerate(scene_index, start=1):
        onset = events.loc[index, "onset"] + 5
        removal_timecourse = img.slicer[:, :, :, onset : onset + 11]
        out_folder = os.path.join(
            container_path, sub, "item_representations", brain_flag
        )
        mkdir(out_folder)

        # Save removal timecourse data
        output_name = os.path.join(
            out_folder,
            f"Sub-0{subID}_study_removal_timecourse_scene_trial{trial}_result.nii.gz",
        )
        nib.save(removal_timecourse, output_name)
        print(f"Saved: {output_name}")


# Run the function in parallel for each subject
Parallel(n_jobs=2)(delayed(item_representation_study)(i) for i in subs)
