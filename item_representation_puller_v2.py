# item_representation_puller.py

# Imports
import os
import sys
import fnmatch
import numpy as np
import nibabel as nib
import pandas as pd
from joblib import Parallel, delayed
from nilearn.image import clean_img

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Dictionary mapping phases to the number of runs
phase_to_runs = {"preremoval": 6, "study": 3, "postremoval": 6}

# Subjects list
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


# Function to clean confounds
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


def find(pattern, path):  # find the pattern we're looking for
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
        return result


def load_and_preprocess_bold_data(subID, brain_flag, phase, roi_name):
    sub = f"sub-0{subID}"
    container_path = (
        "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep"
    )
    bold_path = os.path.join(container_path, sub, "func/")
    # os.chdir(bold_path)

    localizer_files = find(f"*{phase}*bold*.nii.gz", bold_path)

    if brain_flag == "MNI":
        pattern2 = "*MNI152NLin2009cAsym*preproc*"
        localizer_files = fnmatch.filter(localizer_files, pattern2)
    elif brain_flag == "T1w":
        pattern2 = "*T1w*preproc*"
        localizer_files = fnmatch.filter(localizer_files, pattern2)

    localizer_files.sort()

    if brain_flag == "T1w":
        mask_path = os.path.join(
            container_path,
            sub,
            "new_mask",
            f"{roi_name}_{phase}_{brain_flag}_mask.nii.gz",
        )
    else:
        mask_path = os.path.join(container_path, f"group_MNI_{roi_name}.nii.gz")

    mask = nib.load(mask_path)

    processed_file_name = f"sub-0{subID}_{brain_flag}_{phase}_{roi_name}.nii"
    processed_file_path = os.path.join(bold_path, processed_file_name)

    if False:  # os.path.exists(processed_file_path):
        img = nib.load(processed_file_path)
        print(f"{brain_flag} Concatenated Localizer BOLD Loaded...")
    else:
        img = nib.concat_images(localizer_files, axis=3)
        nib.save(img, processed_file_path)
        print(f"{brain_flag} Concatenated Localizer BOLD...saved")

    return img, mask, bold_path


def process_and_clean_bold_data(
    subID, brain_flag, roi_name, img, mask, bold_path, phase
):
    # Dynamically determine the number of runs based on the phase
    num_runs = phase_to_runs.get(phase, 0)  # default to 0 if phase is not found
    if num_runs == 0:
        print(f"Invalid phase: {phase}")
        return

    # Load confounds
    localizer_confounds = [
        find(f"*{phase}*{i+1}*confounds*.tsv", bold_path)[0] for i in range(num_runs)
    ]
    confound_dfs = [
        pd.read_csv(confound_file, sep="\t") for confound_file in localizer_confounds
    ]
    cleaned_confound_dfs = [confound_cleaner(df) for df in confound_dfs]

    # Concatenate all the cleaned confounds
    localizer_confounds = pd.concat(cleaned_confound_dfs, ignore_index=False)

    # Calculate run lengths and create run list based on the dynamically determined number of runs
    run_lengths = [int(img.get_fdata().shape[3] / num_runs) for _ in range(num_runs)]
    run_list = np.concatenate(
        [np.full(length, i + 1) for i, length in enumerate(run_lengths)]
    )

    cleaned_file_name = f"sub-0{subID}_{brain_flag}_{phase}_{roi_name}_cleaned.nii.gz"
    cleaned_file_path = os.path.join(bold_path, cleaned_file_name)

    if False:  # os.path.exists(cleaned_file_path):
        img_clean = nib.load(cleaned_file_path)
        print(
            f"{brain_flag} Concatenated, Cleaned and {roi_name} Masked Localizer BOLD...LOADED"
        )
        del img
    else:
        print("Cleaning and Masking BOLD data...")
        img_clean = clean_img(
            img,
            sessions=run_list,
            t_r=1,
            detrend=False,
            standardize="zscore",
            mask_img=mask,
            confounds=localizer_confounds,
        )
        nib.save(img_clean, cleaned_file_path)
        print(
            f"{brain_flag} Concatenated, Cleaned and {roi_name} Masked Localizer BOLD...saved"
        )
        del img

    return img_clean


def relabel_events(events):
    temp_events = events.copy()
    face_index = [i for i, n in enumerate(temp_events["trial_type"]) if n == "face"]
    scene_index = [i for i, n in enumerate(temp_events["trial_type"]) if n == "scene"]
    for trial in range(len(face_index)):
        temp_events.loc[face_index[trial], "trial_type"] = f"face_trial{trial + 1}"
    for trial in range(len(scene_index)):
        temp_events.loc[scene_index[trial], "trial_type"] = f"scene_trial{trial + 1}"
    return temp_events, face_index, scene_index


def process_and_save_trials(
    subID,
    brain_flag,
    img_clean,
    relabeled_events,
    face_index,
    scene_index,
    phase,
    roi_name,
):
    for trial_type, trial_indexes in zip(["face", "scene"], [face_index, scene_index]):
        for trial in range(len(trial_indexes)):
            print(f"running {trial_type} trial {trial + 1}")
            onset = relabeled_events.loc[trial_indexes[trial], "onset"] + 5
            affine_mat = img_clean.affine
            dimsize = img_clean.header.get_zooms()

            container_path = (
                "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep"
            )

            out_folder = os.path.join(
                container_path,
                f"sub-0{subID}",
                f"item_representations_{roi_name}_{brain_flag}",
            )
            if not os.path.exists(out_folder):
                os.makedirs(out_folder, exist_ok=True)

            trial_pattern = np.mean(
                img_clean.get_fdata()[:, :, :, onset : (onset + 2)], axis=3
            )
            output_name = os.path.join(
                out_folder,
                f"Sub-{subID}_{phase}_{trial_type}_trial{trial + 1}_result.nii.gz",
            )

            trial_pattern = trial_pattern.astype("double")
            trial_pattern[np.isnan(trial_pattern)] = 0

            trial_pattern_nii = nib.Nifti1Image(trial_pattern, affine_mat)
            hdr = trial_pattern_nii.header
            hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2]))
            nib.save(trial_pattern_nii, output_name)

    print("subject finished")


def item_representation(subID, phase, roi_name):
    print(f"Processing subject: {subID}, ROI: {roi_name}, Phase: {phase}")
    # Validate phase argument
    if phase not in ["preremoval", "study", "postremoval"]:
        print(
            "Invalid phase argument. Choose from 'preremoval', 'study', 'postremoval'"
        )
        return

    brain_flags = ["MNI"]  # You can add other flags here as needed

    for brain_flag in brain_flags:
        # Step 1: Load and preprocess BOLD data
        img, mask, bold_path = load_and_preprocess_bold_data(
            subID, brain_flag, phase, roi_name
        )

        # Step 2: Process and clean BOLD data
        img_clean = process_and_clean_bold_data(
            subID, brain_flag, roi_name, img, mask, bold_path, phase
        )

        # Step 3: Load and relabel events
        events_path = f"/scratch/06873/zbretton/repclear_dataset/BIDS/task-{phase}_events.tsv"  # Adjust this path
        events = pd.read_csv(events_path, sep="\t")
        relabeled_events, face_index, scene_index = relabel_events(events)

        # Step 4: Process and save trials
        process_and_save_trials(
            subID,
            brain_flag,
            img_clean,
            relabeled_events,
            face_index,
            scene_index,
            phase,
            roi_name,
        )


# Run the function for each subject and each ROI
# rois = ["Prefrontal_ROI", "Higher_Order_Visual_ROI"]  # VTC
rois = ["hippocampus_ROI"]  # added in for additional analyses
phases = ["preremoval", "postremoval"]

for roi in rois:
    for phase in phases:
        Parallel(n_jobs=2)(delayed(item_representation)(i, phase, roi) for i in subs)


def check_missing_runs(subs, phases, rois):
    missing_combinations = []

    # Root directory where output files are saved
    root_dir = "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep"

    for subID in subs:
        for phase in phases:
            for roi in rois:
                # Construct the expected output folder path
                out_folder = os.path.join(
                    root_dir,
                    f"sub-0{subID}",
                    f"item_representations_{roi}_MNI",  # Assuming MNI as brain_flag
                )

                # If the folder itself doesn't exist, then it's a missing run
                if not os.path.exists(out_folder):
                    missing_combinations.append((subID, phase, roi))
                    continue

                # Otherwise, check the existence of specific output files
                # Adjust this file name pattern according to your actual output files
                expected_file = f"Sub-{subID}_{phase}_face_trial1_result.nii.gz"
                expected_path = os.path.join(out_folder, expected_file)

                if not os.path.exists(expected_path):
                    missing_combinations.append((subID, phase, roi))

    return missing_combinations


def build_missing_dict(missing_combinations):
    missing_dict = {}
    for sub, phase, roi in missing_combinations:
        if sub not in missing_dict:
            missing_dict[sub] = {}
        if phase not in missing_dict[sub]:
            missing_dict[sub][phase] = []
        missing_dict[sub][phase].append(roi)
    return missing_dict


# After running check_missing_runs
missing_combinations = check_missing_runs(subs, phases, rois)

# Build the missing dictionary
missing_dict = build_missing_dict(missing_combinations)

# Create a list to store combinations that failed
failed_combinations = []

# Only run the missing combinations
for sub in missing_dict.keys():
    for phase in missing_dict[sub].keys():
        for roi in missing_dict[sub][phase]:
            try:
                # Call your existing function
                item_representation(sub, phase, roi)
            except Exception as e:
                # Log the exception
                print(
                    f"Failed to process Subject: {sub}, Phase: {phase}, ROI: {roi} due to {e}"
                )

                # Store the failed combination
                failed_combinations.append((sub, phase, roi))

# Output the failed combinations at the end
if failed_combinations:
    print("The following combinations failed:")
    for sub, phase, roi in failed_combinations:
        print(f"Failed combination: Subject {sub}, Phase {phase}, ROI {roi}")
else:
    print("All combinations processed successfully.")
