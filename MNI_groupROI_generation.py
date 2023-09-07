import os
import glob
import nibabel as nib
import numpy as np


def create_group_level_mask(base_dir, task_phases, output_dir, threshold=0.5):
    """
    Create a group-level MNI mask based on individual subject ROI masks in a BIDS structure.

    Parameters:
    - base_dir: Directory holding all the subject folders
    - task_phases: List of task phases
    - output_dir: Directory to save the group-level mask
    - threshold: Fraction of subjects that must have a voxel in the ROI for it to be included in the group mask

    Returns:
    - Saves the group-level masks for each ROI in the output_dir
    """

    # ROIs to focus on
    rois = ["Prefrontal_ROI", "Higher_Order_Visual_ROI"]

    # Dictionary to hold subject-level masks for each ROI
    subject_level_masks = {roi: [] for roi in rois}

    # Get all subject directories and filter out non-directories (like .html files)
    subject_dirs = [
        d for d in glob.glob(os.path.join(base_dir, "sub-*")) if os.path.isdir(d)
    ]

    for subj_dir in subject_dirs:
        mask_dir = os.path.join(subj_dir, "new_mask")

        # Check if the new_mask folder exists
        if not os.path.exists(mask_dir):
            print(f"Warning: new_mask folder not found for {subj_dir}. Skipping...")
            continue

        # Temporary storage for each subject's task-phase specific masks for each ROI
        temp_masks = {roi: [] for roi in rois}

        for task in task_phases:
            for roi in rois:
                pattern = os.path.join(mask_dir, f"{roi}_{task}_MNI.nii.gz")
                files = glob.glob(pattern)

                for file_path in files:
                    # Check if the file is empty
                    if os.path.getsize(file_path) == 0:
                        print(f"Warning: Empty file {file_path}. Skipping...")
                        continue

                    roi_img = nib.load(file_path)
                    roi_data = roi_img.get_fdata()
                    temp_masks[roi].append(roi_data)

        # Average across task phases for each ROI for the current subject
        for roi in rois:
            if len(temp_masks[roi]) == 0:
                print(f"Warning: No masks found for {roi} in {subj_dir}. Skipping...")
                continue

            subject_mask = np.mean(np.array(temp_masks[roi]), axis=0)
            subject_level_masks[roi].append(subject_mask)

            subject_mask_img = nib.Nifti1Image(subject_mask, roi_img.affine)
            subject_mask_path = os.path.join(mask_dir, f"{roi}_Sub_MNI.nii.gz")
            nib.save(subject_mask_img, subject_mask_path)

    # Create and save group-level masks
    for roi in rois:
        if len(subject_level_masks[roi]) == 0:
            print(f"Warning: No subject-level masks to average for {roi}. Skipping...")
            continue

        group_mask_data = np.mean(np.array(subject_level_masks[roi]), axis=0)
        group_mask_data = (group_mask_data >= threshold).astype(int)

        group_mask_img = nib.Nifti1Image(group_mask_data, roi_img.affine)
        output_file_path = os.path.join(output_dir, f"group_MNI_{roi}.nii.gz")
        nib.save(group_mask_img, output_file_path)


# usage
base_dir = "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/"
task_phases = ["preremoval", "study", "postremoval"]
output_dir = base_dir

create_group_level_mask(base_dir, task_phases, output_dir)
