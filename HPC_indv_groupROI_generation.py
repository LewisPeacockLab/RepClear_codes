import os
import fnmatch
import glob
import numpy as np
import nibabel as nib
from nilearn.image import math_img

non_zero_voxels_list = []

def mask_info(mask_img):
    """Print relevant statistics of a mask."""
    mask_data = mask_img.get_fdata()
    print(f"Mask Shape: {mask_data.shape}")
    print(f"Unique values: {np.unique(mask_data)}")
    print(f"Non-zero voxels: {np.count_nonzero(mask_data)}")

# Function to create an individual hippocampus mask
def create_individual_hippocampus_mask(aparc_aseg_file_path, output_dir):
    print(f"Creating individual hippocampus mask for {aparc_aseg_file_path}...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the NIfTI file right before the masking operation
    aparc_aseg_file = nib.load(aparc_aseg_file_path)
    print("Creating hippocampus mask...")
    # Extract the left hippocampus (label 17)
    left_hippocampus_mask = math_img("img == 17", img=aparc_aseg_file)

    # Extract the right hippocampus (label 53)
    right_hippocampus_mask = math_img("img == 53", img=aparc_aseg_file)
    
    # Combine the left and right hippocampus masks
    hippocampus_mask = math_img("img1 + img2", img1=left_hippocampus_mask, img2=right_hippocampus_mask)

    print("Individual Hippocampus Mask Info:")
    mask_info(hippocampus_mask)

    mask_data = hippocampus_mask.get_fdata()
    non_zero_voxels = np.count_nonzero(mask_data)
    
    # Append to list
    non_zero_voxels_list.append(non_zero_voxels)

    # Save the individual hippocampus mask
    hippocampus_mask_file = os.path.join(output_dir, 'hippocampus_mask.nii.gz')
    nib.save(hippocampus_mask, hippocampus_mask_file)


# Function to create a group-level hippocampus mask
def create_group_level_hippocampus_mask(base_dir, output_dir, threshold=0.5):
    subject_dirs = glob.glob(os.path.join(base_dir, "sub-*"))
    group_masks = []

    for subj_dir in subject_dirs:
        mask_dir = os.path.join(subj_dir, "new_mask")
        mask_file = os.path.join(mask_dir, 'hippocampus_mask.nii.gz')

        if os.path.exists(mask_file):
            mask_img = nib.load(mask_file)
            mask_data = mask_img.get_fdata()
            group_masks.append(mask_data)

    # Create and save group-level masks
    if group_masks:
        group_mask_data = np.mean(np.array(group_masks), axis=0)
        group_mask_data = (group_mask_data >= threshold).astype(int)
        group_mask_img = nib.Nifti1Image(group_mask_data, mask_img.affine)

        # Print group mask info
        print("Group-level Hippocampus Mask Info:")
        mask_info(group_mask_img)

        group_mask_file = os.path.join(output_dir, 'group_hippocampus_mask.nii.gz')
        nib.save(group_mask_img, group_mask_file)

    # Calculate and print size of the final group mask
    group_mask_data_non_zero_voxels = np.count_nonzero(group_mask_data)
    print(f"Size of the final group mask (non-zero voxels): {group_mask_data_non_zero_voxels}")


# Main function to iterate over subjects and create masks
def main():
    base_dir = "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/"
    subs = [
    "002",
    "003",
    "004",
    "005",
    "006",
    "007",
    "008",
    "009",
    "010",
    "011",
    "012",
    "013",
    "014",
    "015",
    "016",
    "017",
    "018",
    "020",
    "023",
    "024",
    "025",
    "026",
    ]
    task_flags = ["preremoval", "study", "postremoval"]

    for sub_num in subs:
        sub = f"sub-{sub_num}"
        print(f"Processing subject: {sub}...")        
        for task_flag in task_flags:
            print(f"Task flag: {task_flag}...")            
            bold_path = os.path.join(base_dir, sub, "func/")
            os.chdir(bold_path)

            pattern = f"*{task_flag}*MNI152NLin2009cAsym*aparcaseg*"
            functional_files = fnmatch.filter(os.listdir(), pattern)

            if functional_files:
                # Sort the list of functional files to ensure consistency
                functional_files.sort()
                
                # Take the first functional file (which should correspond to the first run)
                aparc_aseg_file_path = functional_files[0]
                
                print(f"Found functional file: {aparc_aseg_file_path}...")
                output_dir = os.path.join(base_dir, sub, "new_mask")
                create_individual_hippocampus_mask(aparc_aseg_file_path, output_dir)

    create_group_level_hippocampus_mask(base_dir, base_dir)

if __name__ == "__main__":
    main()
    # Calculate mean and standard deviation of non-zero voxels
    mean_non_zero_voxels = np.mean(non_zero_voxels_list)
    std_non_zero_voxels = np.std(non_zero_voxels_list)
    
    print(f"Mean number of non-zero voxels across subjects: {mean_non_zero_voxels}")
    print(f"Standard deviation of non-zero voxels across subjects: {std_non_zero_voxels}")