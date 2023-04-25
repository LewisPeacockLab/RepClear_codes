#!/bin/bash

# Set the path to the reference image
REF_IMAGE="/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/MVPA_cross_subjects/masks/harvardoxford_gm_mask.nii.gz"

# Iterate over each subject folder
for sub_dir in /scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/sub-*/ ; do
  sub=$(basename "$sub_dir")
  
  # Create the output directory if it doesn't exist yet
  out_dir="$sub_dir/func_resampled"
  mkdir -p "$out_dir"

  # Iterate over each run
  for run in {1..3} ; do
    # Set the path to the input file
    input_file="$sub_dir/func/${sub}_task-study_run-${run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    
    # Set the path to the output file
    output_file="$out_dir/${sub}_task-study_run-${run}_resampled.nii.gz"
    
    # Use AFNI's 3dresample to resample the input file to 2mm isotropic voxels
    3dresample -inset "$input_file" -prefix "$output_file" -rmode Li -master "$REF_IMAGE"
  done
done
