#!/bin/bash

# Main directory containing subject directories
main_dir="/Volumes/zbg_eHD/Zachary_Data/processed"

# Group mask will be saved here
group_mask="/Volumes/zbg_eHD/Zachary_Data/processed/group_mask/group_mask.nii.gz"

# Temporary mask for the loop
temp_mask="/Volumes/zbg_eHD/Zachary_Data/processed/group_mask/temp_mask.nii.gz"

# Task of interest
task="preremoval"

# Threshold for creating binary mask
threshold=1000

# Store the start time
start_time=$(date +%s)

# Remove group mask and temporary mask if they exist
if [ -f $group_mask ]; then
    rm $group_mask
fi
if [ -f $temp_mask ]; then
    rm $temp_mask
fi

# Array to store the volumes of the individual masks
volumes=()

# Iterate over subject directories
for item in $main_dir/sub-*; do
    if [ -d "$item" ]; then
        # Extract the subject id from the directory name
        sub_id=$(basename $item)

        echo "Processing: $sub_id"

        # Create masks directory if it doesn't exist
        mkdir -p $item/masks

        # Initialize a mask for the current subject
        subject_mask=$item/masks/${sub_id}_task-${task}_space-MNI152NLin2009cAsym_mask.nii.gz
        if [ -f $subject_mask ]; then
            rm $subject_mask
        fi

        # Check if there are any runs for the current subject and task
        runs=$(find $item/func -name "${sub_id}_task-${task}_*_space-MNI152NLin2009cAsym_desc-aparcaseg_dseg.nii.gz")
        if [ -z "$runs" ]; then
            echo "Warning: No runs found for subject $sub_id and task $task"
            continue
        fi

        # Iterate over all runs for the current subject and task
        for prob_file in $runs; do
            echo "Processing: $prob_file"

            # Create a mask for the current run
            run_mask=$item/masks/$(basename ${prob_file%.*})_mask.nii.gz
            fslmaths $prob_file -thr $threshold -bin $run_mask

            # Add the current run's mask to the subject mask
            if [ ! -f $subject_mask ]; then
                # Copy the first run's mask to the subject mask
                cp $run_mask $subject_mask
            else
                # Add the current run's mask to the subject mask and save to the temporary mask
                fslmaths $subject_mask -add $run_mask -bin $temp_mask

                # Move the temporary mask to the subject mask
                mv $temp_mask $subject_mask
            fi
        done

        # Store the volume of the subject mask
        volumes+=($(fslstats $subject_mask -V | awk '{print $1}'))

        # Add the subject mask to the group mask
        if [ ! -f $group_mask ]; then
            # Copy the first subject mask to the group mask
            cp $subject_mask $group_mask
        else
            # Add the subject mask to the group mask and save to the temporary mask
            fslmaths $group_mask -add $subject_mask -bin $temp_mask

            # Move the temporary mask to the group mask
            mv $temp_mask $group_mask
        fi
    fi
done

# Calculate the mean and standard deviation of the volumes
total=0
for volume in "${volumes[@]}"; do
  total=$(echo "$total + $volume" | bc)
done
mean=$(echo "scale=2; $total/${#volumes[@]}" | bc)

sumsq=0
for volume in "${volumes[@]}"; do
  sumsq=$(echo "$sumsq + ($volume - $mean) * ($volume - $mean)" | bc)
done
stddev=$(echo "scale=2; sqrt($sumsq/${#volumes[@]})" | bc)

# Store the end time
end_time=$(date +%s)

# Calculate the duration
duration=$(($end_time-$start_time))

# Print the results
echo "Script completed in $duration seconds."
echo "The group mask contains $(fslstats $group_mask -V | awk '{print $1}') voxels."
echo "The mean volume of the individual masks is $mean voxels, with a standard deviation of $stddev voxels."