import os
import pandas as pd
import nibabel as nib
import numpy as np
from scipy.ndimage import label


def get_max_z_score(z_map_path):
    try:
        z_map = nib.load(z_map_path)
        z_data = z_map.get_fdata()
        return np.max(z_data)
    except:
        return None


def get_additional_metrics(z_map_path, threshold=3.29):
    try:
        z_map = nib.load(z_map_path)
        z_data = z_map.get_fdata()

        # Cluster size and mean z-score within cluster
        labeled_array, num_features = label(z_data > threshold)
        if num_features > 0:
            cluster_size = np.max(np.bincount(labeled_array.ravel())[1:])
            mean_z_within_cluster = np.mean(z_data[z_data > threshold])
        else:
            cluster_size = 0
            mean_z_within_cluster = 0

        return cluster_size, mean_z_within_cluster
    except:
        return None, None


# Initialize summary DataFrame
summary_columns = [
    "Subject_ID",
    "ROI",
    "Item",
    "Max_Z_Score",
    "Cluster_Size",
    "Mean_Z_Score",
]
summary_df = pd.DataFrame(columns=summary_columns)

# Define subjects and new ROIs
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
new_rois = ["Prefrontal_ROI", "Higher_Order_Visual_ROI"]
container_path = "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep"

# Iterate through subjects, ROIs, and items to populate summary
for sub in subs:
    for roi in new_rois:
        # Assuming that the z-maps are stored in a specific directory structure
        z_map_dir = os.path.join(
            container_path, f"sub-0{sub}", f"preremoval_item_level_MNI_{roi}"
        )

        if os.path.exists(z_map_dir):
            for z_map_file in os.listdir(z_map_dir):
                if "zmap" in z_map_file:
                    z_map_path = os.path.join(z_map_dir, z_map_file)

                    # Extract item name from the file name, assuming it's formatted like "item_name_MNI_full_zmap.nii.gz"
                    item_name = z_map_file.split("_")[0]

                    max_z_score = get_max_z_score(z_map_path)
                    cluster_size, mean_z_within_cluster = get_additional_metrics(
                        z_map_path
                    )

                    summary_df.loc[len(summary_df)] = [
                        sub,
                        roi,
                        item_name,
                        max_z_score,
                        cluster_size,
                        mean_z_within_cluster,
                    ]

        else:
            print(f"Directory does not exist: {z_map_dir}")

# Save the summary DataFrame
summary_df.to_csv(os.path.join(container_path, "GLM_LSA_run_summary.csv"), index=False)
