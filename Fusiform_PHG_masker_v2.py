# Imports
import warnings
import sys
import numpy as np
import nibabel as nib
import scipy as scipy
import scipy.io as sio
from nilearn.input_data import NiftiMasker, MultiNiftiMasker
from scipy import stats
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import os
import fnmatch
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import PredefinedSplit, cross_validate, cross_val_predict
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest
from sklearn.preprocessing import StandardScaler

# Suppress warnings if not already done
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Define subjects
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
]

# Define flags - adjust based on your requirements
brain_flags = ["T1w"]
task_flags = ["preremoval", "study", "postremoval"]


# Define the find function
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


# Define the new_mask function with adjustments for both Fusiform and Parahippocampal ROIs
def new_mask(subject_path, roi_type, task_flag, brain_flag, functional_files):
    outdir = os.path.join(subject_path, "new_mask")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    aparc_aseg = functional_files[0]

    # Define ROIs based on the type
    if roi_type == "Fusiform":
        rois = [
            ("lh_fusiform.nii.gz", 1007),
            ("rh_fusiform.nii.gz", 2007),
            ("lh_inferiortemporal.nii.gz", 1009),
            ("rh_inferiortemporal.nii.gz", 2009),
        ]
    elif roi_type == "Parahippocampal":
        rois = [
            ("lh_parahippocampal_anterior.nii.gz", 1006),
            ("rh_parahippocampal_anterior.nii.gz", 2006),
            ("lh_parahippocampal.nii.gz", 1016),
            ("rh_parahippocampal.nii.gz", 2016),
        ]

    # Generate masks for each ROI
    for roi_name, roi_code in rois:
        roi_path = os.path.join(outdir, roi_name)
        os.system(f"fslmaths {aparc_aseg} -thr {roi_code} -uthr {roi_code} {roi_path}")

    # Combine ROIs into one mask
    roi_paths = [os.path.join(outdir, roi[0]) for roi in rois]
    out_mask = os.path.join(outdir, f"{roi_type}_{task_flag}_{brain_flag}_mask.nii.gz")
    os.system(f"fslmaths {' -add '.join(roi_paths)} -bin {out_mask}")
    print(f"{roi_type} mask for {subject_path} done")


# Main loop for processing each subject
for brain_flag in brain_flags:
    for task_flag in task_flags:
        for sub_num in subs:
            sub = f"sub-0{sub_num}"
            container_path = (
                "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep"
            )
            bold_path = os.path.join(container_path, sub, "func/")
            os.chdir(bold_path)

            functional_files = find(f"*-{task_flag}_*.nii.gz", bold_path)
            wholebrain_mask_path = find(f"*-{task_flag}_*mask*.nii.gz", bold_path)

            # First, filter out any files starting with '._'
            functional_files = [
                f for f in functional_files if not os.path.basename(f).startswith("._")
            ]

            # Adjust patterns based on brain_flag
            if brain_flag == "MNI":
                pattern = "*MNI152NLin2009cAsym*"
                functional_files = fnmatch.filter(
                    functional_files, f"*{pattern}*aparcaseg*"
                )
            elif brain_flag == "T1w":
                functional_files = fnmatch.filter(functional_files, f"*T1w*aparcaseg*")

            functional_files = sorted(functional_files)

            subject_path = os.path.join(container_path, sub)

            # Call new_mask for Fusiform and Parahippocampal ROIs
            new_mask(subject_path, "Fusiform", task_flag, brain_flag, functional_files)
            new_mask(
                subject_path, "Parahippocampal", task_flag, brain_flag, functional_files
            )
