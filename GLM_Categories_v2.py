import nibabel as nib
from nilearn.image import (
    mean_img,
    load_img,
)
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img
import os
import fnmatch
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

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
# new_rois = ["Prefrontal_ROI", "Higher_Order_Visual_ROI"]
new_rois = ["VTC_mask"]  # new ROI for final analyses
container_path = "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep"


def mkdir(path, local=False):
    if not os.path.exists(path):
        os.makedirs(path)


def pad_contrast(contrast_, n_columns):
    """A small routine to append zeros in contrast vectors"""
    return np.hstack((contrast_, np.zeros(n_columns - len(contrast_))))


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


def load_confounds(bold_path):
    confounds = []
    for i in range(1, 7):  # Assuming you have 6 runs
        confound_file = find(f"*preremoval*{i}*confounds*.tsv", bold_path)
        if confound_file:
            confound_df = pd.read_csv(confound_file[0], sep="\t")
            confounds.append(confound_cleaner(confound_df))
        else:
            print(f"Confounds for run {i} not found.")
            return None
    return confounds


def load_events():
    events = pd.read_csv(
        "/scratch/06873/zbretton/repclear_dataset/BIDS/task-preremoval_events.tsv",
        sep="\t",
    )
    events_dict = {g: d for g, d in events.groupby("run")}
    events_list = [pd.DataFrame.from_dict(events_dict[i]) for i in range(1, 7)]
    return events_list


def fit_glm(img, events_list, confounds, roi_mask, sub, roi, container_path):
    # Initialize and fit the GLM for 'face'
    model_face = FirstLevelModel(
        t_r=1,
        hrf_model="glover",
        drift_model=None,
        high_pass=None,
        mask_img=roi_mask,
        signal_scaling=False,
        smoothing_fwhm=8,
        noise_model="ar1",
        n_jobs=1,
        verbose=2,
        memory="/scratch/06873/zbretton/nilearn_cache",
        memory_level=1,
    )
    model_face.fit(run_imgs=img[:2], events=events_list[:2], confounds=confounds[:2])

    # Initialize and fit the GLM for 'scene'
    model_scene = FirstLevelModel(
        t_r=1,
        hrf_model="glover",
        drift_model=None,
        high_pass=None,
        mask_img=roi_mask,
        signal_scaling=False,
        smoothing_fwhm=8,
        noise_model="ar1",
        n_jobs=1,
        verbose=2,
        memory="/scratch/06873/zbretton/nilearn_cache",
        memory_level=1,
    )
    model_scene.fit(run_imgs=img[2:], events=events_list[2:], confounds=confounds[2:])

    # Define the contrasts and compute them for both 'face' and 'scene'
    n_columns = model_face.design_matrices_[0].shape[-1]
    contrasts = {"stimuli": pad_contrast([1], n_columns)}

    # Create or point to the output folder
    out_folder = os.path.join(container_path, sub, f"preremoval_lvl1_MNI_{roi}")
    if not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)

    # Compute and save the contrasts for 'face'
    z_map_f = model_face.compute_contrast(contrasts["stimuli"], output_type="z_score")
    nib.save(z_map_f, os.path.join(out_folder, f"face_stimuli_MNI_zmap.nii.gz"))

    t_map_f = model_face.compute_contrast(
        contrasts["stimuli"], stat_type="t", output_type="stat"
    )
    nib.save(t_map_f, os.path.join(out_folder, f"face_stimuli_MNI_tmap.nii.gz"))

    # Compute and save the contrasts for 'scene'
    z_map_s = model_scene.compute_contrast(contrasts["stimuli"], output_type="z_score")
    nib.save(z_map_s, os.path.join(out_folder, f"scene_stimuli_MNI_zmap.nii.gz"))

    t_map_s = model_scene.compute_contrast(
        contrasts["stimuli"], stat_type="t", output_type="stat"
    )
    nib.save(t_map_s, os.path.join(out_folder, f"scene_stimuli_MNI_tmap.nii.gz"))


def process_subject(sub, container_path, new_rois):
    print(f"Running {sub}...")
    bold_path = os.path.join(container_path, sub, "func/")
    os.chdir(bold_path)

    # Find and sort localizer and mask files
    localizer_files = find("*preremoval*bold*.nii.gz", bold_path)
    wholebrain_mask_path = find("*preremoval*mask*.nii.gz", bold_path)
    pattern = "*MNI*"
    pattern2 = "*MNI152NLin2009cAsym*preproc*"
    brain_mask_path = fnmatch.filter(wholebrain_mask_path, pattern)
    localizer_files = fnmatch.filter(localizer_files, pattern2)
    brain_mask_path.sort()
    localizer_files.sort()

    img = [nib.load(file) for file in localizer_files]

    # Load confounds
    confounds = load_confounds(bold_path)
    if confounds is None:
        print(f"Skipping {sub} due to missing confounds.")
        return

    # Load events
    events_list = load_events()

    for roi in new_rois:
        roi_mask_path = os.path.join(container_path, f"group_MNI_{roi}.nii.gz")
        roi_mask = nib.load(roi_mask_path)

        fit_glm(img, events_list, confounds, roi_mask, sub, roi, container_path)


# Main Loop
# Parallel execution
Parallel(n_jobs=-1, verbose=1)(
    delayed(process_subject)(f"sub-0{sub_num}", container_path, new_rois)
    for sub_num in subs
)


def second_level_analysis(subs, rois, contrasts, container_path):
    # Define the output directory for second-level analysis
    out_dir = os.path.join(container_path, "group_model", "group_category_lvl2")
    mkdir(out_dir)  # Ensure the directory exists

    for roi in rois:
        for contrast in contrasts:
            # Construct the paths to load each subject's first-level results
            maps = [
                nib.load(
                    os.path.join(
                        container_path,
                        sub,
                        f"preremoval_lvl1_MNI_{roi}",
                        f"{contrast}_stimuli_MNI_zmap.nii.gz",
                    )
                )
                for sub in subs
            ]

            # Rest of the code remains largely the same
            design_matrix = pd.DataFrame([1] * len(maps), columns=["intercept"])

            # Initialize and fit the GLM
            second_level_model = SecondLevelModel(
                smoothing_fwhm=None,
                mask_img=os.path.join(
                    container_path, f"group_MNI_{roi}.nii.gz"
                ),  # Mask specific to ROI
                verbose=2,
                n_jobs=-1,
            )
            second_level_model = second_level_model.fit(
                maps, design_matrix=design_matrix
            )

            # Compute and save the T-statistic map
            t_map = second_level_model.compute_contrast(
                second_level_stat_type="t", output_type="stat"
            )
            nib.save(
                t_map, os.path.join(out_dir, f"group_{contrast}_{roi}_zmap.nii.gz")
            )

            # Threshold the map and save
            thresholded_map, _ = threshold_stats_img(
                t_map, alpha=0.05, height_control=None, cluster_threshold=10
            )
            nib.save(
                thresholded_map,
                os.path.join(
                    out_dir, f"group+{contrast}_{roi}_thresholded_zmap.nii.gz"
                ),
            )

            # Generate and save the report
            file_data = second_level_model.generate_report(
                contrasts="intercept",
                alpha=0.05,
                height_control=None,
                cluster_threshold=10,
            )
            file_data.save_as_html(
                os.path.join(out_dir, f"group+{contrast}_{roi}_report.html")
            )

            # Free up resources
            del thresholded_map, t_map, second_level_model, maps


# Ensure the subs list is formatted correctly
formatted_subs = [f"sub-0{sub_num}" for sub_num in subs]

# Define the contrasts
contrasts = ["face", "scene"]

# Call the second-level analysis function
second_level_analysis(formatted_subs, new_rois, contrasts, container_path)
