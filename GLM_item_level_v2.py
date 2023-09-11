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
rois = ["Prefrontal_ROI", "Higher_Order_Visual_ROI"]
container_path = "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep"
brain_flag = "MNI"


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


def load_and_sort_files(bold_path, brain_flag):
    localizer_files = find("*preremoval*bold*.nii.gz", bold_path)
    wholebrain_mask_path = find("*preremoval*mask*.nii.gz", bold_path)

    if brain_flag == "MNI":
        pattern = "*MNI*"
        pattern2 = "*MNI152NLin2009cAsym*preproc*"
    elif brain_flag == "T1w":
        pattern = "*T1w*"
        pattern2 = "*T1w*preproc*"

    brain_mask_path = fnmatch.filter(wholebrain_mask_path, pattern)
    localizer_files = fnmatch.filter(localizer_files, pattern2)

    brain_mask_path.sort()
    localizer_files.sort()

    return localizer_files, brain_mask_path


def prepare_images(localizer_files):
    return [nib.load(file) for file in localizer_files]


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


def setup_paths(subID, brain_flag):
    sub = f"sub-0{subID}"
    container_path = (
        "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep"
    )
    return sub, container_path


def is_file_newer_than(file, delta):
    cutoff = datetime.utcnow() - delta
    mtime = datetime.utcfromtimestamp(os.path.getmtime(file))
    if mtime < cutoff:
        return False
    return True


def prepare_events():
    events_path = (
        "/scratch/06873/zbretton/repclear_dataset/BIDS/task-preremoval_events.tsv"
    )
    events = pd.read_csv(events_path, sep="\t")
    events_dict = {g: d for g, d in events.groupby("run")}

    events_list = []
    face_trials = events.trial_type.value_counts().face
    scene_trials = events.trial_type.value_counts().scene

    for i in range(1, 7):
        current_events = pd.DataFrame.from_dict(events_dict[i])

        if i > 1:
            current_events["onset"] -= current_events["onset"].iat[0]

        current_events = current_events[
            current_events["trial_type"].isin(["face", "scene"])
        ].reset_index(drop=True)

        for trial in range(face_trials):
            if i == 1 and trial < 30:
                current_events.loc[trial, "trial_type"] = f"face_trial{trial + 1}"
            elif i == 2 and trial >= 30:
                current_events.loc[trial - 30, "trial_type"] = f"face_trial{trial + 1}"

        for trial in range(scene_trials):
            if i > 2:
                offset = 30 * (i - 3)
                if (trial >= offset) & (trial < offset + 30):
                    current_events.loc[
                        trial - offset, "trial_type"
                    ] = f"scene_trial{trial + 1}"

        events_list.append(current_events)

    return events_list


def fit_and_save_contrasts(
    model,
    img,
    events_list,
    confounds,
    num_trials,
    trial_type,
    brain_flag,
    container_path,
    sub,
    run_indices,
):
    """Fit the GLM and save the contrasts."""

    # Fit the model using only the relevant runs
    model.fit(
        run_imgs=[img[i] for i in run_indices],
        events=[events_list[i] for i in run_indices],
        confounds=[confounds[i] for i in run_indices],
    )

    # Grab the number of regressors in the model
    n_columns = model.design_matrices_[0].shape[-1]

    contrasts = {}

    for trial in range(num_trials):
        # Define the contrasts for each run separately
        item_contrast = [np.full(n_columns, 0) for _ in range(len(run_indices))]

        # Fill in the contrasts based on trial type and trial number
        for run_idx, run in enumerate(run_indices):
            columns = model.design_matrices_[run].columns
            trial_column = f"{trial_type}_trial{trial + 1}"

            if trial_column in columns:
                item_contrast[run_idx][columns.get_loc(trial_column)] = n_columns - 1

        contrasts[f"{trial_type}_trial{trial + 1}"] = item_contrast

    return contrasts, model


def face_model_contrast(model_face, face_trials, img, events_list, confounds, roi, sub):
    # Create mean image
    mean_img_ = mean_img(img)
    model_face.fit(run_imgs=img[:2], events=events_list[:2], confounds=confounds[:2])
    for trial in range(face_trials):
        """grab the number of regressors in the model"""
        n_columns = model_face.design_matrices_[0].shape[-1]

        # since the columns are not sorted as expected, I will need to located the index of the current trial to place the contrast properly

        """define the contrasts - the order of trial types is stored in model.design_matrices_[0].columns
           pad_contrast() adds 0s to the end of a vector in the case that other regressors are modeled, but not included in the primary contrasts"""
        # order is: trial1...trialN
        # bascially create an array the length of all the items
        item_contrast = [
            np.full(n_columns, 0),
            np.full(n_columns, 0),
        ]  # start with an array of 0's
        item_contrast[0][
            model_face.design_matrices_[0].columns.str.match("face_trial")
        ] = -1  # find all the indices of face_trial and set to -1
        item_contrast[1][
            model_face.design_matrices_[1].columns.str.match("face_trial")
        ] = -1  # find all the indices of face_trial and set to -1

        if trial < 30:
            item_contrast[0][
                model_face.design_matrices_[0].columns.get_loc(
                    "face_trial%s" % (trial + 1)
                )
            ] = (
                face_trials - 1
            )  # now find our trial of interest and set it equal to the sum of the rest of the contrasts
        elif trial >= 30:
            item_contrast[1][
                model_face.design_matrices_[1].columns.get_loc(
                    "face_trial%s" % (trial + 1)
                )
            ] = (
                face_trials - 1
            )  # now find our trial of interest and set it equal to the sum of the rest of the contrasts

        contrasts = {"face_trial%s" % (trial + 1): item_contrast}

        """point to and if necessary create the output folder"""
        if brain_flag == "MNI":
            out_folder = os.path.join(
                container_path, sub, f"preremoval_item_level_MNI_{roi}"
            )
            if not os.path.exists(out_folder):
                os.makedirs(out_folder, exist_ok=True)
        else:
            out_folder = os.path.join(
                container_path, sub, f"preremoval_item_level_T1w_{roi}"
            )
            if not os.path.exists(out_folder):
                os.makedirs(out_folder, exist_ok=True)
        # as of now it is labeling the trial estimates by the trial number, which is helpful since I can look at their individual design matricies to see which stim that is
        # but another way could be to load in the list for that sub right here, grab the number or name of that stim from the trial index and use that to save the name

        """compute and save the contrasts"""
        for contrast in contrasts:
            z_map = model_face.compute_contrast(
                contrasts[contrast], output_type="z_score"
            )
            nib.save(
                z_map,
                os.path.join(out_folder, f"{contrast}_{brain_flag}_full_zmap.nii.gz"),
            )
            t_map = model_face.compute_contrast(
                contrasts[contrast], stat_type="t", output_type="stat"
            )
            nib.save(
                t_map,
                os.path.join(out_folder, f"{contrast}_{brain_flag}_full_tmap.nii.gz"),
            )
            file_data = model_face.generate_report(contrasts, bg_img=mean_img_)
            file_data.save_as_html(
                os.path.join(out_folder, f"{contrast}_{brain_flag}_full_report.html")
            )

        del item_contrast


def scene_model_contrast(
    scene_model, scene_trials, img, events_list, confounds, roi, sub
):
    # Create mean image
    mean_img_ = mean_img(img)
    model_scene.fit(run_imgs=img[2:], events=events_list[2:], confounds=confounds[2:])
    for trial in range(scene_trials):
        """grab the number of regressors in the model"""
        n_columns = model_scene.design_matrices_[0].shape[-1]

        """define the contrasts - the order of trial types is stored in model.design_matrices_[0].columns
           pad_contrast() adds 0s to the end of a vector in the case that other regressors are modeled, but not included in the primary contrasts"""
        # order is: trial1...trialN

        item_contrast = [
            np.full(n_columns, 0),
            np.full(n_columns, 0),
            np.full(n_columns, 0),
            np.full(n_columns, 0),
        ]  # start with an array of 0's
        item_contrast[0][
            model_scene.design_matrices_[0].columns.str.match("scene_trial")
        ] = -1  # find all the indices of scene_trial and set to -1
        item_contrast[1][
            model_scene.design_matrices_[1].columns.str.match("scene_trial")
        ] = -1  # find all the indices of scene_trial and set to -1
        item_contrast[2][
            model_scene.design_matrices_[2].columns.str.match("scene_trial")
        ] = -1  # find all the indices of scene_trial and set to -1
        item_contrast[3][
            model_scene.design_matrices_[3].columns.str.match("scene_trial")
        ] = -1  # find all the indices of scene_trial and set to -1

        if trial < 30:
            item_contrast[0][
                model_scene.design_matrices_[0].columns.get_loc(
                    "scene_trial%s" % (trial + 1)
                )
            ] = (
                scene_trials - 1
            )  # now find our trial of interest and set it equal to the sum of the rest of the contrasts
        elif (trial >= 30) & (trial < 60):
            item_contrast[1][
                model_scene.design_matrices_[1].columns.get_loc(
                    "scene_trial%s" % (trial + 1)
                )
            ] = (
                scene_trials - 1
            )  # now find our trial of interest and set it equal to the sum of the rest of the contrasts
        elif (trial >= 60) & (trial < 90):
            item_contrast[2][
                model_scene.design_matrices_[2].columns.get_loc(
                    "scene_trial%s" % (trial + 1)
                )
            ] = (
                scene_trials - 1
            )  # now find our trial of interest and set it equal to the sum of the rest of the contrasts
        elif trial >= 90:
            item_contrast[3][
                model_scene.design_matrices_[3].columns.get_loc(
                    "scene_trial%s" % (trial + 1)
                )
            ] = (
                scene_trials - 1
            )  # now find our trial of interest and set it equal to the sum of the rest of the contrasts

        contrasts = {"scene_trial%s" % (trial + 1): item_contrast}

        """point to and if necessary create the output folder"""
        if brain_flag == "MNI":
            out_folder = os.path.join(
                container_path, sub, f"preremoval_item_level_MNI_{roi}"
            )
            if not os.path.exists(out_folder):
                os.makedirs(out_folder, exist_ok=True)
        else:
            out_folder = os.path.join(
                container_path, sub, f"preremoval_item_level_T1w_{roi}"
            )
            if not os.path.exists(out_folder):
                os.makedirs(out_folder, exist_ok=True)

        # as of now it is labeling the trial estimates by the trial number, which is helpful since I can look at their individual design matricies to see which stim that is

        """compute and save the contrasts"""
        for contrast in contrasts:
            z_map = model_scene.compute_contrast(
                contrasts[contrast], output_type="z_score"
            )
            nib.save(
                z_map,
                os.path.join(out_folder, f"{contrast}_{brain_flag}_full_zmap.nii.gz"),
            )
            t_map = model_scene.compute_contrast(
                contrasts[contrast], stat_type="t", output_type="stat"
            )
            nib.save(
                t_map,
                os.path.join(out_folder, f"{contrast}_{brain_flag}_full_tmap.nii.gz"),
            )
            file_data = model_scene.generate_report(contrasts, bg_img=mean_img_)
            file_data.save_as_html(
                os.path.join(out_folder, f"{contrast}_{brain_flag}_full_report.html")
            )
        # make sure to clear the item constrast to make sure we dont carry it over in to the next trial
        del item_contrast


def compute_and_save_maps(model, contrasts, brain_flag, container_path, sub):
    """Compute and save the z and t maps."""

    # Create output folder
    out_folder = os.path.join(
        container_path, sub, f"preremoval_item_level_{brain_flag}"
    )
    os.makedirs(out_folder, exist_ok=True)

    for contrast in contrasts:
        # Compute z-map
        z_map = model.compute_contrast(contrasts[contrast], output_type="z_score")
        nib.save(
            z_map, os.path.join(out_folder, f"{contrast}_{brain_flag}_full_zmap.nii.gz")
        )

        # Compute t-map
        t_map = model.compute_contrast(
            contrasts[contrast], stat_type="t", output_type="stat"
        )
        nib.save(
            t_map, os.path.join(out_folder, f"{contrast}_{brain_flag}_full_tmap.nii.gz")
        )

        # Generate and save report
        file_data = model.generate_report(
            contrasts, bg_img=mean_img_
        )  # 'mean_img_' to be defined in your main script
        file_data.save_as_html(
            os.path.join(out_folder, f"{contrast}_{brain_flag}_full_report.html")
        )


def init_GLM(roi):
    """initialize the face GLM"""
    model_face = FirstLevelModel(
        t_r=1,
        hrf_model="glover",
        drift_model=None,
        high_pass=None,
        mask_img=roi,
        signal_scaling=False,
        smoothing_fwhm=8,
        noise_model="ar1",
        n_jobs=1,
        verbose=2,
        memory="/scratch/06873/zbretton/nilearn_cache",
        memory_level=1,
    )

    """initialize the scene GLM"""
    model_scene = FirstLevelModel(
        t_r=1,
        hrf_model="glover",
        drift_model=None,
        high_pass=None,
        mask_img=roi,
        signal_scaling=False,
        smoothing_fwhm=8,
        noise_model="ar1",
        n_jobs=1,
        verbose=2,
        memory="/scratch/06873/zbretton/nilearn_cache",
        memory_level=1,
    )
    return model_face, model_scene


def load_roi_mask(roi_name, roi_directory):
    roi_file_path = os.path.join(roi_directory, f"group_MNI_{roi_name}.nii.gz")
    if os.path.exists(roi_file_path):
        roi_img = nib.load(roi_file_path)
        return roi_img
    else:
        print(f"ROI mask for {roi_name} not found in {roi_directory}.")
        return None


def LSA_GLM(subID, roi):
    # Main Logic
    sub, container_path = setup_paths(subID, "MNI")

    # Load and sort files
    bold_path = os.path.join(container_path, sub, "func")
    localizer_files, brain_mask_path = load_and_sort_files(bold_path, "MNI")

    # Prepare images
    img = prepare_images(localizer_files)

    # Load ROI mask
    roi_directory = container_path
    roi_mask = load_roi_mask(roi, roi_directory)

    # Load confounds
    confounds = load_confounds(bold_path)

    # Prepare events
    events_list = prepare_events()

    # Initialize GLMs
    model_face, model_scene = init_GLM(roi_mask)

    face_model_contrast(model_face, 60, img, events_list, confounds, roi, sub)
    scene_model_contrast(model_scene, 120, img, events_list, confounds, roi, sub)

    # # Fit and save contrasts
    # fit_and_save_contrasts(
    #     model_face,
    #     img,
    #     events_list,
    #     confounds,
    #     60,
    #     "face",
    #     "MNI",
    #     container_path,
    #     sub,
    #     run_indices=[0, 1],
    # )
    # fit_and_save_contrasts(
    #     model_scene,
    #     img,
    #     events_list,
    #     confounds,
    #     120,
    #     "scene",
    #     "MNI",
    #     container_path,
    #     sub,
    #     run_indices=[2, 3, 4, 5],
    # )


# Parallel execution
Parallel(n_jobs=4, verbose=1)(
    delayed(LSA_GLM)(sub_num, roi) for sub_num in subs for roi in rois
)

# sequential test:
sub_num = subs[2]
for roi in rois:
    LSA_GLM(sub_num, roi)
