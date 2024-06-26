import nibabel as nib

nib.openers.Opener.default_compresslevel = 6
from nilearn.image import mean_img, get_data, threshold_img, new_img_like, clean_img
from nilearn.glm.first_level import FirstLevelModel

# from nilearn import plotting
from nilearn.glm.second_level import SecondLevelModel
import os
import fnmatch
import numpy as np
import pandas as pd

subs = ["02", "03", "04"]
brain_flag = "MNI"

# code for the item level weighting for faces and scenes


def mkdir(path, local=False):
    if not os.path.exists(path):
        os.makedirs(path)


def pad_contrast(contrast_, n_columns):
    """A small routine to append zeros in contrast vectors"""
    return np.hstack((contrast_, np.zeros(n_columns - len(contrast_))))


def confound_cleaner(confounds):
    COI = [
        "a_comp_cor_00",
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

    # find the proper nii.gz files


def find(pattern, path):  # find the pattern we're looking for
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
        return result


# level 1 GLM
for num in range(len(subs)):
    sub_num = subs[num]

    print("Running sub-0%s..." % sub_num)
    # define the subject
    sub = "sub-0%s" % sub_num
    container_path = (
        "/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep"
    )

    bold_path = os.path.join(container_path, sub, "func/")
    os.chdir(bold_path)

    # set up the path to the files and then moved into that directory

    localizer_files = find("*preremoval*bold*.nii.gz", bold_path)
    wholebrain_mask_path = find("*preremoval*mask*.nii.gz", bold_path)

    if brain_flag == "MNI":
        pattern = "*MNI*"
        pattern2 = "*MNI152NLin2009cAsym*preproc*"
        brain_mask_path = fnmatch.filter(wholebrain_mask_path, pattern)
        localizer_files = fnmatch.filter(localizer_files, pattern2)
    elif brain_flag == "T1w":
        pattern = "*T1w*"
        pattern2 = "*T1w*preproc*"
        brain_mask_path = fnmatch.filter(wholebrain_mask_path, pattern)
        localizer_files = fnmatch.filter(localizer_files, pattern2)

    brain_mask_path.sort()
    localizer_files.sort()
    face_mask_path = os.path.join(
        "/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/group_model/group_category_lvl2/",
        "group_face_%s_mask.nii.gz" % brain_flag,
    )
    scene_mask_path = os.path.join(
        "/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/group_model/group_category_lvl2/",
        "group_scene_%s_mask.nii.gz" % brain_flag,
    )
    face_mask = nib.load(face_mask_path)
    scene_mask = nib.load(scene_mask_path)

    # load in category mask that was created from the first GLM

    img = nib.concat_images(localizer_files, axis=3)
    # to be used to filter the data
    # First we are removing the confounds
    # get all the folders within the bold path
    # confound_folders=[x[0] for x in os.walk(bold_path)]
    localizer_confounds_1 = find("*preremoval*1*confounds*.tsv", bold_path)
    localizer_confounds_2 = find("*preremoval*2*confounds*.tsv", bold_path)
    localizer_confounds_3 = find("*preremoval*3*confounds*.tsv", bold_path)
    localizer_confounds_4 = find("*preremoval*4*confounds*.tsv", bold_path)
    localizer_confounds_5 = find("*preremoval*5*confounds*.tsv", bold_path)
    localizer_confounds_6 = find("*preremoval*6*confounds*.tsv", bold_path)

    confound_run1 = pd.read_csv(localizer_confounds_1[0], sep="\t")
    confound_run2 = pd.read_csv(localizer_confounds_2[0], sep="\t")
    confound_run3 = pd.read_csv(localizer_confounds_3[0], sep="\t")
    confound_run4 = pd.read_csv(localizer_confounds_4[0], sep="\t")
    confound_run5 = pd.read_csv(localizer_confounds_5[0], sep="\t")
    confound_run6 = pd.read_csv(localizer_confounds_6[0], sep="\t")

    confound_run1 = confound_cleaner(confound_run1)
    confound_run2 = confound_cleaner(confound_run2)
    confound_run3 = confound_cleaner(confound_run3)
    confound_run4 = confound_cleaner(confound_run4)
    confound_run5 = confound_cleaner(confound_run5)
    confound_run6 = confound_cleaner(confound_run6)

    localizer_confounds = pd.concat(
        [
            confound_run1,
            confound_run2,
            confound_run3,
            confound_run4,
            confound_run5,
            confound_run6,
        ],
        ignore_index=False,
    )

    # get run list so I can clean the data across each of the runs
    run1_length = int((img.get_fdata().shape[3]) / 6)
    run2_length = int((img.get_fdata().shape[3]) / 6)
    run3_length = int((img.get_fdata().shape[3]) / 6)
    run4_length = int((img.get_fdata().shape[3]) / 6)
    run5_length = int((img.get_fdata().shape[3]) / 6)
    run6_length = int((img.get_fdata().shape[3]) / 6)

    run1 = np.full(run1_length, 1)
    run2 = np.full(run2_length, 2)
    run3 = np.full(run3_length, 3)
    run4 = np.full(run4_length, 4)
    run5 = np.full(run5_length, 5)
    run6 = np.full(run6_length, 6)

    run_list = np.concatenate((run1, run2, run3, run4, run5, run6))
    # clean data ahead of the GLM
    img_clean = clean_img(
        img, sessions=run_list, t_r=1, detrend=False, standardize="zscore"
    )
    """load in the denoised bold data and events file"""
    events = pd.read_csv(
        "/scratch1/06873/zbretton/repclear_dataset/BIDS/task-preremoval_events.tsv",
        sep="\t",
    )
    # now will need to create a loop where I iterate over the face & scene indexes
    # I then relabel that trial of the face or scene as "face_trial#" or "scene_trial#" and then label rest and all other trials as "other"
    # I can either do this in one loop, or two consecutive

    """initialize the face GLM with face mask"""
    face_model = FirstLevelModel(
        t_r=1,
        hrf_model="glover",
        drift_model=None,
        high_pass=None,
        mask_img=face_mask,
        signal_scaling=False,
        smoothing_fwhm=8,
        noise_model="ar1",
        n_jobs=1,
        verbose=2,
        memory="/scratch1/06873/zbretton/nilearn_cache",
        memory_level=1,
    )
    """initialize the scene GLM with scene mask"""
    scene_model = FirstLevelModel(
        t_r=1,
        hrf_model="glover",
        drift_model=None,
        high_pass=None,
        mask_img=scene_mask,
        signal_scaling=False,
        smoothing_fwhm=8,
        noise_model="ar1",
        n_jobs=1,
        verbose=2,
        memory="/scratch1/06873/zbretton/nilearn_cache",
        memory_level=1,
    )

    # I want to ensure that "trial" is the # of face (e.g., first instance of "face" is trial=1, second is trial=2...)
    face_trials = events.trial_type.value_counts().face
    scene_trials = events.trial_type.value_counts().scene
    # so this will give us a sense of total trials for these two conditions
    # next step is to then get the index of these conditions, and then use the trial# to iterate through the indexes properly

    temp_events = (
        events.copy()
    )  # copy the original events list, so that we can convert the "faces" and "scenes" to include the trial # (which corresponds to a unique image)
    face_index = [
        i for i, n in enumerate(temp_events["trial_type"]) if n == "face"
    ]  # this will find the nth occurance of a desired value in the list
    scene_index = [
        i for i, n in enumerate(temp_events["trial_type"]) if n == "scene"
    ]  # this will find the nth occurance of a desired value in the list
    for trial in range(len(face_index)):
        # this is a rough idea how I will create a temporary new version of the events file to use for the LSS
        temp_events.loc[face_index[trial], "trial_type"] = "face_trial%s" % (trial + 1)
    for trial in range(len(scene_index)):
        temp_events.loc[scene_index[trial], "trial_type"] = "scene_trial%s" % (
            trial + 1
        )

    for trial in range(len(face_index)):
        face_model.fit(
            run_imgs=img_clean, events=temp_events, confounds=localizer_confounds
        )

        """grab the number of regressors in the model"""
        n_columns = face_model.design_matrices_[0].shape[-1]

        # since the columns are not sorted as expected, I will need to located the index of the current trial to place the contrast properly

        """define the contrasts - the order of trial types is stored in model.design_matrices_[0].columns
           pad_contrast() adds 0s to the end of a vector in the case that other regressors are modeled, but not included in the primary contrasts"""
        # order is: trial1...trialN
        # bascially create an array the length of all the items
        item_contrast = np.full(n_columns, 0)  # start with an array of 0's
        item_contrast[
            face_model.design_matrices_[0].columns.str.match("face_trial")
        ] = -1  # find all the indices of face_trial and set to -1
        item_contrast[
            face_model.design_matrices_[0].columns.str.match("scene_trial")
        ] = -1  # find all the indices of scene_trial and set to -1
        item_contrast[
            face_model.design_matrices_[0].columns.get_loc("face_trial%s" % (trial + 1))
        ] = (
            face_trials + scene_trials
        ) - 1  # now find our trial of interest and set it equal to the sum of the rest of the contrasts

        contrasts = {
            "face_trial%s" % (trial + 1): pad_contrast(item_contrast, n_columns)
        }

        """point to and if necessary create the output folder"""
        out_folder = os.path.join(container_path, sub, "localizer_item_level")
        if not os.path.exists(out_folder):
            os.makedirs(out_folder, exist_ok=True)

        # as of now it is labeling the trial estimates by the trial number, which is helpful since I can look at their individual design matricies to see which stim that is
        # but another way could be to load in the list for that sub right here, grab the number or name of that stim from the trial index and use that to save the name

        """compute and save the contrasts"""
        for contrast in contrasts:
            z_map = face_model.compute_contrast(
                contrasts[contrast], output_type="z_score"
            )
            nib.save(
                z_map, os.path.join(out_folder, f"{contrast}_{brain_flag}_zmap.nii.gz")
            )
            t_map = face_model.compute_contrast(
                contrasts[contrast], stat_type="t", output_type="stat"
            )
            nib.save(
                t_map, os.path.join(out_folder, f"{contrast}_{brain_flag}_tmap.nii.gz")
            )
            file_data = face_model.generate_report(contrasts[contrast])
            file_data.save_as_html(
                os.path.join(out_folder, f"{contrast}_{brain_flag}_report.html")
            )

        del item_contrast

    for trial in range(scene_trials):
        scene_model.fit(
            run_imgs=img_clean, events=temp_events, confounds=localizer_confounds
        )

        """grab the number of regressors in the model"""
        n_columns = scene_model.design_matrices_[0].shape[-1]

        """define the contrasts - the order of trial types is stored in model.design_matrices_[0].columns
           pad_contrast() adds 0s to the end of a vector in the case that other regressors are modeled, but not included in the primary contrasts"""
        # order is: trial1...trialN

        # bascially create an array the length of all the items
        item_contrast = np.full(n_columns, 0)  # start with an array of 0's
        item_contrast[
            scene_model.design_matrices_[0].columns.str.match("face_trial")
        ] = -1  # find all the indices of face_trial and set to -1
        item_contrast[
            scene_model.design_matrices_[0].columns.str.match("scene_trial")
        ] = -1  # find all the indices of scene_trial and set to -1
        item_contrast[
            scene_model.design_matrices_[0].columns.get_loc(
                "scene_trial%s" % (trial + 1)
            )
        ] = (
            face_trials + scene_trials
        ) - 1  # now find our trial of interest and set it equal to the sum of the rest of the contrasts

        contrasts = {
            "scene_trial%s" % (trial + 1): pad_contrast(item_contrast, n_columns)
        }

        """point to and if necessary create the output folder"""
        out_folder = os.path.join(container_path, sub, "localizer_item_level")
        if not os.path.exists(out_folder):
            os.makedirs(out_folder, exist_ok=True)

        # as of now it is labeling the trial estimates by the trial number, which is helpful since I can look at their individual design matricies to see which stim that is
        # but another way could be to load in the list for that sub right here, grab the number or name of that stim from the trial index and use that to save the name

        """compute and save the contrasts"""
        for contrast in contrasts:
            z_map = scene_model.compute_contrast(
                contrasts[contrast], output_type="z_score"
            )
            nib.save(
                z_map, os.path.join(out_folder, f"{contrast}_{brain_flag}_zmap.nii.gz")
            )
            t_map = scene_model.compute_contrast(
                contrasts[contrast], stat_type="t", output_type="stat"
            )
            nib.save(
                t_map, os.path.join(out_folder, f"{contrast}_{brain_flag}_tmap.nii.gz")
            )
            file_data = scene_model.generate_report(contrasts[contrast])
            file_data.save_as_html(
                os.path.join(out_folder, f"{contrast}_{brain_flag}_report.html")
            )
        # make sure to clear the item constrast to make sure we dont carry it over in to the next trial
        del item_contrast
