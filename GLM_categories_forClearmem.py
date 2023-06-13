import nibabel as nib

nib.openers.Opener.default_compresslevel = 6
from nilearn.image import (
    mean_img,
    get_data,
    threshold_img,
    new_img_like,
    clean_img,
    concat_imgs,
)
from nilearn.glm.first_level import FirstLevelModel

# from nilearn import plotting
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img
import os
import fnmatch
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


subs = ["61", "69", "77"]
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
def category_RSA(subID):
    print("Running sub-0%s..." % subID)
    # define the subject
    sub = "sub-0%s" % subID
    container_path = "/scratch1/06873/zbretton/clearmem/"

    bold_path = os.path.join(container_path, sub, "func/")
    os.chdir(bold_path)

    # set up the path to the files and then moved into that directory

    localizer_files = find("*localizer*bold*.nii.gz", bold_path)
    wholebrain_mask_path = find("*localizer*mask*.nii.gz", bold_path)

    if brain_flag == "MNI":
        pattern = "*MNI*"
        pattern2 = "*MNI152NLin2009cAsym*preproc_resized*"
        brain_mask_path = fnmatch.filter(wholebrain_mask_path, pattern)
        localizer_files = fnmatch.filter(localizer_files, pattern2)

    brain_mask_path.sort()
    localizer_files.sort()
    vtc_mask_path = os.path.join(
        "/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/group_%s_VTC_mask.nii.gz"
        % brain_flag
    )

    vtc_mask = nib.load(vtc_mask_path)

    img = concat_imgs(localizer_files, memory="/scratch1/06873/zbretton/nilearn_cache")

    # to be used to filter the data
    # First we are removing the confounds
    # get all the folders within the bold path
    # confound_folders=[x[0] for x in os.walk(bold_path)]
    localizer_confounds_1 = find("*localizer*1*confounds*.tsv", bold_path)
    localizer_confounds_2 = find("*localizer*2*confounds*.tsv", bold_path)
    localizer_confounds_3 = find("*localizer*3*confounds*.tsv", bold_path)
    localizer_confounds_4 = find("*localizer*4*confounds*.tsv", bold_path)
    localizer_confounds_5 = find("*localizer*5*confounds*.tsv", bold_path)

    confound_run1 = pd.read_csv(localizer_confounds_1[0], sep="\t")
    confound_run2 = pd.read_csv(localizer_confounds_2[0], sep="\t")
    confound_run3 = pd.read_csv(localizer_confounds_3[0], sep="\t")
    confound_run4 = pd.read_csv(localizer_confounds_4[0], sep="\t")
    confound_run5 = pd.read_csv(localizer_confounds_5[0], sep="\t")

    confound_run1 = confound_cleaner(confound_run1)
    confound_run2 = confound_cleaner(confound_run2)
    confound_run3 = confound_cleaner(confound_run3)
    confound_run4 = confound_cleaner(confound_run4)
    confound_run5 = confound_cleaner(confound_run5)

    localizer_confounds = pd.concat(
        [confound_run1, confound_run2, confound_run3, confound_run4, confound_run5],
        ignore_index=False,
    )

    # get run list so I can clean the data across each of the runs
    run1_length = int((img.get_fdata().shape[3]) / 5)
    run2_length = int((img.get_fdata().shape[3]) / 5)
    run3_length = int((img.get_fdata().shape[3]) / 5)
    run4_length = int((img.get_fdata().shape[3]) / 5)
    run5_length = int((img.get_fdata().shape[3]) / 5)

    run1 = np.full(run1_length, 1)
    run2 = np.full(run2_length, 2)
    run3 = np.full(run3_length, 3)
    run4 = np.full(run4_length, 4)
    run5 = np.full(run5_length, 5)

    run_list = np.concatenate((run1, run2, run3, run4, run5))
    # clean data ahead of the GLM
    img_clean = clean_img(
        img,
        sessions=run_list,
        t_r=1,
        detrend=False,
        standardize="zscore",
        mask_img=vtc_mask,
    )
    del img
    """load in the denoised bold data and events file"""
    events = pd.read_csv(
        "/scratch1/06873/zbretton/clearmem/localizer_events.csv", sep=","
    )

    # this has too much info so we need to only take the important columns
    events = events[["onset", "duration", "trial_type"]]

    """initialize and fit the GLM"""
    model = FirstLevelModel(
        t_r=1,
        hrf_model="glover",
        drift_model=None,
        high_pass=None,
        mask_img=vtc_mask,
        signal_scaling=False,
        smoothing_fwhm=8,
        noise_model="ar1",
        n_jobs=1,
        verbose=2,
        memory="/scratch1/06873/zbretton/nilearn_cache",
        memory_level=1,
    )

    model.fit(run_imgs=img_clean, events=events, confounds=localizer_confounds)

    """grab the number of regressors in the model"""
    n_columns = model.design_matrices_[0].shape[-1]

    """define the contrasts - the order of trial types is stored in model.design_matrices_[0].columns
       pad_contrast() adds 0s to the end of a vector in the case that other regressors are modeled, but not included in the primary contrasts"""
    # order is: face, fruit, rest, scene
    contrasts = {
        "faces": pad_contrast([2, -1, 0, -1], n_columns),
        "scenes": pad_contrast([-1, -1, 0, 2], n_columns),
        "fruit": pad_contrast([-1, 2, 0, -1], n_columns),
        "stimuli": pad_contrast([1, 1, -3, 1], n_columns),
    }

    """point to and if necessary create the output folder"""
    out_folder = os.path.join(container_path, sub, "localizer_lvl1")
    if not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)

    """compute and save the contrasts"""
    for contrast in contrasts:
        z_map = model.compute_contrast(contrasts[contrast], output_type="z_score")
        nib.save(
            z_map, os.path.join(out_folder, f"{contrast}_{brain_flag}_zmap.nii.gz")
        )
        t_map = model.compute_contrast(
            contrasts[contrast], stat_type="t", output_type="stat"
        )
        nib.save(
            t_map, os.path.join(out_folder, f"{contrast}_{brain_flag}_tmap.nii.gz")
        )
        file_data = model.generate_report(contrasts[contrast])
        file_data.save_as_html(
            os.path.join(out_folder, f"{contrast}_{brain_flag}_report.html")
        )
        del z_map, t_map, file_data


Parallel(n_jobs=len(subs))(delayed(category_RSA)(i) for i in subs)

####################################
# level 2 GLM
contrasts = ["faces", "scenes", "fruit", "stimuli"]


def item_RSA_level2(contrast):
    subs = ["sub-061", "sub-069", "sub-077"]
    container_path = "/scratch1/06873/zbretton/clearmem/"
    """point to the save directory"""
    out_dir = os.path.join(container_path, "group_model", "group_category_lvl2")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    """load in the subject maps"""
    maps = [
        nib.load(
            os.path.join(
                container_path,
                sub,
                "localizer_lvl1",
                f"{contrast}_{brain_flag}_tmap.nii.gz",
            )
        )
        for sub in subs
    ]

    """a simple group mean design"""
    design_matrix = pd.DataFrame([1] * len(maps), columns=["intercept"])

    """initialize and fit the GLM"""
    second_level_model = SecondLevelModel(
        smoothing_fwhm=None,
        mask_img="/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/group_MNI_VTC_mask.nii.gz",
        verbose=2,
        n_jobs=-1,
    )
    second_level_model = second_level_model.fit(maps, design_matrix=design_matrix)
    t_map = second_level_model.compute_contrast(
        second_level_stat_type="t", output_type="stat"
    )

    """save the group map"""
    nib.save(t_map, os.path.join(out_dir, f"group_{contrast}_{brain_flag}_tmap.nii.gz"))
    # now I want to treshold this to focus on the important clusters:
    thresholded_map, _ = threshold_stats_img(
        t_map, alpha=0.05, height_control=None, cluster_threshold=0
    )
    file_data = second_level_model.generate_report(
        contrasts="intercept", alpha=0.05, height_control=None, cluster_threshold=0
    )
    file_data.save_as_html(
        os.path.join(out_dir, f"group+{contrast}_{brain_flag}_report.html")
    )
    # use this threshold to look at the second-level results
    nib.save(
        thresholded_map,
        os.path.join(out_dir, f"group+{contrast}_{brain_flag}_thresholded_tmap.nii.gz"),
    )
    del thresholded_map, t_map, second_level_model, maps


Parallel(n_jobs=len(contrasts))(delayed(item_RSA_level2)(i) for i in contrasts)
