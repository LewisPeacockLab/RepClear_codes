# This code is to load in the representations and then either weight them (category vs. item) and then perform RSA
# This will also handle performing this and then comparing Pre-Localizer to Study, Pre-Localizer to Post-Localizer, and Study to Post-Localizer

# Imports
import warnings
import sys

if not sys.warnoptions:
    warnings.simplefilter("ignore")
import numpy as np
import nibabel as nib
import scipy as scipy
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import fnmatch
import pandas as pd
import pickle
import re
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import (
    PredefinedSplit,
    cross_validate,
    cross_val_predict,
    GridSearchCV,
    LeaveOneGroupOut,
)
from sklearn.feature_selection import (
    VarianceThreshold,
    f_classif,
    SelectKBest,
    SelectFpr,
)
from sklearn.preprocessing import StandardScaler
from nilearn.input_data import NiftiMasker, MultiNiftiMasker
from nilearn.image import clean_img
from nilearn.signal import clean
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
from statannot import add_stat_annotation
from statsmodels.stats.anova import AnovaRM


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
brain_flag = "MNI"
stim_labels = {0: "Rest", 1: "Scenes", 2: "Faces"}
sub_cates = {
    # "face": ["male", "female"],         #60
    "scene": ["manmade", "natural"],  # 120
}  # getting rid of faces for now so I can focus on scenes


def mkdir(path, local=False):
    if not os.path.exists(path):
        os.makedirs(path)


def apply_mask(mask=None, target=None):
    coor = np.where(mask == 1)
    values = target[coor]
    if values.ndim > 1:
        values = np.transpose(values)  # swap axes to get feature X sample
    return values


def find(pattern, path):  # find the pattern we're looking for
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
        return result


container_path = "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep"
param_dir = (
    "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/subject_designs"
)

# the subject's list of image number to trial numbers are in the "subject_designs" folder

for subID in subs:
    print("Running sub-0%s..." % subID)
    # define the subject
    sub = "sub-0%s" % subID

    """
    Input: 
    subID: 3 digit string
    phase: single digit int. 1: "pre-exposure", 2: "pre-localizer", 3: "study", 4: "post-localizer"

    Output: 
    face_order: trial numbers ordered by ["female", "male"]. 
    scene_order: trial numbers ordered by ["manmade", "natural"]

    (It's hard to sort the RDM matrix once that's been computed, so the best practice would be to sort the input to MDS before we run it)
    """
    # lets pull out the pre-localizer data here:
    tim_path = os.path.join(param_dir, f"sub-0{subID}_trial_image_match.csv")

    tim_df = pd.read_csv(tim_path)
    tim_df = tim_df[tim_df["phase"] == 2]  # phase 2 is pre-localizer
    tim_df = tim_df.sort_values(by=["category", "subcategory", "trial_id"])

    pre_scene_order = tim_df[tim_df["category"] == 1][
        ["trial_id", "image_id", "condition", "subcategory"]
    ]
    pre_face_order = tim_df[tim_df["category"] == 2][
        ["trial_id", "image_id", "condition"]
    ]

    # lets pull out the study data here:
    tim_df2 = pd.read_csv(tim_path)
    tim_df2 = tim_df2[tim_df2["phase"] == 3]  # phase 3 is study
    tim_df2 = tim_df2.sort_values(by=["category", "subcategory", "trial_id"])

    study_scene_order = tim_df2[tim_df2["category"] == 1][
        ["trial_id", "image_id", "condition", "subcategory"]
    ]

    # lets pull out the postlocalizer data here:
    tim_df3 = pd.read_csv(tim_path)
    tim_df3 = tim_df3[tim_df3["phase"] == 4]  # phase 4 is post-localizer
    tim_df3 = tim_df3.sort_values(by=["category", "subcategory", "trial_id"])

    post_scene_order = tim_df3[tim_df3["category"] == 1][
        ["trial_id", "image_id", "condition"]
    ]

    print(f"Running RSA for sub0{subID}...")

    # ===== load mask for BOLD
    if brain_flag == "MNI":
        mask_path = os.path.join(container_path, "group_MNI_VTC_mask.nii.gz")
    else:
        mask_path = os.path.join(
            "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/",
            sub,
            "new_mask",
            "VVS_preremoval_%s_mask.nii.gz" % brain_flag,
        )

    mask = nib.load(mask_path)

    print("mask shape: ", mask.shape)

    # ===== load ready BOLD for each trial of prelocalizer
    print(f"Loading preprocessed BOLDs for pre-localizer...")
    bold_dir_1 = os.path.join(
        container_path, f"sub-0{subID}", "item_representations_%s" % brain_flag
    )

    all_bolds_1 = {}  # {cateID: {trialID: bold}}
    bolds_arr_1 = []  # sample x vox
    for cateID in sub_cates.keys():
        cate_bolds_fnames_1 = glob.glob(f"{bold_dir_1}/*pre*{cateID}*")
        cate_bolds_1 = {}

        for fname in cate_bolds_fnames_1:
            trialID = fname.split("/")[-1].split("_")[-2]  # "trial1"
            trialID = int(trialID[5:])
            cate_bolds_1[trialID] = nib.load(fname).get_fdata()  # .flatten()
        cate_bolds_1 = {i: cate_bolds_1[i] for i in sorted(cate_bolds_1.keys())}
        all_bolds_1[cateID] = cate_bolds_1

        bolds_arr_1.append(
            np.stack([cate_bolds_1[i] for i in sorted(cate_bolds_1.keys())])
        )

    bolds_arr_1 = np.vstack(bolds_arr_1)
    print("bolds for prelocalizer - shape: ", bolds_arr_1.shape)

    # # ===== load ready BOLD for each trial of study
    # print(f"Loading preprocessed BOLDs for the study operation...")
    # bold_dir_2 = os.path.join(container_path, f"sub-0{subID}", "item_representations_%s" % brain_flag)

    # all_bolds_2 = {}  # {cateID: {trialID: bold}}
    # bolds_arr_2 = []  # sample x vox
    # for cateID in sub_cates.keys():
    #     cate_bolds_fnames_2 = glob.glob(f"{bold_dir_2}/*study_{cateID}*")
    #     cate_bolds_2 = {}
    #     try:
    #         for fname in cate_bolds_fnames_2:
    #             trialID = fname.split("/")[-1].split("_")[-2]  # "trial1"
    #             trialID = int(trialID[5:])
    #             cate_bolds_2[trialID] = nib.load(fname).get_fdata()  #.flatten()
    #         cate_bolds_2 = {i: cate_bolds_2[i] for i in sorted(cate_bolds_2.keys())}
    #         all_bolds_2[cateID] = cate_bolds_2

    #         bolds_arr_2.append(np.stack( [ cate_bolds_2[i] for i in sorted(cate_bolds_2.keys()) ] ))
    #     except:
    #         print('no %s trials' % cateID)
    # bolds_arr_2 = np.vstack(bolds_arr_2)
    # print("bolds for study - shape: ", bolds_arr_2.shape)

    # # ===== load ready BOLD for each removal phase of study
    # print(f"Loading preprocessed BOLDs for the removal part of study operation...")
    # bold_dir_4 = os.path.join(container_path, f"sub-0{subID}", "item_representations_%s" % brain_flag)

    # all_bolds_4 = {}  # {cateID: {trialID: bold}}
    # bolds_arr_4 = []  # sample x vox
    # for cateID in sub_cates.keys():
    #     cate_bolds_fnames_4 = glob.glob(f"{bold_dir_4}/*study_removal_timecourse_{cateID}*")
    #     cate_bolds_4 = {}
    #     try:
    #         for fname in cate_bolds_fnames_4:
    #             trialID = fname.split("/")[-1].split("_")[-2]  # "trial1"
    #             trialID = int(trialID[5:])
    #             cate_bolds_4[trialID] = nib.load(fname).get_fdata()  #.flatten()
    #         cate_bolds_4 = {i: cate_bolds_4[i] for i in sorted(cate_bolds_4.keys())}
    #         all_bolds_4[cateID] = cate_bolds_4

    #         bolds_arr_4.append(np.stack( [ cate_bolds_4[i] for i in sorted(cate_bolds_4.keys()) ] ))
    #     except:
    #         print('no %s trials' % cateID)
    # bolds_arr_4 = np.vstack(bolds_arr_4)
    # print("bolds for study - shape: ", bolds_arr_4.shape)

    # ===== load ready BOLD for each trial of postlocalizer
    print(f"Loading preprocessed BOLDs for post-localizer...")
    bold_dir_3 = os.path.join(
        container_path, f"sub-0{subID}", "item_representations_%s" % brain_flag
    )

    all_bolds_3 = {}  # {cateID: {trialID: bold}}
    bolds_arr_3 = []  # sample x vox
    for cateID in sub_cates.keys():
        cate_bolds_fnames_3 = glob.glob(f"{bold_dir_1}/*post*{cateID}*")
        cate_bolds_3 = {}

        for fname in cate_bolds_fnames_3:
            trialID = fname.split("/")[-1].split("_")[-2]  # "trial1"
            trialID = int(trialID[5:])
            cate_bolds_3[trialID] = nib.load(fname).get_fdata()  # .flatten()
        cate_bolds_3 = {i: cate_bolds_3[i] for i in sorted(cate_bolds_3.keys())}
        all_bolds_3[cateID] = cate_bolds_3

        bolds_arr_3.append(
            np.stack([cate_bolds_3[i] for i in sorted(cate_bolds_3.keys())])
        )

    bolds_arr_3 = np.vstack(bolds_arr_3)
    print("bolds for prelocalizer - shape: ", bolds_arr_3.shape)

    # when comparing pre vs. study, the pre scene have 120 trials, while study has 90 trials
    # when comparing pre vs. post, the pre scene have 120 trials, while post has 180 (120 old, 60 new)

    # apply VTC mask on prelocalizer BOLD
    masked_bolds_arr_1 = []
    for bold in bolds_arr_1:
        masked_bolds_arr_1.append(
            apply_mask(mask=mask.get_fdata(), target=bold).flatten()
        )
    masked_bolds_arr_1 = np.vstack(masked_bolds_arr_1)
    print("masked prelocalizer bold array shape: ", masked_bolds_arr_1.shape)

    # # apply mask on study BOLD
    # masked_bolds_arr_2 = []
    # for bold in bolds_arr_2:
    #     masked_bolds_arr_2.append(apply_mask(mask=mask.get_fdata(), target=bold).flatten())
    # masked_bolds_arr_2 = np.vstack(masked_bolds_arr_2)
    # print("masked study bold array shape: ", masked_bolds_arr_2.shape)

    # # apply mask on study removal BOLD
    # masked_bolds_arr_4 = []
    # for bold in bolds_arr_4:
    #     masked_bolds_arr_4.append(apply_mask(mask=mask.get_fdata(), target=bold))
    # masked_bolds_arr_4 = np.dstack(masked_bolds_arr_4)
    # print("masked study removal phase bold array shape: ", masked_bolds_arr_4.shape) #with this method, to access the trials you do: masked_bolds_arr_4[:,:,trial#]

    # apply VTC mask on postlocalizer BOLD
    masked_bolds_arr_3 = []
    for bold in bolds_arr_3:
        masked_bolds_arr_3.append(
            apply_mask(mask=mask.get_fdata(), target=bold).flatten()
        )
    masked_bolds_arr_3 = np.vstack(masked_bolds_arr_3)
    print("masked postlocalizer bold array shape: ", masked_bolds_arr_3.shape)

    # ===== load weights
    print(f"Loading weights...")
    # prelocalizer
    if brain_flag == "MNI":
        cate_weights_dir = os.path.join(
            f"/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/sub-0{subID}",
            "preremoval_lvl1_%s/scene_stimuli_MNI_zmap.nii.gz" % brain_flag,
        )
        item_weights_dir = os.path.join(
            container_path, f"sub-0{subID}", "preremoval_item_level_MNI"
        )
    else:
        cate_weights_dir = os.path.join(
            f"/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/sub-0{subID}",
            "preremoval_lvl1_%s/scene_stimuli_T1w_zmap.nii.gz" % brain_flag,
        )
        item_weights_dir = os.path.join(
            container_path, f"sub-0{subID}", "preremoval_item_level_T1w"
        )

    # prelocalizer weights (category and item) get applied to study/post representations

    all_weights = {}
    weights_arr = []

    # load in all the item specific weights, which come from the LSA contrasts per subject
    for cateID in sub_cates.keys():
        item_weights_fnames = glob.glob(f"{item_weights_dir}/{cateID}*full*zmap*")
        print(cateID, len(item_weights_fnames))
        item_weights = {}

        for fname in item_weights_fnames:
            trialID = fname.split("/")[-1].split("_")[1]  # "trial1"
            trialID = int(trialID[5:])
            item_weights[trialID] = nib.load(fname).get_fdata()
        item_weights = {i: item_weights[i] for i in sorted(item_weights.keys())}
        all_weights[cateID] = item_weights
        weights_arr.append(
            np.stack([item_weights[i] for i in sorted(item_weights.keys())])
        )

    # this now masks the item weights to ensure that they are all in the same ROI (group VTC):
    weights_arr = np.vstack(weights_arr)
    print("weights shape: ", weights_arr.shape)
    # apply mask on BOLD
    masked_weights_arr = []
    for weight in weights_arr:
        masked_weights_arr.append(
            apply_mask(mask=mask.get_fdata(), target=weight).flatten()
        )
    masked_weights_arr = np.vstack(masked_weights_arr)
    print("masked item weights arr shape: ", masked_weights_arr.shape)

    # this masks the category weight from the group results:
    cate_weights_arr = nib.load(cate_weights_dir)
    masked_cate_weights_arr = apply_mask(
        mask=mask.get_fdata(), target=cate_weights_arr.get_fdata()
    ).flatten()
    print(
        "masked category weight arr shape: ", masked_cate_weights_arr.shape
    )  # so this is just 1D of the voxels in the VTC mask

    # ===== multiply
    # prelocalizer patterns and prelocalizer item weights
    item_repress_pre = np.multiply(
        masked_bolds_arr_1, masked_weights_arr
    )  # these are lined up since the trials goes to correct trials

    # study patterns, prelocalizer item weights and postlocalizer
    # this has to come in the loop below to make sure I am weighting the correct trial with the correct item weights

    print("item representations pre shape: ", item_repress_pre.shape)

    # these work right out the box since there is only 1 "category" weighting we are using, and that can be applied to all scene trials in both pre and study (and post)
    cate_repress_pre = np.multiply(
        masked_bolds_arr_1, masked_cate_weights_arr
    )  # these are multiplied elementwise
    # cate_repress_study = np.multiply(masked_bolds_arr_2,masked_cate_weights_arr) #since there is only 1 cate_weight, this multiplies all of masked_bold_arr_2 with the cate_weights
    cate_repress_post = np.multiply(
        masked_bolds_arr_3, masked_cate_weights_arr
    )  # weight the post representations with category weights

    print("category representations pre shape: ", cate_repress_pre.shape)
    # print("category representations study shape: ", cate_repress_study.shape)
    print("category representations post shape: ", cate_repress_post.shape)

    # the way the data is currently sorted is by the index:
    # pre-localizer: 0-59 = Face trial 1 - 60 | 60-179 = Scene trial 1 - 120
    # study: 0-89 = Scene trial 1 - 90
    # post-localizer: 0-179 = Scene trial 1-180 (but 60 of these are novel)

    # the specific output of this "order" DataFrame is by subcategory within the category, but we can still use this to sort by trial since the conditions are in order

    # key arrays being used: pre_scene_order (which is the order of the images in the prelocalizer, after sorted for subcate)
    # study_scene_order (again sorted for subcate)

    item_repress_study_comp = np.zeros_like(
        item_repress_pre[:90, :]
    )  # set up this array in the same size as the pre, so I can size things in the correct trial order

    category_weight_study_comp = np.zeros_like(item_repress_study_comp)
    category_weight_pre_comp = np.zeros_like(item_repress_study_comp)
    category_weight_post_comp = np.zeros_like(item_repress_study_comp)

    # these are used to hold the fidelity changes from pre to post (scene-weighted)
    change_cw_maintain_dict = {}
    change_cw_replace_dict = {}
    change_cw_suppress_dict = {}

    counter = 0

    m_counter = 0
    s_counter = 0
    r_counter = 0

    # this loop is limited by the smaller index, so thats the study condition (only 90 stims)
    for trial in study_scene_order["trial_id"].values:
        study_trial_index = study_scene_order.index[
            study_scene_order["trial_id"] == trial
        ].tolist()[
            0
        ]  # find the order
        study_image_id = study_scene_order.loc[
            study_trial_index, "image_id"
        ]  # this now uses the index of the dataframe to find the image_id
        # study_trial_num=study_scene_order.loc[study_trial_index,'trial_id'] #now we used the image ID to find the proper trial in the post condition to link to

        pre_trial_index = pre_scene_order.index[
            pre_scene_order["image_id"] == study_image_id
        ].tolist()[
            0
        ]  # find the index in the pre for this study trial
        pre_trial_num = pre_scene_order.loc[pre_trial_index, "trial_id"]

        image_condition = pre_scene_order.loc[pre_trial_index, "condition"]

        post_trial_index = post_scene_order.index[
            post_scene_order["image_id"] == study_image_id
        ].tolist()[0]
        post_trial_num = post_scene_order.loc[post_trial_index, "trial_id"]
        # this mean we now know both the trial #, the image id and we can also grab the condition to help sort

        if brain_flag == "MNI":
            # category_weight_study_comp[counter]=cate_repress_study[trial-1,:]
            category_weight_pre_comp[counter] = cate_repress_pre[pre_trial_num - 1, :]
            category_weight_post_comp[counter] = cate_repress_post[
                post_trial_num - 1, :
            ]

            # This is to get the fidelity of the current item/trial from pre to post (scene-weighted)
            pre_post_trial_cw_fidelity = np.corrcoef(
                category_weight_pre_comp[counter, :],
                category_weight_post_comp[counter, :],
            )
            # This is to get the fidelity of the current item/trial from study to post (scene-weighted)
            # study_post_trial_cw_fidelity=np.corrcoef(category_weight_study_comp[counter,:],category_weight_post_comp[counter,:])
            # #This is to get the fidelity of the current item/trial from pre to study (scene-weighted)
            # pre_study_trial_cw_fidelity=np.corrcoef(category_weight_pre_comp[counter,:],category_weight_study_comp[counter,:])

        # unoperated is being dropped for now since this loop is focused on the study stims
        # if image_condition==0:
        #     LSS_unoperated_dict['image ID: %s' % image_id] = LSS_trial_fidelity[1][0]
        #     LSA_unoperated_dict['image ID: %s' % image_id] = LSA_trial_fidelity[1][0]
        if image_condition == 1:
            change_cw_maintain_dict[
                "image ID: %s" % study_image_id
            ] = pre_post_trial_cw_fidelity[1][0]

        elif image_condition == 2:
            change_cw_replace_dict[
                "image ID: %s" % study_image_id
            ] = pre_post_trial_cw_fidelity[1][0]

        elif image_condition == 3:
            change_cw_suppress_dict[
                "image ID: %s" % study_image_id
            ] = pre_post_trial_cw_fidelity[1][0]

        counter = counter + 1

    if not os.path.exists(
        os.path.join(
            container_path,
            "sub-0%s" % subID,
            "Representational_Changes_%s" % brain_flag,
        )
    ):
        os.makedirs(
            os.path.join(
                container_path,
                "sub-0%s" % subID,
                "Representational_Changes_%s" % brain_flag,
            ),
            exist_ok=True,
        )

    category_pre_post_comp = np.corrcoef(
        category_weight_pre_comp, category_weight_post_comp
    )
    temp_df = pd.DataFrame(data=category_pre_post_comp)
    temp_df.to_csv(
        os.path.join(
            container_path,
            "sub-0%s" % subID,
            "Representational_Changes_%s" % brain_flag,
            "scene_weight_pre_post_RSA.csv",
        )
    )
    del temp_df

    # here is where I need to add in some code to sort these dictionaries by the memory result. For each sub I have a file I can load: 'memory_and_familiar_sub-00x.csv'
    # then I just need to take the imageID, and see if the memory column is a 1 or a 0... this will allow me to split up the data better and visualize better
    memory_csv = pd.read_csv(
        "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/subject_designs/memory_and_familiar_sub-0%s.csv"
        % subID
    )

    catew_pre_df = pd.DataFrame()
    catew_pre_df["maintain"] = np.array(list(change_cw_maintain_dict.values()))
    catew_pre_df["replace"] = np.array(list(change_cw_replace_dict.values()))
    catew_pre_df["suppress"] = np.array(list(change_cw_suppress_dict.values()))
    catew_pre_df.to_csv(
        os.path.join(
            container_path,
            "sub-0%s" % subID,
            "Representational_Changes_%s" % brain_flag,
            "sceneweighted_pre_post_fidelity.csv",
        )
    )

    catew_remembered_df = pd.DataFrame(
        columns=["maintain", "replace", "suppress"], index=range(0, 30)
    )
    catew_forgot_df = pd.DataFrame(
        columns=["maintain", "replace", "suppress"], index=range(0, 30)
    )

    operation_list = np.repeat(["maintain", "replace", "suppress"], 30)

    indexer_r = 0
    indexer_f = 0

    indexer_r_agg = 0
    indexer_f_agg = 0
    for img_key in change_cw_maintain_dict.keys():
        img_num = re.split("(\d+)", img_key)
        img_num = int(img_num[1])
        if (memory_csv["memory"][memory_csv["image_num"] == img_num]).values[0] == 1:
            catew_remembered_df.loc[indexer_r]["maintain"] = change_cw_maintain_dict[
                "image ID: %s" % img_num
            ]

            indexer_r = indexer_r + 1

        elif (memory_csv["memory"][memory_csv["image_num"] == img_num]).values[0] == 0:
            catew_forgot_df.loc[indexer_f]["maintain"] = change_cw_maintain_dict[
                "image ID: %s" % img_num
            ]

            indexer_f = indexer_f + 1

    indexer_r = 0
    indexer_f = 0
    for img_key in change_cw_replace_dict.keys():
        img_num = re.split("(\d+)", img_key)
        img_num = int(img_num[1])

        if (memory_csv["memory"][memory_csv["image_num"] == img_num]).values[0] == 1:
            catew_remembered_df.loc[indexer_r]["replace"] = change_cw_replace_dict[
                "image ID: %s" % img_num
            ]

            indexer_r = indexer_r + 1

        elif (memory_csv["memory"][memory_csv["image_num"] == img_num]).values[0] == 0:
            catew_forgot_df.loc[indexer_f]["replace"] = change_cw_replace_dict[
                "image ID: %s" % img_num
            ]

            indexer_f = indexer_f + 1

    indexer_r = 0
    indexer_f = 0
    for img_key in change_cw_suppress_dict.keys():
        img_num = re.split("(\d+)", img_key)
        img_num = int(img_num[1])

        if (memory_csv["memory"][memory_csv["image_num"] == img_num]).values[0] == 1:
            catew_remembered_df.loc[indexer_r]["suppress"] = change_cw_suppress_dict[
                "image ID: %s" % img_num
            ]

            indexer_r = indexer_r + 1

        elif (memory_csv["memory"][memory_csv["image_num"] == img_num]).values[0] == 0:
            catew_forgot_df.loc[indexer_f]["suppress"] = change_cw_suppress_dict[
                "image ID: %s" % img_num
            ]

            indexer_f = indexer_f + 1

    # itemw_remembered_df.dropna(inplace=True)
    # itemw_forgot_df.dropna(inplace=True)

    catew_remembered_df.to_csv(
        os.path.join(
            container_path,
            "sub-0%s" % subID,
            "Representational_Changes_%s" % brain_flag,
            "cateweighted_pre_post_remember_fidelity.csv",
        )
    )
    catew_forgot_df.to_csv(
        os.path.join(
            container_path,
            "sub-0%s" % subID,
            "Representational_Changes_%s" % brain_flag,
            "cateweighted_pre_post_forgot_fidelity.csv",
        )
    )

    print("Subject is done... saving everything")
    print(
        "==============================================================================="
    )

# now I want to take the subject level "fidelity" results and look at it on a group level - I will also want to quantify the off diagonal of the RSA to random pairs

group_cate_weighted_pre_post = pd.DataFrame()

group_catew_remember_pre_post = pd.DataFrame()
group_catew_forgot_pre_post = pd.DataFrame()

group_by_sub_catew_remember_pre_post = pd.DataFrame()
group_by_sub_catew_forgot_pre_post = pd.DataFrame()

for subID in subs:
    data_path = os.path.join(
        container_path, "sub-0%s" % subID, "Representational_Changes_%s" % brain_flag
    )

    cate_pre_post = find("scene*pre*post*fidelity*", data_path)
    group_cate_weighted_pre_post = group_cate_weighted_pre_post.append(
        pd.read_csv(cate_pre_post[0], usecols=[1, 2, 3]), ignore_index=True
    )

    # now pull the ones sorted by memory:
    cate_r_pre_post = os.path.join(
        data_path, "cateweighted_pre_post_remember_fidelity.csv"
    )
    group_catew_remember_pre_post = group_catew_remember_pre_post.append(
        pd.read_csv(cate_r_pre_post, usecols=[1, 2, 3]), ignore_index=True
    )

    cate_f_pre_post = os.path.join(
        data_path, "cateweighted_pre_post_forgot_fidelity.csv"
    )
    group_catew_forgot_pre_post = group_catew_forgot_pre_post.append(
        pd.read_csv(cate_f_pre_post, usecols=[1, 2, 3]), ignore_index=True
    )

    if subID in [
        "06",
        "07",
        "14",
        "23",
    ]:  # these subjects have an operation without forgetting, thus this messes up my subject level assesment and I am leaving them out for now
        continue

    # this drops the index of the operation but the pattern is [maintain, replace, suppress] since I am getting operation averages per sub
    group_by_sub_catew_remember_pre_post = group_by_sub_catew_remember_pre_post.append(
        pd.read_csv(cate_r_pre_post, usecols=[1, 2, 3]).mean(), ignore_index=True
    )

    group_by_sub_catew_forgot_pre_post = group_by_sub_catew_forgot_pre_post.append(
        pd.read_csv(cate_f_pre_post, usecols=[1, 2, 3]).mean(), ignore_index=True
    )


if not os.path.exists(
    os.path.join(
        container_path, "group_model", "Representational_Changes_%s" % brain_flag
    )
):
    os.makedirs(
        os.path.join(
            container_path, "group_model", "Representational_Changes_%s" % brain_flag
        ),
        exist_ok=True,
    )
#######################

grouped_by_sub_catew_remember_pre_post = group_by_sub_catew_remember_pre_post.melt()
grouped_by_sub_catew_remember_pre_post["memory"] = "remembered"
grouped_by_sub_catew_forgot_pre_post = group_by_sub_catew_forgot_pre_post.melt()
grouped_by_sub_catew_forgot_pre_post["memory"] = "forgot"

grouped_catew_memory_pre_post = pd.concat(
    (grouped_by_sub_catew_remember_pre_post, grouped_by_sub_catew_forgot_pre_post)
)

fig = sns.barplot(
    data=grouped_catew_memory_pre_post,
    x="variable",
    y="value",
    hue="memory",
    ci=95,
    palette=["gray", "white"],
    edgecolor=".5",
)
fig.set_xlabel("Operations")
fig.set_ylabel("Fidelity of item-RSA")
fig.set_title(
    "Category Weighted (Group Level) - Pre vs. Post RSA", loc="center", wrap=True
)
plt.legend(loc="upper left")
plt.savefig(
    os.path.join(
        container_path,
        "group_model",
        "Representational_Changes_%s" % brain_flag,
        "Group_Category_Weighted_pre_post_summary.png",
    )
)
plt.clf()

######

grouped_catew_memory_pre_post.rename(
    columns={"variable": "operation", "value": "fidelity"}
)

# new plotting code to prepare for pub:
fig = sns.barplot(
    data=grouped_catew_memory_pre_post,
    x="operation",
    y="fidelity",
    hue="memory",
    ci=95,
    palette={"remembered": "black", "forgot": "grey"},
    edgecolor=".7",
)
for bar_group, desaturate_value in zip(fig.containers, [0.4, 1]):
    for bar, color in zip(bar_group, ["green", "blue", "red"]):
        bar.set_facecolor(sns.desaturate(color, desaturate_value))

# sns.stripplot(x='operation',y='fidelity',hue='memory',data=plotting_df,palette=['black'],dodge=True, jitter=False, alpha=0.15, ax=fig)
fig.set_title("Category Weighted - Pre vs. Post RSA", loc="center", wrap=True)
fig.set_xlabel("Operations")
fig.set_ylabel("Fidelity of RSA")
plt.tight_layout()
fig.set_ylim([0, 0.05])

from matplotlib.legend_handler import HandlerTuple

fig.legend(
    handles=[tuple(bar_group) for bar_group in fig.containers],
    labels=[bar_group.get_label() for bar_group in fig.containers],
    title=fig.legend_.get_title().get_text(),
    handlelength=4,
    handler_map={tuple: HandlerTuple(ndivide=None, pad=0.1)},
)

plt.savefig(
    os.path.join(
        container_path,
        "group_model",
        "Representational_Changes_%s" % brain_flag,
        "Allsubs_Category_Weighted_pre_post_summary.svg",
    )
)
plt.clf()
