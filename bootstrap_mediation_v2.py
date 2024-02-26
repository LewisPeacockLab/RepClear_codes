# code to bootstrap results for operation evidence and memory outcome - including versions for usual bootstrap and median split bootstrap
import warnings
import sys

# if not sys.warnoptions:
#     warnings.simplefilter("ignore")
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nibabel as nib
from nilearn.image import clean_img
from nilearn.signal import clean
import scipy as scipy
import scipy.io as sio
import seaborn as sns

cmap = sns.color_palette("crest", as_cmap=True)
import fnmatch
import pickle
import time
from random import choices  # random sampling with replacement
from sklearn.svm import LinearSVC
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    LinearRegression,
)
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneGroupOut,
    PredefinedSplit,
)  # train_test_split, PredefinedSplit, cross_validate, cross_val_predict,
from sklearn.feature_selection import (
    SelectFpr,
    f_classif,
)  # VarianceThreshold, SelectKBest,
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from nilearn.input_data import NiftiMasker, MultiNiftiMasker
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
from joblib import Parallel, delayed
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests


# global consts
subIDs = [
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
temp_subIDs = [
    "002",
    "003",
    "004",
    "005",
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
    "024",
    "025",
    "026",
]


phases = ["rest", "preremoval", "study", "postremoval"]
runs = np.arange(6) + 1
spaces = {"T1w": "T1w", "MNI": "MNI152NLin2009cAsym"}
descs = ["brain_mask", "preproc_bold"]
ROIs = ["wholebrain"]
shift_sizes_TR = [5]

save = 1
shift_size_TR = shift_sizes_TR[0]


stim_labels = {0: "Rest", 1: "Scenes", 2: "Faces"}

operation_labels = {0: "Rest", 1: "Maintain", 2: "Replace", 3: "Suppress"}

workspace = "scratch"
data_dir = "/Volumes/zbg_eHD/Zachary_Data/processed/"
param_dir = "/scratch/06873/zbretton/repclear_dataset/BIDS/params/"


def organize_evidence(subID, space, task, condition, save=True):
    ROIs = ["wholebrain"]

    print("\n *** loading evidence values from subject dataframe ***")

    sub_dir = os.path.join(data_dir, f"sub-{subID}")

    in_fname_template_2 = (
        f"sub-{subID}_{space}_trained-preremoval_tested-study_evidence.csv"
    )

    in_fname_template = f"sub-{subID}_{space}_{task}_operation_evidence.csv"

    sub_df = pd.read_csv(os.path.join(sub_dir, in_fname_template))
    sub_df.drop(
        columns=sub_df.columns[0], axis=1, inplace=True
    )  # now drop the extra index column

    category_sub_df = pd.read_csv(os.path.join(sub_dir, in_fname_template_2))
    category_sub_df.drop(
        columns=category_sub_df.columns[0], axis=1, inplace=True
    )  # now drop the extra index column

    sub_images, sub_index = np.unique(
        sub_df["image_id"], return_index=True
    )  # this searches through the dataframe to find each occurance of the image_id. This allows me to find the start of each trial, and linked to the image_ID #
    category_sub_images, category_sub_index = np.unique(
        category_sub_df["image_id"], return_index=True
    )  # this searches through the dataframe to find each occurance of the image_id. This allows me to find the start of each trial, and linked to the image_ID #
    # category_df contains the 0's for the shift, while operation evidence already has removed that

    # now to sort the trials, we need to figure out what the operation performed is:
    sub_condition_list = sub_df["operation"][sub_index].values.astype(
        int
    )  # so using the above indices, we will now grab what the operation is on each image
    category_sub_images, category_sub_index = (
        category_sub_images[1:],
        category_sub_index[1:],
    )  # need to drop the first index to help line up image_id order

    counter = 0
    maintain_trials_mean = {}
    replace_trials_mean = {}
    suppress_trials_mean = {}

    maintain_trials = {}
    replace_trials = {}
    suppress_trials = {}

    m_memory_outcome = {}
    r_memory_outcome = {}
    s_memory_outcome = {}

    maintain_category_trials = {}
    replace_category_trials = {}
    suppress_category_trials = {}

    maintain_category_trials_mean = {}
    replace_category_trials_mean = {}
    suppress_category_trials_mean = {}

    subject_design_dir = "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/subject_designs/"

    memory_file_path = os.path.join(
        subject_design_dir, f"memory_and_familiar_sub-{subID}.csv"
    )
    memory_df = pd.read_csv(memory_file_path)

    for i in np.unique(sub_df["image_id"].values):
        if i == 0:
            continue
        else:
            response = memory_df[memory_df["image_num"] == i].resp.values[0]

            if response == 4:
                sub_df.loc[sub_df["image_id"] == i, "memory"] = 1
                category_sub_df.loc[category_sub_df["image_id"] == i, "memory"] = 1
            else:
                sub_df.loc[sub_df["image_id"] == i, "memory"] = 0
                category_sub_df.loc[category_sub_df["image_id"] == i, "memory"] = 0

    if task == "study":
        x = 9
    if task == "postremoval":
        x = 5

    category_sub_df["scene_evi"] = stats.zscore(
        category_sub_df["scene_evi"]
    )  # zscore the scene evidence to get a better sense of the relationship when across subjects

    for i in sub_condition_list:
        if (
            i == 0
        ):  # this is the first rest period (because we had to shift of hemodynamics. So this "0" condition is nothing)
            print("i==0")
            counter += 1
            continue
        elif i == 1:
            temp_image = sub_images[counter]
            maintain_trials[temp_image] = sub_df[["maintain_evi"]][
                sub_index[counter] : sub_index[counter] + 4
            ].values  # this gets the 4TR operation time
            maintain_trials_mean[temp_image] = sub_df[["maintain_evi"]][
                sub_index[counter] : sub_index[counter] + 4
            ].values.mean()

            maintain_category_trials[temp_image] = category_sub_df[["scene_evi"]][
                category_sub_index[counter] + 2 : category_sub_index[counter] + 6
            ].values  # since category_df had the stimuli times, we need to shift 2 to get the operation time from the stimuli onset
            maintain_category_trials_mean[temp_image] = category_sub_df[["scene_evi"]][
                category_sub_index[counter] + 2 : category_sub_index[counter] + 6
            ].values.mean()

            m_memory_outcome[temp_image] = sub_df[sub_df["image_id"] == temp_image][
                "memory"
            ].values[0]
            counter += 1

        elif i == 2:
            temp_image = sub_images[counter]
            replace_trials[temp_image] = sub_df[["replace_evi"]][
                sub_index[counter] : sub_index[counter] + 4
            ].values
            replace_trials_mean[temp_image] = sub_df[["replace_evi"]][
                sub_index[counter] : sub_index[counter] + 4
            ].values.mean()

            replace_category_trials[temp_image] = category_sub_df[["scene_evi"]][
                category_sub_index[counter] + 2 : category_sub_index[counter] + 6
            ].values
            replace_category_trials_mean[temp_image] = category_sub_df[["scene_evi"]][
                category_sub_index[counter] + 2 : category_sub_index[counter] + 6
            ].values.mean()

            r_memory_outcome[temp_image] = sub_df[sub_df["image_id"] == temp_image][
                "memory"
            ].values[0]
            counter += 1

        elif i == 3:
            temp_image = sub_images[counter]
            suppress_trials[temp_image] = sub_df[["suppress_evi"]][
                sub_index[counter] : sub_index[counter] + 4
            ].values
            suppress_trials_mean[temp_image] = sub_df[["suppress_evi"]][
                sub_index[counter] : sub_index[counter] + 4
            ].values.mean()

            suppress_category_trials[temp_image] = category_sub_df[["scene_evi"]][
                category_sub_index[counter] + 2 : category_sub_index[counter] + 6
            ].values
            suppress_category_trials_mean[temp_image] = category_sub_df[["scene_evi"]][
                category_sub_index[counter] + 2 : category_sub_index[counter] + 6
            ].values.mean()

            s_memory_outcome[temp_image] = sub_df[sub_df["image_id"] == temp_image][
                "memory"
            ].values[0]
            counter += 1

    # need to grab the total trials of the conditions, will need to then pull memory evidence of these trials and append to the dataframe:
    all_maintain = pd.DataFrame(
        data=maintain_trials_mean.values(),
        index=maintain_trials_mean.keys(),
        columns=["evidence"],
    )
    all_replace = pd.DataFrame(
        data=replace_trials_mean.values(),
        index=replace_trials_mean.keys(),
        columns=["evidence"],
    )
    all_suppress = pd.DataFrame(
        data=suppress_trials_mean.values(),
        index=suppress_trials_mean.keys(),
        columns=["evidence"],
    )

    all_category_maintain = pd.DataFrame(
        data=maintain_category_trials_mean.values(),
        index=maintain_category_trials_mean.keys(),
        columns=["evidence"],
    )
    all_category_replace = pd.DataFrame(
        data=replace_category_trials_mean.values(),
        index=replace_category_trials_mean.keys(),
        columns=["evidence"],
    )
    all_category_suppress = pd.DataFrame(
        data=suppress_category_trials_mean.values(),
        index=suppress_category_trials_mean.keys(),
        columns=["evidence"],
    )

    all_suppress[
        "memory"
    ] = (
        s_memory_outcome.values()
    )  # now we have the mean evidence and the memory outcome of the subject in one DF
    all_replace[
        "memory"
    ] = (
        r_memory_outcome.values()
    )  # now we have the mean evidence and the memory outcome of the subject in one DF
    all_maintain[
        "memory"
    ] = (
        m_memory_outcome.values()
    )  # now we have the mean evidence and the memory outcome of the subject in one DF

    all_category_suppress[
        "memory"
    ] = (
        s_memory_outcome.values()
    )  # now we have the mean evidence and the memory outcome of the subject in one DF
    all_category_replace[
        "memory"
    ] = (
        r_memory_outcome.values()
    )  # now we have the mean evidence and the memory outcome of the subject in one DF
    all_category_maintain[
        "memory"
    ] = (
        m_memory_outcome.values()
    )  # now we have the mean evidence and the memory outcome of the subject in one DF

    # now we have a dataframe with an index of the item #, a column for mean classifier evidence and a column for memory outcome:
    # we need to then loop across subjects and append their dataframe, so we are left with a dataframe of 30 trials per subject (22 total).
    # then we can create a function to loop through that total dataframe to bootstrap out linear regression betas and median split data.

    if condition == "suppress":
        new_sub_df = all_suppress.reset_index(
            drop=True
        )  # this removes the information of which item it was
        new_category_sub_df = all_category_suppress.reset_index(
            drop=True
        )  # this removes the information of which item it was
    elif condition == "maintain":
        new_sub_df = all_maintain.reset_index(
            drop=True
        )  # this removes the information of which item it was
        new_category_sub_df = all_category_maintain.reset_index(
            drop=True
        )  # this removes the information of which item it was
    elif condition == "replace":
        new_sub_df = all_replace.reset_index(
            drop=True
        )  # this removes the information of which item it was
        new_category_sub_df = all_category_replace.reset_index(
            drop=True
        )  # this removes the information of which item it was

    new_sub_df["evidence"] = stats.zscore(new_sub_df["evidence"])

    # save for future use
    if save:
        sub_dir = os.path.join(data_dir, f"sub-{subID}")
        out_fname_template = f"sub-{subID}_{space}_{task}_{condition}_zscore_evidence_withmemory_dataframe.csv"
        print(
            f"\n Saving the sorted {condition} evidence dataframe for {subID} - phase: {task} - as {out_fname_template}"
        )
        new_sub_df.to_csv(os.path.join(sub_dir, out_fname_template))

        out_fname_template_2 = f"sub-{subID}_{space}_{task}_{condition}_zscore_category_evidence_withmemory_dataframe.csv"
        print(
            f"\n Saving the sorted {condition} category evidence dataframe for {subID} - phase: {task} - as {out_fname_template}"
        )
        new_category_sub_df.to_csv(os.path.join(sub_dir, out_fname_template_2))
    return new_sub_df, new_category_sub_df


def group_bootstrap(space):
    conditions = ["maintain", "replace", "suppress"]

    total_df = pd.DataFrame(columns=["betas", "condition"])

    for condition in conditions:
        temp_total_df = pd.DataFrame(columns=["betas", "condition"])

        group_evidence_df = pd.DataFrame()
        group_category_evidence_df = pd.DataFrame()
        group_bootstrap_median = pd.DataFrame(columns=["memory", "split"])

        for subID in temp_subIDs:
            temp_subject_df, temp_category_subject_df = organize_evidence(
                subID, space, "study", condition
            )
            group_evidence_df = pd.concat(
                [group_evidence_df, temp_subject_df], ignore_index=True, sort=False
            )
            group_category_evidence_df = pd.concat(
                [group_category_evidence_df, temp_category_subject_df],
                ignore_index=True,
                sort=False,
            )

        # now that the df's all stacked, need to iterate and create my "bootstrap subjects":
        iterations = 10000
        bootstrap_betas = []  # operation evidence predicting memory
        bootstrap_high_oper = []
        bootstrap_low_oper = []

        bootstrap_category_beta = []  # category evidence predicting memory
        bootstrap_oper_pred_cate = []  # operation evidence predicting category evidence
        bootstrap_cate_high_oper = []  # category evidence when operation is high
        bootstrap_cate_low_oper = []  # category evidence when operation is low

        group_evidence_df["scene_evi"] = group_category_evidence_df["evidence"]

        total_tic = time.perf_counter()  # get a timer for the WHOLE bootstrap
        for i in range(iterations):
            tic = time.perf_counter()  # get a timer for each bootstrap iteration
            bootstrap_sub_df = pd.DataFrame(
                columns=[
                    "operation_evi",
                    "scene_evi",
                    "memory",
                    "beta",
                    "category_beta",
                    "oper_cate_beta",
                    "high_evi",
                    "low_evi",
                    "high_scene",
                    "low_scene",
                ]
            )
            bootstrap_mediation_df = pd.DataFrame(
                columns=["sample_id", "subject", "operation_evi", "scene_evi", "memory"]
            )

            # now we need to iterate through this dataframe and pull 30 trials with replacement for each subject:
            for sub in temp_subIDs:
                temp_sub_df = pd.DataFrame(columns=["evidence", "memory", "scene_evi"])
                temp_sub_df = group_evidence_df.sample(n=30, replace=True)

                while (
                    np.unique(temp_sub_df["memory"].values).size
                ) < 2:  # since we NEED at least a remember and a forget trial to do the regression, we check the memory outcome and if no remember or forgotten trial is sampled... resample it
                    temp_sub_df = group_evidence_df.sample(n=30, replace=True)

                temp_df_high, temp_df_low = [
                    x
                    for _, x in temp_sub_df.groupby(
                        temp_sub_df["evidence"] < temp_sub_df.evidence.median()
                    )
                ]

                # fit the "bootstrap subject" data to regression: collect coef
                temp_LR = LogisticRegression().fit(
                    temp_sub_df["evidence"].values.reshape(-1, 1),
                    temp_sub_df["memory"].values.reshape(-1),
                )  # operation evidence predicting memory outcome
                sub_beta = temp_LR.coef_[0][0]

                temp_LR_category = LogisticRegression().fit(
                    temp_sub_df["scene_evi"].values.reshape(-1, 1),
                    temp_sub_df["memory"].values.reshape(-1),
                )  # category evidence predicting memory outcome
                sub_category_beta = temp_LR_category.coef_[0][0]

                temp_LR_oper_pred_cate = LinearRegression().fit(
                    temp_sub_df["evidence"].values.reshape(-1, 1),
                    temp_sub_df["scene_evi"].values.reshape(-1, 1),
                )  # operation evidence predicting category evidence
                sub_oper_pred_cate_beta = temp_LR_oper_pred_cate.coef_[0][0]

                temp_df = pd.DataFrame(
                    columns=[
                        "operation_evi",
                        "scene_evi",
                        "memory",
                        "beta",
                        "category_beta",
                        "oper_cate_beta",
                    ]
                )
                temp_df["operation_evi"] = [temp_sub_df["evidence"].mean()]
                temp_df["scene_evi"] = [temp_sub_df["scene_evi"].mean()]
                temp_df["memory"] = [temp_sub_df["memory"].mean()]
                temp_df["beta"] = [sub_beta]
                temp_df["category_beta"] = [sub_category_beta]
                temp_df["oper_cate_beta"] = [sub_oper_pred_cate_beta]

                temp_df_mediation = pd.DataFrame(
                    columns=[
                        "sample_id",
                        "subject",
                        "operation_evi",
                        "scene_evi",
                        "memory",
                    ]
                )
                temp_df_mediation["sample_id"] = np.repeat(i, 30)
                temp_df_mediation["subject"] = np.repeat(sub, 30)
                temp_df_mediation["operation_evi"] = temp_sub_df["evidence"].values
                temp_df_mediation["scene_evi"] = temp_sub_df["scene_evi"].values
                temp_df_mediation["memory"] = temp_sub_df["memory"].values

                # temp_df['high_evi']=[temp_df_high['memory'].mean()]
                # temp_df['low_evi']=[temp_df_low['memory'].mean()]
                # temp_df['high_scene']=[temp_df_high['scene_evi'].mean()]
                # temp_df['low_scene']=[temp_df_low['scene_evi'].mean()]

                bootstrap_sub_df = pd.concat(
                    [bootstrap_sub_df, temp_df], ignore_index=True, sort=False
                )
                bootstrap_mediation_df = pd.concat(
                    [bootstrap_mediation_df, temp_df_mediation],
                    ignore_index=True,
                    sort=False,
                )

                del temp_df, temp_sub_df, temp_df_mediation

            toc = time.perf_counter()
            print(f"Bootstrap iteration {i} completed in {toc - tic:0.4f} seconds")

            bootstrap_betas = np.append(
                bootstrap_betas, bootstrap_sub_df["beta"].mean()
            )
            # bootstrap_high_oper=np.append(bootstrap_high_oper,bootstrap_sub_df['high_evi'].mean())
            # bootstrap_low_oper=np.append(bootstrap_low_oper,bootstrap_sub_df['low_evi'].mean())

            bootstrap_category_beta = np.append(
                bootstrap_category_beta, bootstrap_sub_df["category_beta"].mean()
            )  # category evidence predicting memory
            bootstrap_oper_pred_cate = np.append(
                bootstrap_oper_pred_cate, bootstrap_sub_df["oper_cate_beta"].mean()
            )  # operation evidence predicting category evidence
            # bootstrap_cate_high_oper=np.append(bootstrap_cate_high_oper,bootstrap_sub_df['high_scene'].mean()) #category evidence when operation is high
            # bootstrap_cate_low_oper=np.append(bootstrap_cate_low_oper,bootstrap_sub_df['low_scene'].mean()) #category evidence when operation is low

            out_fname_template = (
                f"bootstrap_{space}_study_{condition}_iteration_{i}_dataframe.csv"
            )
            print(
                f"\n Saving the bootstrapping dataframe for iteration {i}, phase: study - as {out_fname_template}"
            )
            bootstrap_mediation_df.to_csv(
                os.path.join(data_dir, "iteration_data", out_fname_template)
            )

            del bootstrap_mediation_df
        total_toc = time.perf_counter()
        print("##############################################################")
        print(f"Bootstrap completed in {total_toc - total_tic:0.4f} seconds")
        print("##############################################################")

        ci_value_upper = np.percentile(bootstrap_betas, 95)
        ci_value_lower = np.percentile(bootstrap_betas, 5)
        print(f"95% CI for Operation on memory: {ci_value_lower} to {ci_value_upper}")

        ci_value_2_upper = np.percentile(bootstrap_oper_pred_cate, 95)
        ci_value_2_lower = np.percentile(bootstrap_oper_pred_cate, 5)
        print(
            f"95% CI for Operation on category: {ci_value_2_lower} to {ci_value_2_upper}"
        )

        ci_value_3_upper = np.percentile(bootstrap_category_beta, 95)
        ci_value_3_lower = np.percentile(bootstrap_category_beta, 5)
        print(
            f"95% CI for category on memory: {ci_value_3_lower} to {ci_value_3_upper}"
        )

        export_df = pd.DataFrame(
            columns=["oper_beta", "cate_beta", "oper_pred_cate_beta"]
        )
        export_df["oper_beta"] = bootstrap_betas
        export_df["cate_beta"] = bootstrap_category_beta
        export_df["oper_pred_cate_beta"] = bootstrap_oper_pred_cate

        out_fname_template = f"bootstrap_{space}_study_{condition}_dataframe.csv"
        print(
            f"\n Saving the bootstrapping dataframe for phase: study - as {out_fname_template}"
        )
        export_df.to_csv(os.path.join(data_dir, out_fname_template))


def plot_viusals(space):
    conditions = ["maintain", "replace", "suppress"]
    # space='MNI'

    total_df = pd.DataFrame(
        columns=[
            "operation_memory",
            "category_memory",
            "operation_category",
            "condition",
        ]
    )
    plt.style.use("seaborn-paper")

    for condition in conditions:
        load_df = pd.read_csv(
            os.path.join(data_dir, f"bootstrap_{space}_study_{condition}_dataframe.csv")
        )
        temp_total_df = pd.DataFrame(
            columns=[
                "operation_memory",
                "category_memory",
                "operation_category",
                "condition",
            ]
        )
        if condition == "maintain":
            temp_total_df["operation_memory"] = load_df["oper_beta"]
            temp_total_df["category_memory"] = load_df["cate_beta"]
            temp_total_df["operation_category"] = load_df["oper_pred_cate_beta"]
            temp_total_df["condition"] = np.repeat("maintain", len(load_df))
            total_df = pd.concat(
                [total_df, temp_total_df], ignore_index=True, sort=False
            )
        elif condition == "replace":
            temp_total_df["operation_memory"] = load_df["oper_beta"]
            temp_total_df["category_memory"] = load_df["cate_beta"]
            temp_total_df["operation_category"] = load_df["oper_pred_cate_beta"]
            temp_total_df["condition"] = np.repeat("replace", len(load_df))
            total_df = pd.concat(
                [total_df, temp_total_df], ignore_index=True, sort=False
            )
        elif condition == "suppress":
            temp_total_df["operation_memory"] = load_df["oper_beta"]
            temp_total_df["category_memory"] = load_df["cate_beta"]
            temp_total_df["operation_category"] = load_df["oper_pred_cate_beta"]
            temp_total_df["condition"] = np.repeat("suppress", len(load_df))
            total_df = pd.concat(
                [total_df, temp_total_df], ignore_index=True, sort=False
            )

    ax = sns.violinplot(
        data=total_df,
        x="condition",
        y="operation_memory",
        palette=["green", "blue", "red"],
    )
    ax.set(xlabel=f"Operation", ylabel="Beta")
    ax.set_title(
        f"Operation Evidence predicting Memory - 10,000 Bootstrap Iterations",
        loc="center",
        wrap=True,
    )
    ax.axhline(0, color="k", linestyle="--")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            data_dir,
            "figs",
            f"{space}_group_level_maintain+replace+suppress_bootstrap_operation_memory.png",
        )
    )
    plt.savefig(
        os.path.join(
            data_dir,
            "figs",
            f"{space}_group_level_maintain+replace+suppress_bootstrap_operation_memory.svg",
        )
    )
    plt.clf()

    ax = sns.violinplot(
        data=total_df,
        x="condition",
        y="category_memory",
        palette=["green", "blue", "red"],
    )
    ax.set(xlabel=f"Operation", ylabel="Beta")
    ax.set_title(
        f"Category Evidence predicting Memory - 10,000 Bootstrap Iterations",
        loc="center",
        wrap=True,
    )
    ax.axhline(0, color="k", linestyle="--")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            data_dir,
            "figs",
            f"{space}_group_level_maintain+replace+suppress_bootstrap_category_memory.png",
        )
    )
    plt.savefig(
        os.path.join(
            data_dir,
            "figs",
            f"{space}_group_level_maintain+replace+suppress_bootstrap_category_memory.svg",
        )
    )
    plt.clf()

    ax = sns.violinplot(
        data=total_df,
        x="condition",
        y="operation_category",
        palette=["green", "blue", "red"],
    )
    ax.set(xlabel=f"Operation", ylabel="Beta")
    ax.set_title(
        f"Operation Evidence predicting Category Evidence - 10,000 Bootstrap Iterations",
        loc="center",
        wrap=True,
    )
    ax.axhline(0, color="k", linestyle="--")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            data_dir,
            "figs",
            f"{space}_group_level_maintain+replace+suppress_bootstrap_operation_category.png",
        )
    )
    plt.savefig(
        os.path.join(
            data_dir,
            "figs",
            f"{space}_group_level_maintain+replace+suppress_bootstrap_operation_category.svg",
        )
    )
    plt.clf()

    m_ci_upper_value = np.percentile(
        total_df["operation_memory"][total_df["condition"] == "maintain"], 95
    )
    m_ci_lower_value = np.percentile(
        total_df["operation_memory"][total_df["condition"] == "maintain"], 5
    )

    r_ci_upper_value = np.percentile(
        total_df["operation_memory"][total_df["condition"] == "replace"], 95
    )
    r_ci_lower_value = np.percentile(
        total_df["operation_memory"][total_df["condition"] == "replace"], 5
    )

    s_ci_upper_value = np.percentile(
        total_df["operation_memory"][total_df["condition"] == "suppress"], 95
    )
    s_ci_lower_value = np.percentile(
        total_df["operation_memory"][total_df["condition"] == "suppress"], 5
    )
    print(
        f"95% CI for Operation on Memory: \n Maintain:{m_ci_lower_value} - {m_ci_upper_value} \n Replace: {r_ci_lower_value} - {r_ci_upper_value} \n Suppress: {s_ci_lower_value} - {s_ci_upper_value}"
    )

    m_ci_upper_value = np.percentile(
        total_df["category_memory"][total_df["condition"] == "maintain"], 95
    )
    m_ci_lower_value = np.percentile(
        total_df["category_memory"][total_df["condition"] == "maintain"], 5
    )

    r_ci_upper_value = np.percentile(
        total_df["category_memory"][total_df["condition"] == "replace"], 95
    )
    r_ci_lower_value = np.percentile(
        total_df["category_memory"][total_df["condition"] == "replace"], 5
    )

    s_ci_upper_value = np.percentile(
        total_df["category_memory"][total_df["condition"] == "suppress"], 95
    )
    s_ci_lower_value = np.percentile(
        total_df["category_memory"][total_df["condition"] == "suppress"], 5
    )
    print(
        f"95% CI for Category on Memory: \n Maintain:{m_ci_lower_value} - {m_ci_upper_value} \n Replace: {r_ci_lower_value} - {r_ci_upper_value} \n Suppress: {s_ci_lower_value} - {s_ci_upper_value}"
    )

    m_ci_upper_value = np.percentile(
        total_df["operation_category"][total_df["condition"] == "maintain"], 95
    )
    m_ci_lower_value = np.percentile(
        total_df["operation_category"][total_df["condition"] == "maintain"], 5
    )

    r_ci_upper_value = np.percentile(
        total_df["operation_category"][total_df["condition"] == "replace"], 95
    )
    r_ci_lower_value = np.percentile(
        total_df["operation_category"][total_df["condition"] == "replace"], 5
    )

    s_ci_upper_value = np.percentile(
        total_df["operation_category"][total_df["condition"] == "suppress"], 95
    )
    s_ci_lower_value = np.percentile(
        total_df["operation_category"][total_df["condition"] == "suppress"], 5
    )
    print(
        f"95% CI for Operation on Category: \n Maintain:{m_ci_lower_value} - {m_ci_upper_value} \n Replace: {r_ci_lower_value} - {r_ci_upper_value} \n Suppress: {s_ci_lower_value} - {s_ci_upper_value}"
    )


group_bootstrap("MNI")
plot_viusals("MNI")
