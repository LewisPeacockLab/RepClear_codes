# script to take in the trained model from clearmem and the PCA, apply to subjects and export results:
OUTDATED_IGNORE = 1
import os
import glob
import numpy as np
import pandas as pd
import pickle
import time
import json
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from nilearn.signal import clean
from nilearn.maskers import NiftiMasker
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import PredefinedSplit
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

import psutil
import nibabel as nib
from scipy.signal import resample
from scipy.stats import ttest_1samp

import fnmatch


def calccohensd(array1, mean2):
    mean1 = np.array(array1).mean()
    std = np.array(array1).std()

    cohens_d = (mean1 - mean2) / std
    return cohens_d


def load_model(filename):
    # Load the pickled models from disk
    with open(filename, "rb") as f:
        models = pickle.load(f)
    return models["classifier"], models["pca"]


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


# this function takes the mask data and applies it to the bold data
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


def sample_for_training(full_data, label_df, include_rest=False):
    """
    sample data by runs.
    Return: sampled labels and bold data
    """

    # operation_list: 1 - Maintain, 2 - Replace, 3 - Suppress
    # stim_on labels: 1 actual stim; 2 operation; 3 ITI; 0 rest between runs
    print("\n***** Subsampling data points by runs...")

    category_list = label_df["condition"]
    stim_on = label_df["stim_present"]
    run_list = label_df["run"]
    image_list = label_df["image_id"]

    # get faces
    oper_inds = np.where((stim_on == 2) | (stim_on == 3))[0]
    rest_inds = []

    runs = run_list.unique()[1:]

    if include_rest:
        print("Including resting category...")
        # get TR intervals for rest TRs between stims (stim_on == 2)
        rest_bools = ((run_list == runi) | (run_list == runj)) & (stim_on == 2)
        padded_bools = np.r_[
            False, rest_bools, False
        ]  # pad the rest_bools 1 TR before and after to separate it from trial information
        rest_diff = np.diff(
            padded_bools
        )  # get the pairwise diff in the array --> True for start and end indices of rest periods
        rest_intervals = rest_diff.nonzero()[0].reshape(
            (-1, 2)
        )  # each pair is the interval of rest periods

        # get desired time points in the middle of rest periods for rest samples; if 0.5, round up
        rest_intervals[:, -1] -= 1
        rest_inds = [
            np.ceil(np.average(interval)).astype(int) for interval in rest_intervals
        ] + [
            np.ceil(np.average(interval)).astype(int) + 1 for interval in rest_intervals
        ]

    operation_reg = category_list.values[oper_inds]
    run_reg = run_list.values[oper_inds]
    image_reg = image_list.values[oper_inds]

    # === get sample_bold & sample_regressor
    sample_bold = []
    sample_regressor = operation_reg
    sample_runs = run_reg

    sample_bold = full_data[oper_inds]

    return sample_bold, sample_regressor, sample_runs, image_reg


def load_process_data(subID, task, space, mask_paths):
    save = True
    runs = np.arange(3) + 1
    mask_ROIS = [mask_paths]

    # ======= generate file names to load
    # get list of data names
    fname_template = f"sub-{subID}_task-{task}_run-{{}}_resampled.nii.gz"
    bold_fnames = [fname_template.format(i) for i in runs]
    bold_paths = [
        os.path.join(
            "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/",
            f"sub-{subID}",
            "func_resampled",
            fname,
        )
        for fname in bold_fnames
    ]

    # get mask names
    mask_paths = xargs["feat_mask"]

    runs = np.arange(3) + 1

    # get confound filenames
    confound_fnames = [f"*{task}*{run}*confounds*.tsv" for run in runs]
    confound_paths = [
        os.path.join(
            "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/",
            f"sub-{subID}",
            "func",
            f,
        )
        for f in confound_fnames
    ]  # template for each run
    confound_paths = [glob.glob(p)[0] for p in confound_paths]  # actual file names

    # ======= load bold data & preprocess

    # ===== load data files
    print("\n*** Loading & cleaning data...")
    print("Loading bold data...")
    # loaded bold shape: (x/y/z x time))
    bolds = [nib.load(p) for p in bold_paths]

    print("Loading masks...")
    masks = nib.load(mask_paths)

    print("Loading confound files...")
    confounds = [pd.read_csv(p, sep="\t") for p in confound_paths]
    confounds_cleaned = [confound_cleaner(c) for c in confounds]

    # ===== for each run & ROI, mask & clean
    print("\n*** Masking & cleaing bold data...")
    cleaned_bolds = [[None for _ in range(len(runs))] for _ in range(len(mask_ROIS))]

    for rowi, mask in enumerate([masks]):
        print(f"Processing mask {rowi}...")
        for coli, (bold, confound) in enumerate(zip(bolds, confounds_cleaned)):
            print(f"Processing run {coli}...")

            # masked: time x vox
            masked = apply_mask(mask=mask.get_fdata(), target=bold.get_fdata())

            # *** clean: confound rows are time;
            cleaned_bolds[rowi][coli] = clean(
                masked, confounds=confound, t_r=1, detrend=False, standardize="zscore"
            )
            print(f"ROI {rowi}, run {coli}")
            print(f"shape: {cleaned_bolds[rowi][coli].shape}")

        # {ROI: time x vox}
        preproc_data = {
            ROI: np.vstack(run_data) for ROI, run_data in zip(mask_ROIS, cleaned_bolds)
        }

    print("processed data shape: ", [d.shape for d in preproc_data.values()])
    print("*** Done with preprocessing!")

    # save for future use
    if save:
        for ROI, run_data in preproc_data.items():
            if ROI == "VVS":
                ROI = "VTC"
            bold_dir = os.path.join(
                "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/",
                f"sub-{subID}",
                "func_resampled",
            )
            out_fname_template = f"sub-{subID}_{space}_{task}_{{}}_masked_cleaned.npy"
            out_fname = out_fname_template.format("feature_mask")
            print(f"Saving to file {bold_dir}/{out_fname}...")
            np.save(f"{bold_dir}/{out_fname}", run_data)

    # this will handle mutliple ROIs and then if needed save them into 1 dictionary
    try:
        full_dict = {**ready_data, **preproc_data}
    except:
        full_dict = {**preproc_data}
    # array: all_runs_time x all_ROI_vox
    full_data = np.hstack(list(full_dict.values()))
    return full_data


def get_shifted_labels(task, shift_size_TR, rest_tag=0):
    # load labels, & add hemodynamic shift to all vars

    def shift_timing(label_df, TR_shift_size, tag=0):
        # Shift 2D df labels by given TRs with paddings of tag
        # Input label_df must be time x nvars
        nvars = len(label_df.loc[0])
        shift = pd.DataFrame(
            np.zeros((TR_shift_size, nvars)) + tag, columns=label_df.columns
        )
        shifted = pd.concat([shift, label_df])
        return shifted[: len(label_df)]  # trim time points outside of scanning time

    print("\n***** Loading labels...")

    subject_design_dir = "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/subject_designs/"

    # using the task tag, we want to get the proper tag to pull the subject and phase specific dataframes
    if task == "preremoval":
        temp_task = "pre-localizer"
    if task == "postremoval":
        temp_task = "post-localizer"
    if task == "study":
        temp_task = "study"

    sub_design = f"*{xsubject_id}*{temp_task}*tr*"
    sub_design_file = find(sub_design, subject_design_dir)
    sub_design_matrix = pd.read_csv(
        sub_design_file[0]
    )  # this is the correct, TR by TR list of what happened during this subject's study phase

    shifted_df = shift_timing(sub_design_matrix, shift_size_TR, rest_tag)

    return shifted_df


def remove_label(X, Y, label):
    # Find the indices of the time points with the given label
    idx_remove = np.where(Y == label)[0]

    # Remove the time points with the given label from the data and regressors
    X_new = np.delete(X, idx_remove, axis=0)
    Y_new = np.delete(Y, idx_remove, axis=0)

    return X_new, Y_new


def rename_labels(Y, flag):
    # Create a new array to hold the renamed labels
    Y_new = np.zeros_like(Y, dtype="str")

    # Map the old label values to new label values based on the flag
    if flag == "clearmem":
        label_map = {1: "maintain", 2: "replace", 4: "suppress"}
    elif flag == "repclear":
        label_map = {1: "maintain", 2: "replace", 3: "suppress"}
    else:
        print("Invalid flag")
        return None

    # Rename the labels
    for old_label, new_label in label_map.items():
        idx = np.where(Y == old_label)[0]
        Y_new[idx] = new_label

    return Y_new


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


for subID in subIDs:
    print(f"Running sub-{subID}...")

    xsubject_id = subID
    xargs = {
        "subject_id": xsubject_id,
        "n_subj": 50,
        "align": "anatomical",
        "phase": "operation",
        "bold": "bold_mcf_brain_hpass_dt_mni_2mm",
        "patterns": "zpats_mni",
        "mask": "harvardoxford_gm_mask",
        "shift_tr": 10,
        "operation": "4",
        "n_seps": 20,
        "feat_top": 10000,
        "n_conds": 4,
        "condition": ["maintain", "replace_category", "supress", "clear"],
        "PCA": True,
        "classifier": "L2Logistic",
        "penalty": 50,
        "max_iter": 100,
        "patterns": "zpats_mni",
        "n_comps": 70,
    }

    xargs[
        "feat_mask"
    ] = "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/MVPA_cross_subjects/masks/feature_mask.nii.gz"

    sub_dict = {}
    # regressor + selector:
    task = "study"
    shift_size_TR = 5
    rest_tag = 0
    label_df = get_shifted_labels(task, shift_size_TR, rest_tag)

    full_data = load_process_data(xsubject_id, task, "MNI", xargs["feat_mask"])
    print(f"Full_data shape: {full_data.shape}")

    X, Y, runs, imgs = sample_for_training(full_data, label_df)

    print("Loading in the saved models...")
    clf, pca = load_model(
        "/scratch/06873/zbretton/tacc_tools/cross_exp_code/trained_models.pkl"
    )

    X_pca = pca.transform(X)
    Y_named = rename_labels(Y, "repclear")

    y_score = clf.decision_function(X_pca)
    n_classes = np.unique(Y_named).size
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        temp_y = np.zeros(Y_named.size)
        label_ind = np.where(Y == (i + 1))
        temp_y[label_ind] = 1

        fpr[i], tpr[i], _ = roc_curve(temp_y, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    pred = clf.predict(X_pca)
    evi = clf.predict_proba(X_pca)
    xscore = accuracy_score(Y_named, pred)

    print(f"ROC_AUC: {roc_auc}")
    print(f"Accuracy Score: {xscore}")
    sub_dict["roc_auc"] = roc_auc
    sub_dict["accuracy score"] = xscore

    # Create a folder named "group_cross_exp_results" if it doesn't exist
    if not os.path.exists(
        os.path.join("/scratch/06873/zbretton", "group_cross_exp_results")
    ):
        os.makedirs(os.path.join("/scratch/06873/zbretton", "group_cross_exp_results"))

    file_path = os.path.join(
        "/scratch/06873/zbretton",
        f"group_cross_exp_results/{subID}_cross_exp_results.json",
    )

    # Write the dictionary to the JSON file
    with open(file_path, "w") as f:
        json.dump(sub_dict, f, indent=4)

    print("Subject data saved...")

# pool subject data:
# define the folder path where the json files are stored
folder_path = "/scratch/06873/zbretton/group_cross_exp_results/"

# define the column names for the output dataframe
col_names = ["maintain", "replace", "suppress"]

# initialize an empty list to store the roc_auc values for each json file
roc_auc_values = []

# loop through each json file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path) as f:
            data = json.load(f)
            roc_auc_values.append(data["roc_auc"])

# create a dictionary from the roc_auc values
roc_auc_dict = {col: [] for col in col_names}

for item in roc_auc_values:
    for i, col in enumerate(col_names):
        roc_auc_dict[col].append(item[str(i)])

# create a dataframe from the roc_auc values
df = pd.DataFrame.from_dict(roc_auc_dict)

# save the dataframe as a csv file
df.to_csv(os.path.join(folder_path, "group_level_cross_exp_rocauc.csv"), index=False)

# now plot the data:

plot_df = df.melt()
plot_df = plot_df.rename(columns={"variable": "Operation", "value": "AUCs"})

plt.style.use("seaborn-paper")

ax = sns.violinplot(
    data=plot_df, x="Operation", y="AUCs", palette="Greys", inner="quartile"
)
sns.swarmplot(data=plot_df, x="Operation", y="AUCs", color="white")
ax.set(xlabel="Operation", ylabel="AUCs")
ax.set_title("Between-Experiment Operation AUCs", loc="center", wrap=True)
ax.axhline(0.5, color="k", linestyle="--")
ax.set_ylim([0.3, 1.1])
plt.tight_layout()

data_dir = "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/"

plt.savefig(os.path.join(data_dir, "figs", "Between-Experiment_AUC_Operation.svg"))
plt.savefig(os.path.join(data_dir, "figs", "Between-Experiment_AUC_Operation.png"))
plt.clf()

# get the stats:
group_m_aucs = df["maintain"]  # takes the first column which is maintain
group_r_aucs = df["replace"]  # takes the second column which is replace
group_s_aucs = df["suppress"]  # takes the third column which is suppress

# one-sample t-test against chance values: AUC=0.5, Scores=0.33

# maintain:
auc_t_stat_m, auc_p_value_m = ttest_1samp(group_m_aucs, 0.5)
print("##### Reporting stats for Maintain AUC #####")
print(f"Mean = {np.array(group_m_aucs).mean()}")
print(f"SEM = {scipy.stats.sem(group_m_aucs)}")
print(f"Degrees of freedom = {(len(group_m_aucs)-1)}")
print(f"T-stat = {auc_t_stat_m}")
print(f"p-value = {auc_p_value_m}")
print(f"cohens d =  {calccohensd(group_m_aucs,0.5)}")
print("")

# replace:
auc_t_stat_r, auc_p_value_r = ttest_1samp(group_r_aucs, 0.5)
print("##### Reporting stats for Replace AUC #####")
print(f"Mean = {np.array(group_r_aucs).mean()}")
print(f"SEM = {scipy.stats.sem(group_r_aucs)}")
print(f"Degrees of freedom = {(len(group_r_aucs)-1)}")
print(f"T-stat = {auc_t_stat_r}")
print(f"p-value = {auc_p_value_r}")
print(f"cohens d =  {calccohensd(group_r_aucs,0.5)}")
print("")

# suppress:
auc_t_stat_s, auc_p_value_s = ttest_1samp(group_s_aucs, 0.5)
print("##### Reporting stats for Suppress AUC #####")
print(f"Mean = {np.array(group_s_aucs).mean()}")
print(f"SEM = {scipy.stats.sem(group_s_aucs)}")
print(f"Degrees of freedom = {(len(group_s_aucs)-1)}")
print(f"T-stat = {auc_t_stat_s}")
print(f"p-value = {auc_p_value_s}")
print(f"cohens d =  {calccohensd(group_s_aucs,0.5)}")
print("")
