import warnings
import sys

if not sys.warnoptions:
    warnings.simplefilter("ignore")
import numpy as np
import nibabel as nib
import scipy as scipy
import scipy.io as sio
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import fnmatch
import pandas as pd
import pickle

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


mask_flag = "wholebrain"  #'vtc'/'wholebrain'/'GM'/'GM_group'/'PHG'/'FG'
brain_flag = "T1w"
TR_shift = 5
ses = "study"  # study/localizer/btwnsub
clear_data = 1
rest = "off"
subcat = "off"

if ses == "study":
    ses_label = "operation"
elif ses == "localizer":
    ses_label = "category"
elif ses == "btwnsub":
    ses_label = "operation"


def group_cmatrix(subs):
    group_mean_confusion = []
    oper_confusion_mean = []
    if ses == "btwnsub":
        print("Running...")
        # define the path
        container_path = (
            "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep"
        )
        bold_path = container_path
        os.chdir(bold_path)

        sub_dict = pickle.load(
            open(
                "btwnsuboperation_%s_%s_%sTR lag_data_cleaned.pkl"
                % (brain_flag, mask_flag, TR_shift),
                "rb",
            )
        )
        sub_dict_pred = sub_dict["L2 Predictions (No Rest)"]
        sub_dict_true = sub_dict["L2 True (No Rest)"]
        sub_confusion_1 = confusion_matrix(
            sub_dict_true[0], sub_dict_pred[0], normalize="true"
        )
        sub_confusion_2 = confusion_matrix(
            sub_dict_true[1], sub_dict_pred[1], normalize="true"
        )
        sub_confusion_3 = confusion_matrix(
            sub_dict_true[2], sub_dict_pred[2], normalize="true"
        )
        sub_confusion_mean = np.array(
            [sub_confusion_1, sub_confusion_2, sub_confusion_3]
        )
        group_mean_confusion = sub_confusion_mean
    else:
        for num in subs:
            print("Running sub-0%s..." % num)
            # define the subject
            sub = "sub-0%s" % num
            container_path = (
                "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep"
            )

            bold_path = os.path.join(container_path, sub)
            os.chdir(bold_path)

            if ses == "study":
                if clear_data == 0:
                    sub_dict = pickle.load(
                        open(
                            "%s-studyoperation_%s_%s_%sTR lag_data.pkl"
                            % (sub, brain_flag, mask_flag, TR_shift),
                            "rb",
                        )
                    )
                    sub_dict_pred = sub_dict["L2 Predictions (No Rest)"]
                    sub_dict_true = sub_dict["L2 True (No Rest)"]
                    sub_confusion_1 = confusion_matrix(
                        sub_dict_true[0], sub_dict_pred[0], normalize="true"
                    )
                    sub_confusion_2 = confusion_matrix(
                        sub_dict_true[1], sub_dict_pred[1], normalize="true"
                    )
                    sub_confusion_3 = confusion_matrix(
                        sub_dict_true[2], sub_dict_pred[2], normalize="true"
                    )
                    sub_confusion_mean = np.mean(
                        np.array([sub_confusion_1, sub_confusion_2, sub_confusion_3]),
                        axis=0,
                    )
                else:
                    sub_dict = pickle.load(
                        open(
                            "%s-studyoperation_%s_%s_%sTR lag_data_cleaned.pkl"
                            % (sub, brain_flag, mask_flag, TR_shift),
                            "rb",
                        )
                    )
                    sub_dict_pred = sub_dict["L2 Predictions (No Rest)"]
                    sub_dict_true = sub_dict["L2 True (No Rest)"]
                    sub_confusion_1 = confusion_matrix(
                        sub_dict_true[0], sub_dict_pred[0], normalize="true"
                    )
                    sub_confusion_2 = confusion_matrix(
                        sub_dict_true[1], sub_dict_pred[1], normalize="true"
                    )
                    sub_confusion_3 = confusion_matrix(
                        sub_dict_true[2], sub_dict_pred[2], normalize="true"
                    )
                    sub_confusion_mean = np.mean(
                        np.array([sub_confusion_1, sub_confusion_2, sub_confusion_3]),
                        axis=0,
                    )
            elif (ses == "localizer") & (subcat == "on"):
                if rest == "off":
                    if clear_data == 0:
                        sub_dict = pickle.load(
                            open(
                                "%s-preremoval_subcategory_rest_%s_%s_%s_%sTR lag_data.pkl"
                                % (sub, rest, brain_flag, mask_flag, TR_shift),
                                "rb",
                            )
                        )
                        sub_dict_pred = sub_dict["L2 Predictions (only On)"]
                        sub_dict_true = sub_dict["L2 True (only On)"]
                        sub_confusion_1 = confusion_matrix(
                            sub_dict_true[0], sub_dict_pred[0], normalize="true"
                        )
                        sub_confusion_2 = confusion_matrix(
                            sub_dict_true[1], sub_dict_pred[1], normalize="true"
                        )
                        sub_confusion_mean = np.mean(
                            np.array([sub_confusion_1, sub_confusion_2]), axis=0
                        )
                    else:
                        sub_dict = pickle.load(
                            open(
                                "%s-preremoval_subcategory_rest_%s_%s_%s_%sTR lag_data_cleaned.pkl"
                                % (sub, rest, brain_flag, mask_flag, TR_shift),
                                "rb",
                            )
                        )
                        sub_dict_pred = sub_dict["L2 Predictions (only On)"]
                        sub_dict_true = sub_dict["L2 True (only On)"]
                        sub_confusion_1 = confusion_matrix(
                            sub_dict_true[0], sub_dict_pred[0], normalize="true"
                        )
                        sub_confusion_2 = confusion_matrix(
                            sub_dict_true[1], sub_dict_pred[1], normalize="true"
                        )
                        sub_confusion_mean = np.mean(
                            np.array([sub_confusion_1, sub_confusion_2]), axis=0
                        )
                elif rest == "on":
                    if clear_data == 0:
                        sub_dict = pickle.load(
                            open(
                                "%s-preremoval_subcategory_rest_%s_%s_%s_%sTR lag_data.pkl"
                                % (sub, rest, brain_flag, mask_flag, TR_shift),
                                "rb",
                            )
                        )
                        sub_dict_pred = sub_dict["L2 Predictions (With Rest)"]
                        sub_dict_true = sub_dict["L2 True (With Rest)"]
                        sub_confusion_1 = confusion_matrix(
                            sub_dict_true[0], sub_dict_pred[0], normalize="true"
                        )
                        sub_confusion_2 = confusion_matrix(
                            sub_dict_true[1], sub_dict_pred[1], normalize="true"
                        )
                        sub_confusion_mean = np.mean(
                            np.array([sub_confusion_1, sub_confusion_2]), axis=0
                        )
                    else:
                        sub_dict = pickle.load(
                            open(
                                "%s-preremoval_subcategory_rest_%s_%s_%s_%sTR lag_data_cleaned.pkl"
                                % (sub, rest, brain_flag, mask_flag, TR_shift),
                                "rb",
                            )
                        )
                        sub_dict_pred = sub_dict["L2 Predictions (With Rest)"]
                        sub_dict_true = sub_dict["L2 True (With Rest)"]
                        sub_confusion_1 = confusion_matrix(
                            sub_dict_true[0], sub_dict_pred[0], normalize="true"
                        )
                        sub_confusion_2 = confusion_matrix(
                            sub_dict_true[1], sub_dict_pred[1], normalize="true"
                        )
                        sub_confusion_mean = np.mean(
                            np.array([sub_confusion_1, sub_confusion_2]), axis=0
                        )

            elif ses == "localizer":
                if rest == "off":
                    if clear_data == 0:
                        sub_dict = pickle.load(
                            open(
                                "%s-preremoval_rest_%s_%s_%s_%sTR lag_data.pkl"
                                % (sub, rest, brain_flag, mask_flag, TR_shift),
                                "rb",
                            )
                        )
                        sub_dict_pred = sub_dict["L2 Predictions (only On)"]
                        sub_dict_true = sub_dict["L2 True (only On)"]
                        sub_confusion_1 = confusion_matrix(
                            sub_dict_true[0], sub_dict_pred[0], normalize="true"
                        )
                        sub_confusion_2 = confusion_matrix(
                            sub_dict_true[1], sub_dict_pred[1], normalize="true"
                        )
                        sub_confusion_mean = np.mean(
                            np.array([sub_confusion_1, sub_confusion_2]), axis=0
                        )
                    else:
                        sub_dict = pickle.load(
                            open(
                                "%s-preremoval_rest_%s_%s_%s_%sTR lag_data_cleaned.pkl"
                                % (sub, rest, brain_flag, mask_flag, TR_shift),
                                "rb",
                            )
                        )
                        sub_dict_pred = sub_dict["L2 Predictions (only On)"]
                        sub_dict_true = sub_dict["L2 True (only On)"]
                        sub_confusion_1 = confusion_matrix(
                            sub_dict_true[0], sub_dict_pred[0], normalize="true"
                        )
                        sub_confusion_2 = confusion_matrix(
                            sub_dict_true[1], sub_dict_pred[1], normalize="true"
                        )
                        sub_confusion_mean = np.mean(
                            np.array([sub_confusion_1, sub_confusion_2]), axis=0
                        )
                elif rest == "on":
                    if clear_data == 0:
                        sub_dict = pickle.load(
                            open(
                                "%s-preremoval_rest_%s_%s_%s_%sTR lag_data.pkl"
                                % (sub, rest, brain_flag, mask_flag, TR_shift),
                                "rb",
                            )
                        )
                        sub_dict_pred = sub_dict["L2 Predictions (With Rest)"]
                        sub_dict_true = sub_dict["L2 True (With Rest)"]
                        sub_confusion_1 = confusion_matrix(
                            sub_dict_true[0], sub_dict_pred[0], normalize="true"
                        )
                        sub_confusion_2 = confusion_matrix(
                            sub_dict_true[1], sub_dict_pred[1], normalize="true"
                        )
                        sub_confusion_mean = np.mean(
                            np.array([sub_confusion_1, sub_confusion_2]), axis=0
                        )
                    else:
                        sub_dict = pickle.load(
                            open(
                                "%s-preremoval_rest_%s_%s_%s_%sTR lag_data_cleaned.pkl"
                                % (sub, rest, brain_flag, mask_flag, TR_shift),
                                "rb",
                            )
                        )
                        sub_dict_pred = sub_dict["L2 Predictions (With Rest)"]
                        sub_dict_true = sub_dict["L2 True (With Rest)"]
                        sub_confusion_1 = confusion_matrix(
                            sub_dict_true[0], sub_dict_pred[0], normalize="true"
                        )
                        sub_confusion_2 = confusion_matrix(
                            sub_dict_true[1], sub_dict_pred[1], normalize="true"
                        )
                        sub_confusion_mean = np.mean(
                            np.array([sub_confusion_1, sub_confusion_2]), axis=0
                        )

            group_mean_confusion.append(sub_confusion_mean)
    return group_mean_confusion


if rest == "off":
    labels = ["scenes", "faces"]
    oper_labels = ["Maintain", "Replace", "Suppress"]
elif rest == "on":
    if subcat == "on":
        labels = ["rest", "manmade", "natural", "female", "male"]
        oper_labels = ["rest", "maintain", "replace", "suppress"]

    else:
        labels = ["rest", "scenes", "faces"]
        oper_labels = ["rest", "maintain", "replace", "suppress"]

fig = plt.figure()
group_mean_confusion = group_cmatrix(subs)
plot_confusion = np.mean(group_mean_confusion, axis=0)
plt.style.use("fivethirtyeight")
ax = sns.heatmap(
    data=(plot_confusion * 100), annot=True, cmap="viridis", vmin=10, vmax=90, center=33
)
if ses_label == "category":
    ax.set(
        xlabel="Predicted",
        ylabel="True",
        xticklabels=labels,
        yticklabels=labels,
        title="Group Level - Category Classifier",
    )
elif ses_label == "operation":
    ax.set(
        xlabel="Predicted",
        ylabel="True",
        xticklabels=oper_labels,
        yticklabels=oper_labels,
        title="Group Level - Operation Classifier",
    )
ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.tight_layout()
os.chdir("/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/figs")
if clear_data == 0:
    if subcat == "on":
        fig.savefig(
            "%s_xvalidation_subcategory_rest_%s_%sclassifier_%s_TR%s_%s.svg"
            % (ses, rest, ses_label, brain_flag, TR_shift, mask_flag),
            dpi=fig.dpi,
        )
    else:
        fig.savefig(
            "%s_xvalidation_rest_%s_%sclassifier_%s_TR%s_%s.svg"
            % (ses, rest, ses_label, brain_flag, TR_shift, mask_flag),
            dpi=fig.dpi,
        )
else:
    if subcat == "on":
        fig.savefig(
            "%s_xvalidation_subcategory_rest_%s_%sclassifier_%s_TR%s_%s_cleaned.svg"
            % (ses, rest, ses_label, brain_flag, TR_shift, mask_flag),
            dpi=fig.dpi,
        )
    else:
        fig.savefig(
            "%s_xvalidation_rest_%s_%sclassifier_%s_TR%s_%s_cleaned.svg"
            % (ses, rest, ses_label, brain_flag, TR_shift, mask_flag),
            dpi=fig.dpi,
        )
plt.show()
