# code to calculate d' and prep data for P-CIT toolbox for NMPH
import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import scipy as scipy
from scipy import stats
import scipy.io as sio
import fnmatch
import seaborn as sns
from statannot import add_stat_annotation

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
    "023",
    "024",
    "025",
]

workspace = "scratch"
data_dir = "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/"
param_dir = (
    "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/subject_designs"
)
figs_dir = "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/figs"

group_maintain_d_prime = []
group_replace_d_prime = []
group_suppress_d_prime = []
group_pre_exp_d_prime = []

group_maintain_a_prime = []
group_replace_a_prime = []
group_suppress_a_prime = []
group_pre_exp_a_prime = []

for subID in subIDs:
    memory_df = pd.read_csv(
        os.path.join(param_dir, "memory_and_familiar_sub-%s.csv" % subID)
    )

    maintain_hits = memory_df[
        ((memory_df["condition_num"] == 1) & (memory_df["resp"] >= 3))
    ]  # this is pulling all the index with correct answers
    replace_hits = memory_df[
        ((memory_df["condition_num"] == 2) & (memory_df["resp"] >= 3))
    ]  # this is pulling all the index with correct answers
    suppress_hits = memory_df[
        ((memory_df["condition_num"] == 3) & (memory_df["resp"] >= 3))
    ]  # this is pulling all the index with correct answers
    pre_exp_hits = memory_df[((memory_df["old_novel"] == 3) & (memory_df["resp"] >= 3))]
    novel_FA = memory_df[((memory_df["old_novel"] == 2) & (memory_df["resp"] >= 3))]

    m_hit_rate = len(maintain_hits) / 30
    r_hit_rate = len(replace_hits) / 30
    s_hit_rate = len(suppress_hits) / 30
    pe_hit_rate = len(pre_exp_hits) / 30
    fa_rate = len(novel_FA) / 60

    maintain_hit_zscore = stats.norm.ppf(
        m_hit_rate
    )  # this calculates the z-score based on an inverse cumulative distribution function (hits / total)
    replace_hit_zscore = stats.norm.ppf(
        r_hit_rate
    )  # this calculates the z-score based on an inverse cumulative distribution function (hits / total)
    suppress_hit_zscore = stats.norm.ppf(
        s_hit_rate
    )  # this calculates the z-score based on an inverse cumulative distribution function (hits / total)
    pre_exp_hit_zscore = stats.norm.ppf(
        pe_hit_rate
    )  # this calculates the z-score based on an inverse cumulative distribution function (hits / total)
    novel_fa_zscore = stats.norm.ppf(
        fa_rate
    )  # this is over 60 since there are 60 total novel images

    sub_maintain_d_prime = maintain_hit_zscore - novel_fa_zscore
    sub_replace_d_prime = replace_hit_zscore - novel_fa_zscore
    sub_suppress_d_prime = suppress_hit_zscore - novel_fa_zscore
    sub_pre_exp_d_prime = pre_exp_hit_zscore - novel_fa_zscore

    def a_prime(hit_rate, false_alarm_rate):
        a_prime_value = 1 - (
            (1 / 4)
            * (
                (false_alarm_rate / hit_rate)
                + ((1 - hit_rate) / (1 - false_alarm_rate))
            )
        )
        return a_prime_value

    sub_maintain_a_prime = a_prime(m_hit_rate, fa_rate)
    sub_replace_a_prime = a_prime(r_hit_rate, fa_rate)
    sub_suppress_a_prime = a_prime(s_hit_rate, fa_rate)
    sub_pre_exp_a_prime = a_prime(pe_hit_rate, fa_rate)

    group_maintain_a_prime = np.append(group_maintain_a_prime, sub_maintain_a_prime)
    group_replace_a_prime = np.append(group_replace_a_prime, sub_replace_a_prime)
    group_suppress_a_prime = np.append(group_suppress_a_prime, sub_suppress_a_prime)
    group_pre_exp_a_prime = np.append(group_pre_exp_a_prime, sub_pre_exp_a_prime)

    group_maintain_d_prime = np.append(group_maintain_d_prime, sub_maintain_d_prime)
    group_replace_d_prime = np.append(group_replace_d_prime, sub_replace_d_prime)
    group_suppress_d_prime = np.append(group_suppress_d_prime, sub_suppress_d_prime)
    group_pre_exp_d_prime = np.append(group_pre_exp_d_prime, sub_pre_exp_d_prime)


d_prime_df = pd.DataFrame(columns=["maintain", "replace", "suppress", "pre exposed"])
d_prime_df["maintain"] = group_maintain_d_prime
d_prime_df["replace"] = group_replace_d_prime
d_prime_df["suppress"] = group_suppress_d_prime
d_prime_df["pre exposed"] = group_pre_exp_d_prime

d_prime_df = d_prime_df[
    np.isfinite(d_prime_df).all(axis=1)
]  # because of some 0's and 1's, we get some 'inf' we need to remove, sadly that means removing that whole row

ax = sns.barplot(data=d_prime_df, palette=["green", "blue", "red", "gray"])
ax.set_ylabel("d' Memory")
ax.set_xlabel("Operation")
ax.set_title("Memory by Operation (Group-level)")
for i in ax.containers:
    ax.bar_label(i, size=12, label_type="center")
ax, test_results = add_stat_annotation(
    ax,
    data=d_prime_df,
    box_pairs=[
        ("maintain", "replace"),
        ("maintain", "suppress"),
        ("maintain", "pre exposed"),
        ("suppress", "pre exposed"),
        ("replace", "pre exposed"),
    ],
    test="t-test_paired",
    text_format="star",
    loc="inside",
    verbose=2,
)
plt.savefig(os.path.join(figs_dir, "d_prime_group_memory.png"))
plt.clf()

a_prime_df = pd.DataFrame(columns=["maintain", "replace", "suppress", "pre exposed"])
a_prime_df["maintain"] = group_maintain_a_prime
a_prime_df["replace"] = group_replace_a_prime
a_prime_df["suppress"] = group_suppress_a_prime
a_prime_df["pre exposed"] = group_pre_exp_a_prime

ax = sns.barplot(data=a_prime_df, palette=["green", "blue", "red", "gray"])
ax.set_ylabel("A' Memory")
ax.set_xlabel("Operation")
ax.set_title("Memory by Operation (Group-level)")
for i in ax.containers:
    ax.bar_label(i, size=12, label_type="center")
ax, test_results = add_stat_annotation(
    ax,
    data=a_prime_df,
    box_pairs=[
        ("maintain", "replace"),
        ("maintain", "suppress"),
        ("maintain", "pre exposed"),
        ("suppress", "pre exposed"),
        ("replace", "pre exposed"),
    ],
    test="t-test_paired",
    text_format="star",
    loc="inside",
    verbose=2,
)
plt.savefig(os.path.join(figs_dir, "a_prime_group_memory.png"))
plt.clf()
