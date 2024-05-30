import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

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
task = "study"
space = "T1w"
rois = ["Parahippocampal"]
data_dir = "/Volumes/zbg_eHD/Zachary_Data/processed"


def aggregate_data(subIDs, task, space, rois, data_dir):
    group_data = pd.DataFrame()
    for subID in subIDs:
        for roi in rois:
            file_path = os.path.join(
                data_dir,
                f"sub-{subID}",
                f"sub-{subID}_{space}_{task}_{roi}_evidence_dataframe.csv",
            )
            sub_df = pd.read_csv(file_path)
            sub_df["subID"] = subID  # Add subject ID column
            sub_df["ROI"] = roi  # Add ROI column
            group_data = pd.concat([group_data, sub_df], ignore_index=True)
    return group_data


def calculate_2TR_averages(group_data, rois):
    averages = []
    for roi in rois:
        roi_data = group_data[
            (group_data["ROI"] == roi) & (group_data["evidence_class"] != "rest")
        ]
        for subID in roi_data["subID"].unique():
            sub_data = roi_data[roi_data["subID"] == subID]
            for tr in range(2, 12, 2):
                avg_evidence = (
                    sub_data[(sub_data["TR"] >= tr) & (sub_data["TR"] < tr + 2)]
                    .groupby(["condition"])["evidence"]
                    .mean()
                )
                avg_evidence = avg_evidence.reset_index()
                avg_evidence["TR_block"] = f"TR{tr}-{tr+1}"
                avg_evidence["subID"] = subID
                avg_evidence["ROI"] = roi
                averages.append(avg_evidence)
    averages_df = pd.concat(averages, ignore_index=True)
    return averages_df


def perform_paired_ttests(averages_df, conditions=["maintain", "replace", "suppress"]):
    results = []
    for tr_block in averages_df["TR_block"].unique():
        for roi in averages_df["ROI"].unique():
            tr_data = averages_df[
                (averages_df["TR_block"] == tr_block) & (averages_df["ROI"] == roi)
            ]
            for i, cond1 in enumerate(conditions):
                for cond2 in conditions[i + 1 :]:
                    cond1_data = tr_data[tr_data["condition"] == cond1]["evidence"]
                    cond2_data = tr_data[tr_data["condition"] == cond2]["evidence"]
                    t_stat, p_val = ttest_rel(cond1_data, cond2_data)
                    results.append(
                        {
                            "TR_block": tr_block,
                            "ROI": roi,
                            "Condition_1": cond1,
                            "Condition_2": cond2,
                            "t_stat": t_stat,
                            "p_val": p_val,
                        }
                    )
    results_df = pd.DataFrame(results)
    return results_df


def apply_bonferroni_correction(results_df):
    results_df["p_val_corrected"] = multipletests(
        results_df["p_val"], alpha=0.05, method="bonferroni"
    )[1]
    return results_df


def visualize_evidence(group_data, rois):
    for roi in rois:
        roi_data = group_data[
            (group_data["ROI"] == roi) & (group_data["evidence_class"] == "scenes")
        ]
        mean_rest_evidence = (
            group_data[
                (group_data["ROI"] == roi) & (group_data["evidence_class"] == "rest")
            ]
            .groupby(["TR", "subID"])["evidence"]
            .mean()
            .reset_index()
        )
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=mean_rest_evidence,
            x="TR",
            y="evidence",
            color="gray",
            linestyle="--",
            ci=68,
            label="rest",
        )
        replace_face_data = group_data[
            (group_data["ROI"] == roi)
            & (group_data["condition"] == "replace")
            & (group_data["evidence_class"] == "faces")
        ]
        sns.lineplot(
            data=replace_face_data,
            x="TR",
            y="evidence",
            color="black",
            linestyle="--",
            ci=68,
            label="replace (face evidence)",
        )
        sns.lineplot(
            data=roi_data,
            x="TR",
            y="evidence",
            hue="condition",
            ci=68,
            palette={"maintain": "green", "replace": "blue", "suppress": "red"},
        )
        plt.title(f"Evidence Trajectory in {roi}")
        plt.xlabel("Time (TR)")
        plt.ylabel("Evidence")
        plt.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"evidence_trajectory_{roi}.png", dpi=300)
        plt.show()


group_data = aggregate_data(subIDs, task, space, rois, data_dir)
averages_df = calculate_2TR_averages(group_data, rois)
results_df = perform_paired_ttests(averages_df)
corrected_results_df = apply_bonferroni_correction(results_df)

print(corrected_results_df)

visualize_evidence(group_data, rois)


# =========#

# print(corrected_results_df)
#    TR_block              ROI Condition_1 Condition_2    t_stat     p_val  p_val_corrected
# 0     TR2-3  Parahippocampal    maintain     replace -0.546110  0.590745         1.000000
# 1     TR2-3  Parahippocampal    maintain    suppress -1.494675  0.149875         1.000000
# 2     TR2-3  Parahippocampal     replace    suppress -1.217048  0.237087         1.000000
# 3     TR4-5  Parahippocampal    maintain     replace -0.424282  0.675675         1.000000
# 4     TR4-5  Parahippocampal    maintain    suppress -0.087369  0.931206         1.000000
# 5     TR4-5  Parahippocampal     replace    suppress  0.362961  0.720262         1.000000
# 6     TR6-7  Parahippocampal    maintain     replace  1.891649  0.072410         1.000000
# 7     TR6-7  Parahippocampal    maintain    suppress  1.836385  0.080499         1.000000
# 8     TR6-7  Parahippocampal     replace    suppress  0.131537  0.896603         1.000000
# 9     TR8-9  Parahippocampal    maintain     replace  4.486297  0.000203         0.003047
# 10    TR8-9  Parahippocampal    maintain    suppress  2.653449  0.014864         0.222954
# 11    TR8-9  Parahippocampal     replace    suppress -1.820033  0.083039         1.000000
# 12  TR10-11  Parahippocampal    maintain     replace  5.163866  0.000041         0.000611
# 13  TR10-11  Parahippocampal    maintain    suppress  2.454336  0.022925         0.343879
# 14  TR10-11  Parahippocampal     replace    suppress -2.020127  0.056309         0.844642


# ============#

# version with normalization to maintain brought back:

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, ttest_rel
from statsmodels.stats.multitest import multipletests

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
task = "study"
space = "T1w"
rois = ["Parahippocampal"]
data_dir = "/Volumes/zbg_eHD/Zachary_Data/processed"


def aggregate_data(subIDs, task, space, rois, data_dir):
    group_data = pd.DataFrame()
    for subID in subIDs:
        for roi in rois:
            file_path = os.path.join(
                data_dir,
                f"sub-{subID}",
                f"sub-{subID}_{space}_{task}_{roi}_evidence_dataframe.csv",
            )
            sub_df = pd.read_csv(file_path)
            sub_df["subID"] = subID  # Add subject ID column
            sub_df["ROI"] = roi  # Add ROI column
            group_data = pd.concat([group_data, sub_df], ignore_index=True)
    return group_data


def normalize_data(group_data):
    normalized_data = group_data.copy()
    for subID in group_data["subID"].unique():
        sub_data = group_data[group_data["subID"] == subID]
        maintain_data = sub_data[sub_data["condition"] == "maintain"]
        for tr in sub_data["TR"].unique():
            maintain_evidence = maintain_data[maintain_data["TR"] == tr][
                "evidence"
            ].values
            for condition in ["replace", "suppress"]:
                condition_data = sub_data[
                    (sub_data["condition"] == condition) & (sub_data["TR"] == tr)
                ]
                normalized_evidence = (
                    condition_data["evidence"].values - maintain_evidence
                )
                normalized_data.loc[
                    (normalized_data["subID"] == subID)
                    & (normalized_data["condition"] == condition)
                    & (normalized_data["TR"] == tr),
                    "normalized_evidence",
                ] = normalized_evidence
    return normalized_data


def calculate_2TR_averages(normalized_data, rois):
    averages = []
    for roi in rois:
        roi_data = normalized_data[
            (normalized_data["ROI"] == roi)
            & (normalized_data["evidence_class"] != "rest")
        ]
        for subID in roi_data["subID"].unique():
            sub_data = roi_data[roi_data["subID"] == subID]
            for tr in range(2, 12, 2):
                avg_evidence = (
                    sub_data[(sub_data["TR"] >= tr) & (sub_data["TR"] < tr + 2)]
                    .groupby(["condition"])["normalized_evidence"]
                    .mean()
                )
                avg_evidence = avg_evidence.reset_index()
                avg_evidence["TR_block"] = f"TR{tr}-{tr+1}"
                avg_evidence["subID"] = subID
                avg_evidence["ROI"] = roi
                averages.append(avg_evidence)
    averages_df = pd.concat(averages, ignore_index=True)
    return averages_df


def perform_stat_tests(averages_df, conditions=["replace", "suppress"]):
    results = []
    for tr_block in averages_df["TR_block"].unique():
        for roi in averages_df["ROI"].unique():
            tr_data = averages_df[
                (averages_df["TR_block"] == tr_block) & (averages_df["ROI"] == roi)
            ]
            for condition in conditions:
                cond_data = tr_data[tr_data["condition"] == condition][
                    "normalized_evidence"
                ]
                t_stat, p_val = ttest_1samp(cond_data, 0)
                results.append(
                    {
                        "TR_block": tr_block,
                        "ROI": roi,
                        "Condition": condition,
                        "Test": "one-sample t-test vs. 0",
                        "t_stat": t_stat,
                        "p_val": p_val,
                    }
                )
            # Paired t-test between replace and suppress
            replace_data = tr_data[tr_data["condition"] == "replace"][
                "normalized_evidence"
            ]
            suppress_data = tr_data[tr_data["condition"] == "suppress"][
                "normalized_evidence"
            ]
            t_stat, p_val = ttest_rel(replace_data, suppress_data)
            results.append(
                {
                    "TR_block": tr_block,
                    "ROI": roi,
                    "Condition": "replace vs. suppress",
                    "Test": "paired t-test",
                    "t_stat": t_stat,
                    "p_val": p_val,
                }
            )
    results_df = pd.DataFrame(results)
    return results_df


def apply_bonferroni_correction(results_df):
    # Apply Bonferroni correction
    results_df["p_val_corrected"] = multipletests(
        results_df["p_val"], alpha=0.05, method="bonferroni"
    )[1]
    return results_df


def visualize_normalized_evidence(normalized_data, rois):
    for roi in rois:
        roi_data = normalized_data[
            (normalized_data["ROI"] == roi)
            & (normalized_data["evidence_class"] != "rest")
        ]
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=roi_data[roi_data["condition"].isin(["replace", "suppress"])],
            x="TR",
            y="normalized_evidence",
            hue="condition",
            errorbar="ci",
            palette={"replace": "blue", "suppress": "red"},
        )
        plt.axhline(0, color="gray", linestyle="--")  # Add a horizontal line at y=0
        plt.title(f"Normalized Evidence Trajectory in {roi}")
        plt.xlabel("Time (TR)")
        plt.ylabel("Normalized Evidence")
        plt.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"normalized_evidence_trajectory_{roi}.png", dpi=300)
        plt.show()


group_data = aggregate_data(subIDs, task, space, rois, data_dir)
normalized_data = normalize_data(group_data)
averages_df = calculate_2TR_averages(normalized_data, rois)
results_df = perform_stat_tests(averages_df)
corrected_results_df = apply_bonferroni_correction(results_df)

print(corrected_results_df)

visualize_normalized_evidence(normalized_data, rois)


# ===========#

#    TR_block              ROI             Condition                     Test    t_stat     p_val  p_val_corrected
# 0     TR2-3  Parahippocampal               replace  one-sample t-test vs. 0  0.546110  0.590745         1.000000
# 1     TR2-3  Parahippocampal              suppress  one-sample t-test vs. 0  1.494675  0.149875         1.000000
# 2     TR2-3  Parahippocampal  replace vs. suppress            paired t-test -1.217048  0.237087         1.000000
# 3     TR4-5  Parahippocampal               replace  one-sample t-test vs. 0  0.424282  0.675675         1.000000
# 4     TR4-5  Parahippocampal              suppress  one-sample t-test vs. 0  0.087369  0.931206         1.000000
# 5     TR4-5  Parahippocampal  replace vs. suppress            paired t-test  0.362961  0.720262         1.000000
# 6     TR6-7  Parahippocampal               replace  one-sample t-test vs. 0 -1.891649  0.072410         1.000000
# 7     TR6-7  Parahippocampal              suppress  one-sample t-test vs. 0 -1.836385  0.080499         1.000000
# 8     TR6-7  Parahippocampal  replace vs. suppress            paired t-test  0.131537  0.896603         1.000000
# 9     TR8-9  Parahippocampal               replace  one-sample t-test vs. 0 -4.486297  0.000203         0.003047
# 10    TR8-9  Parahippocampal              suppress  one-sample t-test vs. 0 -2.653449  0.014864         0.222954
# 11    TR8-9  Parahippocampal  replace vs. suppress            paired t-test -1.820033  0.083039         1.000000
# 12  TR10-11  Parahippocampal               replace  one-sample t-test vs. 0 -5.163866  0.000041         0.000611
# 13  TR10-11  Parahippocampal              suppress  one-sample t-test vs. 0 -2.454336  0.022925         0.343879
# 14  TR10-11  Parahippocampal  replace vs. suppress            paired t-test -2.020127  0.056309         0.844642
