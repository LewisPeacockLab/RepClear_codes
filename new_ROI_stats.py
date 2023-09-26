import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from sklearn.utils import resample


def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    return df.dropna()


def bootstrap_paired_ttest_matched(data1, data2, n_iterations=10000):
    t_stats = []
    p_values = []

    common_subjects = list(set(data1.index) & set(data2.index))

    for i in range(n_iterations):
        sample1 = resample(data1.loc[common_subjects])
        sample2 = resample(data2.loc[common_subjects])

        t_stat, p_value = stats.ttest_rel(sample1, sample2)
        t_stats.append(t_stat)
        p_values.append(p_value)

    return t_stats, p_values


def perform_anova_and_posthoc(data, dependent_var, subject_var, within_vars):
    anova = AnovaRM(
        data=data, depvar=dependent_var, subject=subject_var, within=within_vars
    )
    anova_fit = anova.fit()
    return anova_fit.summary()


def comprehensive_analysis(input_files, output_folder):
    data_frames = {
        os.path.splitext(os.path.basename(file))[0]: load_and_clean_data(file)
        for file in input_files
    }

    data_frames_cleaned = {}
    for key, df in data_frames.items():
        df = df.set_index("Subject")
        data_frames_cleaned[key] = df.dropna()

    operation_list = data_frames_cleaned[list(data_frames_cleaned.keys())[0]][
        "Operation"
    ].unique()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    summaries = []
    for i in range(0, len(input_files), 2):
        key1, key2 = (
            list(data_frames_cleaned.keys())[i],
            list(data_frames_cleaned.keys())[i + 1],
        )

        fig = plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        sns.boxplot(
            x="Operation",
            y="Fidelity",
            data=data_frames_cleaned[key1].reset_index(),
            palette="pastel",
        )
        plt.title(key1)
        plt.subplot(1, 2, 2)
        sns.boxplot(
            x="Operation",
            y="Fidelity",
            data=data_frames_cleaned[key2].reset_index(),
            palette="pastel",
        )
        plt.title(key2)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{key1}_vs_{key2}_boxplots.png"))

        ttest_summary = f"Paired t-test results for {key1} vs {key2}:\n\n"
        anova_summary = f"ANOVA results for {key1} and {key2}:\n\n"
        bootstrap_summary = f"Bootstrap Paired t-test results for {key1} vs {key2}:\n\n"

        for operation in operation_list:
            data1 = data_frames_cleaned[key1][
                data_frames_cleaned[key1]["Operation"] == operation
            ]["Fidelity"]
            data2 = data_frames_cleaned[key2][
                data_frames_cleaned[key2]["Operation"] == operation
            ]["Fidelity"]
            common_subjects = list(set(data1.index) & set(data2.index))
            data1 = data1.loc[common_subjects]
            data2 = data2.loc[common_subjects]

            t_stat, p_value = stats.ttest_rel(data1, data2)
            ttest_summary += f"{operation}: t-statistic = {t_stat:.3f}, p-value = {p_value:.4f}\n"  # Using \n for new lines
            # Bootstrap paired t-test
            t_stats, p_values = bootstrap_paired_ttest_matched(data1, data2)
            avg_t_stat = np.mean(t_stats)
            avg_p_value = np.mean(p_values)
            bootstrap_summary += f"{operation}: Avg t-statistic = {avg_t_stat:.3f}, Avg p-value = {avg_p_value:.4f}\n"

            # ANOVA (assuming the DataFrame is structured properly for this)
            anova_result = perform_anova_and_posthoc(
                data=pd.concat([data1, data2]),
                dependent_var="Fidelity",
                subject_var="Subject",
                within_vars=["Operation"],
            )
            anova_summary += f"{operation}: {anova_result}\n"

        summaries.append(ttest_summary)
        summaries.append(bootstrap_summary)
        summaries.append(anova_summary)

    with open(os.path.join(output_folder, "all_summaries.txt"), "w") as f:
        for i in range(len(summaries) // 3):
            f.write(f"=== Standard t-test for {summaries[i*3].split(':')[0]} ===\n")
            f.write(summaries[i * 3])
            f.write("\n\n")

            f.write(f"=== Bootstrap t-test for {summaries[i*3].split(':')[0]} ===\n")
            f.write(summaries[i * 3 + 1])
            f.write("\n\n")

            f.write(f"=== ANOVA for {summaries[i*3].split(':')[0]} ===\n")
            f.write(summaries[i * 3 + 2])
            f.write("\n\n")

    print(f"Analysis completed. Results have been saved in {output_folder}.")


for roi in ["Prefrontal_ROI", "Higher_Order_Visual_ROI"]:
    comprehensive_analysis(
        [
            f"/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/{roi}_group_level_data/paired_group_itemweighted_remembered_fidelity_{roi}.csv",
            f"/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/{roi}_group_level_data/paired_group_itemweighted_forgot_fidelity_{roi}.csv",
            f"/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/{roi}_group_level_data/paired_group_cateweighted_remembered_fidelity_{roi}.csv",
            f"/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/{roi}_group_level_data/paired_group_cateweighted_forgot_fidelity_{roi}.csv",
        ],
        f"/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/{roi}_group_level_data/new_ROI_stats",
    )
