import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os
import pingouin as pg
from matplotlib.legend_handler import HandlerTuple
from sklearn.utils import resample
import numpy as np

rois = ["Prefrontal_ROI", "Higher_Order_Visual_ROI", "hippocampus_ROI", "VTC_mask"]


def load_data(file_path):
    """
    Load a CSV file into a DataFrame.
    """
    return pd.read_csv(file_path)


def aggregate_data(df, memory_status):
    """
    Aggregate the data by operation and calculate the mean fidelity.
    Adds a 'Memory' column to indicate if the data is from remembered or forgotten instances.
    """
    df = df[df["Fidelity"].notna()]
    agg_df = df.groupby("Operation")["Fidelity"].mean().reset_index()
    agg_df["Memory"] = memory_status
    return agg_df


def perform_t_tests(agg_df1, agg_df2, roi, test_type="paired", n_iterations=1000):
    operations = agg_df1["Operation"].unique()
    stats_results = []
    for op in operations:
        data1 = agg_df1.loc[agg_df1["Operation"] == op, "Fidelity"]
        data2 = agg_df2.loc[agg_df2["Operation"] == op, "Fidelity"]
        DoF = len(data1) - 1 if test_type == "paired" else len(data1) + len(data2) - 2

        # Bootstrap
        bootstrap_t_stats = []
        for i in range(n_iterations):
            sample1 = resample(data1, replace=True)
            sample2 = resample(data2, replace=True)
            if test_type == "paired":
                t_stat, _ = stats.ttest_rel(sample1, sample2, nan_policy="omit")
            else:
                t_stat, _ = stats.ttest_ind(sample1, sample2, nan_policy="omit")
            bootstrap_t_stats.append(t_stat)

        # Calculate 95% CI for t-statistic
        lower = np.percentile(bootstrap_t_stats, 2.5)
        upper = np.percentile(bootstrap_t_stats, 97.5)

        # Existing t-test and Bayes Factor calculation
        if test_type == "paired":
            t_stat, p_val = stats.ttest_rel(data1, data2, nan_policy="omit")
            bayes_result = pg.ttest(data1, data2, paired=True)
        else:
            t_stat, p_val = stats.ttest_ind(data1, data2, nan_policy="omit")
            bayes_result = pg.ttest(data1, data2, paired=False)

        if bayes_result.shape[0] > 0:
            bayes_factor = bayes_result.iloc[0]["BF10"]
        else:
            bayes_factor = "DataFrame is empty"

        print(
            f"{test_type.capitalize()} t-test for operation {op}: t = {t_stat}, p = {p_val}, BF10 = {bayes_factor}, DoF = {DoF}, Bootstrap 95% CI = ({lower}, {upper})"
        )
        stats_results.append({
            'Operation': op,
            'TestType': test_type,
            't_stat': t_stat,
            'p_val': p_val,
            'BF10': bayes_factor,
            'DoF': DoF,
            'Bootstrap_CI_lower': lower,
            'Bootstrap_CI_upper': upper
        })

    # Save to text file
    stats_df = pd.DataFrame(stats_results)

    save_path = f"/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/ROI_{roi}_{test_type}_t_test_stats.txt"
    if os.path.exists(save_path):
        write_mode = 'a'  # append if already exists
    else:
        write_mode = 'w'  # make a new file if not

    with open(save_path, write_mode) as file:
        # Write the statistics to the file
        file.write(stats_df.to_csv(index=False, sep='\t'))
        
        # Add a separator for readability
        file.write("--------------------------------------------------\n")


def plot_data(combined_df, plot_title, save_path):
    """
    Generate a Seaborn barplot for the data.
    """
    fig = sns.barplot(
        data=combined_df,
        x="Operation",
        y="Fidelity",
        hue="Memory",
        ci=95,
        palette={"remembered": "black", "forgot": "grey"},
        edgecolor=".7",
    )
    for bar_group, desaturate_value in zip(fig.containers, [0.4, 1]):
        for bar, color in zip(bar_group, ["green", "blue", "red"]):
            bar.set_facecolor(sns.desaturate(color, desaturate_value))

    fig.set_title(plot_title, loc="center", wrap=True)
    fig.set_xlabel("Operations")
    fig.set_ylabel("Fidelity of item-RSA")
    plt.tight_layout()

    fig.legend(
        handles=[tuple(bar_group) for bar_group in fig.containers],
        labels=[bar_group.get_label() for bar_group in fig.containers],
        title=fig.legend_.get_title().get_text(),
        handlelength=4,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.1)},
    )

    plt.savefig(save_path)
    plt.clf()


def check_for_nans(df, df_name, subID):
    nan_columns = df.columns[df.isna().any()].tolist()
    if nan_columns:
        print(
            f"Subject {subID}, DataFrame {df_name} contains NaN values in columns: {nan_columns}"
        )
        for col in nan_columns:
            print(f"Number of NaNs in column {col}: {df[col].isna().sum()}")


def check_missing_operations(df, expected_operations):
    present_operations = df["Operation"].unique()
    missing_operations = [
        op for op in expected_operations if op not in present_operations
    ]
    return missing_operations


def save_statistics_to_file(roi, stats_list):
    with open(f"/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/{roi}_statistics.txt", "a") as file:
        for stat in stats_list:
            file.write(f"{stat}\n")
        file.write("--------------------------------------------------\n")


expected_operations = ["Replace", "Maintain", "Suppress"]


def main(roi):
    subject_ids = [
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
    unpaired_all_remembered_item = pd.DataFrame()
    unpaired_all_forgot_item = pd.DataFrame()
    unpaired_all_remembered_cate = pd.DataFrame()
    unpaired_all_forgot_cate = pd.DataFrame()

    paired_agg_all_remembered_item = pd.DataFrame()
    paired_agg_all_forgot_item = pd.DataFrame()
    paired_agg_all_remembered_cate = pd.DataFrame()
    paired_agg_all_forgot_cate = pd.DataFrame()

    for subID in subject_ids:
        base_path = f"/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/sub-0{subID}/Representational_Changes_MNI_{roi}"
        remembered_item_df = load_data(
            os.path.join(base_path, "itemweighted_remembered_fidelity.csv")
        )
        check_for_nans(remembered_item_df, "remembered_item_df", subID)

        forgot_item_df = load_data(
            os.path.join(base_path, "itemweighted_forgot_fidelity.csv")
        )
        check_for_nans(forgot_item_df, "forgot_item_df", subID)

        remembered_cate_df = load_data(
            os.path.join(base_path, "cateweighted_remembered_fidelity.csv")
        )
        check_for_nans(remembered_cate_df, "remembered_cate_df", subID)

        forgot_cate_df = load_data(
            os.path.join(base_path, "cateweighted_forgot_fidelity.csv")
        )
        check_for_nans(forgot_cate_df, "forgot_cate_df", subID)

        # Add the 'Subject' column to the raw data DataFrames
        remembered_item_df["Subject"] = subID
        remembered_item_df["Memory"] = "remembered"
        forgot_item_df["Subject"] = subID
        forgot_item_df["Memory"] = "forgot"
        remembered_cate_df["Subject"] = subID
        remembered_cate_df["Memory"] = "remembered"
        forgot_cate_df["Subject"] = subID
        forgot_cate_df["Memory"] = "forgot"

        # Concatenate to unpaired DataFrames without aggregation
        unpaired_all_remembered_item = pd.concat(
            [unpaired_all_remembered_item, remembered_item_df]
        )
        unpaired_all_forgot_item = pd.concat([unpaired_all_forgot_item, forgot_item_df])
        unpaired_all_remembered_cate = pd.concat(
            [unpaired_all_remembered_cate, remembered_cate_df]
        )
        unpaired_all_forgot_cate = pd.concat([unpaired_all_forgot_cate, forgot_cate_df])

        # Check for missing operations
        missing_ops = check_missing_operations(forgot_item_df, expected_operations)
        if missing_ops:
            print(f"Subject {subID} is missing the following operations: {missing_ops}")
            continue  # Skip this subject for paired analysis, but still use it for unpaired analysis

        # Aggregate and append to paired DataFrames
        agg_remembered_item = aggregate_data(remembered_item_df, "remembered")
        agg_remembered_item["Subject"] = subID
        paired_agg_all_remembered_item = pd.concat(
            [paired_agg_all_remembered_item, agg_remembered_item]
        )

        agg_forgot_item = aggregate_data(forgot_item_df, "forgot")
        agg_forgot_item["Subject"] = subID
        paired_agg_all_forgot_item = pd.concat(
            [paired_agg_all_forgot_item, agg_forgot_item]
        )

        agg_remembered_cate = aggregate_data(remembered_cate_df, "remembered")
        agg_remembered_cate["Subject"] = subID
        paired_agg_all_remembered_cate = pd.concat(
            [paired_agg_all_remembered_cate, agg_remembered_cate]
        )

        agg_forgot_cate = aggregate_data(forgot_cate_df, "forgot")
        agg_forgot_cate["Subject"] = subID
        paired_agg_all_forgot_cate = pd.concat(
            [paired_agg_all_forgot_cate, agg_forgot_cate]
        )

    # Path to save group-level CSVs
    group_save_path = f"/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/{roi}_group_level_data"
    if not os.path.exists(group_save_path):
        os.makedirs(group_save_path)
    # Save group-level data
    paired_agg_all_remembered_item.to_csv(
        os.path.join(
            group_save_path, f"paired_group_itemweighted_remembered_fidelity_{roi}.csv"
        ),
        index=False,
    )
    paired_agg_all_forgot_item.to_csv(
        os.path.join(
            group_save_path, f"paired_group_itemweighted_forgot_fidelity_{roi}.csv"
        ),
        index=False,
    )
    paired_agg_all_remembered_cate.to_csv(
        os.path.join(
            group_save_path, f"paired_group_cateweighted_remembered_fidelity_{roi}.csv"
        ),
        index=False,
    )
    paired_agg_all_forgot_cate.to_csv(
        os.path.join(
            group_save_path, f"paired_group_cateweighted_forgot_fidelity_{roi}.csv"
        ),
        index=False,
    )

    unpaired_all_remembered_item.to_csv(
        os.path.join(
            group_save_path, f"unpaired_all_itemweighted_remembered_fidelity_{roi}.csv"
        ),
        index=False,
    )
    unpaired_all_forgot_item.to_csv(
        os.path.join(
            group_save_path, f"unpaired_all_itemweighted_forgot_fidelity_{roi}.csv"
        ),
        index=False,
    )
    unpaired_all_remembered_cate.to_csv(
        os.path.join(
            group_save_path, f"unpaired_all_cateweighted_remembered_fidelity_{roi}.csv"
        ),
        index=False,
    )
    unpaired_all_forgot_cate.to_csv(
        os.path.join(
            group_save_path, f"unpaired_all_cateweighted_forgot_fidelity_{roi}.csv"
        ),
        index=False,
    )

    # Perform t-tests for paired data
    print("Performing Paired t-tests for Item Weighted Data:")
    perform_t_tests(
        paired_agg_all_remembered_item, paired_agg_all_forgot_item, roi, "paired"
    )
    print("--------------------------------------------------")

    print("Performing Paired t-tests for Category Weighted Data:")
    perform_t_tests(
        paired_agg_all_remembered_cate, paired_agg_all_forgot_cate, roi, "paired"
    )
    print("--------------------------------------------------")

    # Perform t-tests for unpaired data
    print("Performing Unpaired t-tests for Item Weighted Data:")
    perform_t_tests(unpaired_all_remembered_item, unpaired_all_forgot_item, roi, "unpaired")
    print("--------------------------------------------------")

    print("Performing Unpaired t-tests for Category Weighted Data:")
    perform_t_tests(unpaired_all_remembered_cate, unpaired_all_forgot_cate, roi, "unpaired")
    print("--------------------------------------------------")

    # Generate and save plots for paired data
    plot_data(
        pd.concat([paired_agg_all_remembered_item, paired_agg_all_forgot_item]),
        f"Paired Item Weighted - {roi}",
        f"/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/figs/paired_item_summary_plot_{roi}.svg",
    )
    plot_data(
        pd.concat([paired_agg_all_remembered_cate, paired_agg_all_forgot_cate]),
        f"Paired Category Weighted - {roi}",
        f"/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/figs/paired_cate_summary_plot_{roi}.svg",
    )

    # Generate and save plots for unpaired data
    plot_data(
        pd.concat([unpaired_all_remembered_item, unpaired_all_forgot_item]),
        f"Unpaired Item Weighted - {roi}",
        f"/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/figs/unpaired_item_summary_plot_{roi}.svg",
    )
    plot_data(
        pd.concat([unpaired_all_remembered_cate, unpaired_all_forgot_cate]),
        f"Unpaired Category Weighted - {roi}",
        f"/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/figs/unpaired_cate_summary_plot_{roi}.svg",
    )


if __name__ == "__main__":
    for roi in [
        "Prefrontal_ROI",
        "Higher_Order_Visual_ROI",
        "hippocampus_ROI",
        "VTC_mask",
    ]:
        main(roi)
