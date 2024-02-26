import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os
import pingouin as pg
from matplotlib.legend_handler import HandlerTuple
from sklearn.utils import resample
import numpy as np
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from statsmodels.formula.api import ols, mixedlm, smf

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


def perform_t_tests(agg_df1, agg_df2, roi, test_type="paired", n_iterations=10000):
    operations = agg_df1["Operation"].unique()
    stats_results = []
    p_values = []
    empirical_p_values = []

    for op in operations:
        data1 = agg_df1.loc[agg_df1["Operation"] == op, "Fidelity"].to_numpy()
        data2 = agg_df2.loc[agg_df2["Operation"] == op, "Fidelity"].to_numpy()

        # Perform the bootstrap
        bootstrap_differences, empirical_p, observed_difference = perform_bootstrap(
            data1, data2, paired=test_type == "paired", n_iterations=n_iterations
        )

        # Calculate 95% CI for difference
        lower = np.percentile(bootstrap_differences, 2.5)
        upper = np.percentile(bootstrap_differences, 97.5)

        # Perform the appropriate t-test based on test_type
        if test_type == "paired":
            t_stat, p_val = stats.ttest_rel(data1, data2, nan_policy="omit")
        else:
            t_stat, p_val = stats.ttest_ind(data1, data2, nan_policy="omit")

        # Calculate degrees of freedom
        DoF = len(data1) - 1 if test_type == "paired" else len(data1) + len(data2) - 2

        # Store p-values for correction
        p_values.append(p_val)
        empirical_p_values.append(empirical_p)

        # Prepare the results for this operation
        result = {
            "Operation": op,
            "TestType": test_type,
            "ObservedDifference": observed_difference,
            "EmpiricalP": empirical_p,
            "CI95_Lower": lower,
            "CI95_Upper": upper,
            "t_stat": t_stat,
            "p_val": p_val,
            "DoF": DoF,
        }
        stats_results.append(result)
        print(
            f"{test_type.capitalize()} t-test for operation {op}: t = {t_stat}, p = {p_val}, "
            f"DoF = {DoF}, Empirical p = {empirical_p}, "
            f"Bootstrap 95% CI for difference = ({lower}, {upper})"
        )

    # After applying correction
    corrected_p_values = multipletests(p_values, alpha=0.05, method="fdr_bh")[1]
    corrected_empirical_p_values = multipletests(
        empirical_p_values, alpha=0.05, method="fdr_bh"
    )[1]

    for idx, op in enumerate(operations):
        stats_results[idx]["CorrectedP"] = corrected_p_values[idx]
        stats_results[idx]["CorrectedEmpiricalP"] = corrected_empirical_p_values[idx]
        print(
            f"Operation {op} has a corrected p-value of: {corrected_p_values[idx]} and a corrected empirical p-value of: {corrected_empirical_p_values[idx]}"
        )

    # Save to text file
    stats_df = pd.DataFrame(stats_results)

    save_path = f"/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/ROI_{roi}_{test_type}_t_test_stats.txt"
    if os.path.exists(save_path):
        write_mode = "a"  # append if already exists
    else:
        write_mode = "w"  # make a new file if not

    with open(save_path, write_mode) as file:
        # Write the statistics to the file
        file.write(stats_df.to_csv(index=False, sep="\t"))

        # Add a separator for readability
        file.write("--------------------------------------------------\n")


def perform_bootstrap_t_test(data1, data2, n_iterations=10000):
    # Original test statistic
    original_t_stat, _ = ttest_rel(data1, data2)

    # Store the test statistics from the bootstrap samples
    bootstrap_t_statistics = []

    for i in range(n_iterations):
        # Sample with replacement from the paired differences
        indices = np.arange(len(data1))
        resampled_indices = np.random.choice(indices, size=len(indices), replace=True)
        resampled_data1 = data1[resampled_indices]
        resampled_data2 = data2[resampled_indices]

        # Compute the test statistic for the resampled data
        t_stat, _ = ttest_rel(resampled_data1, resampled_data2)
        bootstrap_t_statistics.append(t_stat)

    # Empirical p-value: proportion of resampled t-stats as extreme as the original
    empirical_p = np.mean(np.abs(bootstrap_t_statistics) >= np.abs(original_t_stat))

    return bootstrap_t_statistics, empirical_p


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
    with open(
        f"/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/{roi}_statistics.txt",
        "a",
    ) as file:
        for stat in stats_list:
            file.write(f"{stat}\n")
        file.write("--------------------------------------------------\n")


def perform_anova(*args):
    f_stat, p_val = f_oneway(*args)
    print(f"ANOVA results: F = {f_stat}, p = {p_val}")
    return f_stat, p_val


def perform_mixed_effects_model(df):
    """
    This function fits a mixed-effects model to the provided dataframe, which should
    contain the combined data for all subjects and conditions.

    Parameters:
    df (DataFrame): A pandas DataFrame with the following columns:
                    'Subject', 'Memory', 'Fidelity_Type', 'Fidelity'

    Returns:
    Summary: A summary of the mixed-effects model results.
    """
    # Ensure the subject column is treated as a categorical variable for the model
    df["Subject"] = df["Subject"].astype("category")

    # Define the model formula
    model_formula = "Fidelity ~ C(Memory) * C(Fidelity_Type)"

    # Fit the mixed-effects model
    mixed_effects_model = smf.mixedlm(
        model_formula, df, groups=df["Subject"], re_formula="~Memory"
    )
    mixed_effects_model_fit = mixed_effects_model.fit(
        method="nm", maxiter=200, full_output=True
    )

    # Return the summary of the model
    return mixed_effects_model_fit.summary()


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

    # Combine the remembered and forgotten item data into one DataFrame
    combined_paired_item_data = pd.concat(
        [paired_agg_all_remembered_item, paired_agg_all_forgot_item]
    )

    # Combine the remembered and forgotten category data into one DataFrame
    combined_paired_cate_data = pd.concat(
        [paired_agg_all_remembered_cate, paired_agg_all_forgot_cate]
    )

    # Combine the remembered and forgotten item data into one DataFrame
    combined_unpaired_item_data = pd.concat(
        [unpaired_all_remembered_item, unpaired_all_forgot_item]
    )

    # Combine the remembered and forgotten category data into one DataFrame
    combined_unpaired_cate_data = pd.concat(
        [unpaired_all_remembered_cate, unpaired_all_forgot_cate]
    )

    # Performing mixed-effects model analysis on the combined paired item data
    mixed_model_item, summary_item = perform_mixed_effects_analysis(
        combined_paired_item_data,
        dependent_var="Fidelity",
        fixed_effects=["Operation", "Memory"],
        group_var="Subject",
    )
    print("Mixed-effects model analysis for Paired Item Data:")
    print(summary_item)

    # Performing mixed-effects model analysis on the combined paired category data
    mixed_model_cate, summary_cate = perform_mixed_effects_analysis(
        combined_paired_cate_data,
        dependent_var="Fidelity",
        fixed_effects=["Operation", "Memory"],
        group_var="Subject",
    )
    print("Mixed-effects model analysis for Paired Category Data:")
    print(summary_cate)

    def perform_two_way_anova(dataframe, dependent_var, factor1, factor2):
        model = ols(
            f"{dependent_var} ~ C({factor1}) * C({factor2})", data=dataframe
        ).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(anova_table)

    # Perform the 3x2 ANOVA for item weighted data
    print("3x2 ANOVA for Item Weighted Data - Paired:")
    perform_two_way_anova(combined_paired_item_data, "Fidelity", "Operation", "Memory")

    # Perform the 3x2 ANOVA for category weighted data
    print("3x2 ANOVA for Category Weighted Data - Paired:")
    perform_two_way_anova(combined_paired_cate_data, "Fidelity", "Operation", "Memory")

    # Perform the 3x2 ANOVA for item weighted data
    print("3x2 ANOVA for Item Weighted Data - Unpaired:")
    perform_two_way_anova(
        combined_unpaired_item_data, "Fidelity", "Operation", "Memory"
    )

    # Perform the 3x2 ANOVA for category weighted data
    print("3x2 ANOVA for Category Weighted Data - Unpaired:")
    perform_two_way_anova(
        combined_unpaired_cate_data, "Fidelity", "Operation", "Memory"
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

    print("--------------------------------------------------")
    print(f"STATS FOR {roi}")
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
    perform_t_tests(
        unpaired_all_remembered_item, unpaired_all_forgot_item, roi, "unpaired"
    )
    print("--------------------------------------------------")

    print("Performing Unpaired t-tests for Category Weighted Data:")
    perform_t_tests(
        unpaired_all_remembered_cate, unpaired_all_forgot_cate, roi, "unpaired"
    )
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
