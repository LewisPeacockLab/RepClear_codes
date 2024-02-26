import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os
from matplotlib.legend_handler import HandlerTuple
from sklearn.utils import resample
import numpy as np
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from statsmodels.formula.api import ols, mixedlm

rois = ["VTC_mask"]


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

    for op in operations:
        data1 = agg_df1.loc[agg_df1["Operation"] == op, "Fidelity"].to_numpy()
        data2 = agg_df2.loc[agg_df2["Operation"] == op, "Fidelity"].to_numpy()

        # Perform the appropriate t-test based on test_type
        if test_type == "paired":
            t_stat, p_val = stats.ttest_rel(data1, data2, nan_policy="omit")
        else:
            t_stat, p_val = stats.ttest_ind(data1, data2, nan_policy="omit")

        # Calculate degrees of freedom
        DoF = len(data1) - 1 if test_type == "paired" else len(data1) + len(data2) - 2

        # Store p-values for correction
        p_values.append(p_val)

        # Prepare the results for this operation
        result = {
            "Operation": op,
            "TestType": test_type,
            "t_stat": t_stat,
            "p_val": p_val,
            "DoF": DoF,
        }
        stats_results.append(result)
        print(
            f"{test_type.capitalize()} t-test for operation {op}: t = {t_stat}, p = {p_val}, "
            f"DoF = {DoF}"
        )

    # Save to text file
    stats_df = pd.DataFrame(stats_results)

    save_path = (
        f"/Volumes/zbg_eHD/Zachary_Data/processed/{roi}_{test_type}_t_test_stats.txt"
    )
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


def plot_data(data_list, title_list, save_path_list, plot_types, y_lim=None):
    """
    Generate Seaborn plots for the list of data provided.
    Each entry in the data_list corresponds to a set of data to plot.
    The title_list and save_path_list contain corresponding titles and file paths.
    The plot_types list determines the kind of plot to generate ('bar' or 'violin').
    """
    if y_lim is None:
        y_lim = (
            pd.concat(data_list)["Fidelity"].min(),
            pd.concat(data_list)["Fidelity"].max(),
        )

    for data, title, save_path, plot_type in zip(
        data_list, title_list, save_path_list, plot_types
    ):
        plt.figure(figsize=(10, 6))  # You can adjust the figure size as needed
        if plot_type == "bar":
            fig = sns.barplot(
                data=data,
                x="Operation",
                y="Fidelity",
                hue="Memory",
                ci=95,
                palette={"remembered": "black", "forgot": "grey"},
                edgecolor=".7",
            )
        elif plot_type == "violin":
            fig = sns.violinplot(
                data=data,
                x="Operation",
                y="Fidelity",
                hue="Memory",
                palette={"remembered": "black", "forgot": "grey"},
                split=True,
            )

        plt.ylim(y_lim)
        plt.title(title, loc="center", wrap=True)
        plt.xlabel("Operations")
        plt.ylabel("Fidelity of item-RSA")
        plt.tight_layout()

        # Only create legend if it's a barplot
        if plot_type == "bar":
            plt.legend(
                handles=[tuple(bar_group) for bar_group in fig.containers],
                labels=[bar_group.get_label() for bar_group in fig.containers],
                title="Memory",
                handlelength=4,
                handler_map={tuple: HandlerTuple(ndivide=None, pad=0.1)},
            )

        plt.savefig([save_path + f"{plot_type}.svg"])
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
        f"/Volumes/zbg_eHD/Zachary_Data/processed/{roi}_statistics.txt",
        "a",
    ) as file:
        for stat in stats_list:
            file.write(f"{stat}\n")
        file.write("--------------------------------------------------\n")


def perform_anova(*args):
    f_stat, p_val = f_oneway(*args)
    print(f"ANOVA results: F = {f_stat}, p = {p_val}")
    return f_stat, p_val


def perform_mixed_effects_model(combined_data, roi):
    # Convert 'Subject' to a categorical variable
    combined_data["Subject"] = combined_data["Subject"].astype("category")
    # Convert 'Memory' and 'Operation' to categorical variables
    combined_data["Memory"] = combined_data["Memory"].astype("category")
    combined_data["Operation"] = combined_data["Operation"].astype("category")

    # Define the model formula to include the operation effects
    model_formula = "Fidelity ~ Memory * Operation"

    # Fit the mixed-effects model
    mixed_model = mixedlm(
        model_formula,
        data=combined_data,
        re_formula="1",
        groups=combined_data["Subject"],
    )
    mixed_model_result = mixed_model.fit()

    # Print the summary of the mixed-effects model
    print(mixed_model_result.summary())

    # Save the summary to a text file
    summary_path = (
        f"/Volumes/zbg_eHD/Zachary_Data/processed/{roi}_mixed_effects_model_summary.txt"
    )
    with open(summary_path, "w") as summary_file:
        summary_file.write(mixed_model_result.summary().as_text())

    return mixed_model_result


def perform_mixed_effects_model_condition_only(combined_data, condition, roi):
    # Filter the data to only include the 'Suppress' operation and create a copy to avoid SettingWithCopyWarning
    suppress_data = combined_data[combined_data["Operation"] == condition].copy()

    # Convert 'Subject' to a categorical variable
    suppress_data["Subject"] = suppress_data["Subject"].astype("category")
    # Convert 'Memory' to a categorical variable
    suppress_data["Memory"] = suppress_data["Memory"].astype("category")

    # Define the model formula to include the operation effects
    model_formula = "Fidelity ~ Memory"

    # Fit the mixed-effects model
    mixed_model = mixedlm(
        model_formula, data=suppress_data, groups=suppress_data["Subject"]
    )
    mixed_model_result = mixed_model.fit()

    # Print the summary of the mixed-effects model
    print(mixed_model_result.summary())

    # Save the summary to a text file
    summary_path = f"/Volumes/zbg_eHD/Zachary_Data/processed/{roi}_suppress_only_mixed_effects_model_summary.txt"
    with open(summary_path, "w") as summary_file:
        summary_file.write(mixed_model_result.summary().as_text())

    return mixed_model_result


def plot_suppress_data(item_df, category_df, plot_title, save_base_path):
    """
    Generate a Seaborn barplot and a split violin plot for "Suppress" operation data only,
    with one pair of bars/violins for item-level data and another for category-level data,
    all in one plot. The function will produce clean and editable SVG files.
    """
    # Filter for "Suppress" operation
    suppress_item = item_df[item_df["Operation"] == "Suppress"].copy()
    suppress_category = category_df[category_df["Operation"] == "Suppress"].copy()

    # Add a new column to distinguish between item and category data
    suppress_item["Type"] = "Item"
    suppress_category["Type"] = "Category"

    # Combine the two dataframes
    combined_df = pd.concat([suppress_item, suppress_category])

    # Bar plot
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=combined_df,
        x="Type",
        y="Fidelity",
        hue="Memory",
        palette={"remembered": "white", "forgot": "gray"},
        edgecolor=".7",
    )
    plt.title(plot_title + " - Bar Plot")
    plt.xlabel("")
    plt.ylabel("Fidelity of item-RSA")
    plt.legend(title="Memory", loc="upper right")
    plt.savefig(f"{save_base_path}_Bar.svg", format="svg")
    plt.close()

    # Violin plot
    plt.figure(figsize=(8, 6))
    sns.violinplot(
        data=combined_df,
        x="Type",
        y="Fidelity",
        hue="Memory",
        split=True,  # This creates the split violin plot
        palette={"remembered": "white", "forgot": "gray"},
        gap=0.1,
        # inner="quart",
        linewidth=1.25,
    )
    plt.title(plot_title + " - Violin Plot")
    plt.xlabel("")
    plt.ylabel("Fidelity of item-RSA")
    plt.legend(title="Memory", loc="upper right")
    plt.savefig(f"{save_base_path}_Violin.svg", format="svg")
    plt.close()


def calculate_and_log_mean_fidelity(df, label, subID):
    # Calculate the mean fidelity for the current subject and log it
    mean_fidelity = df.groupby(["Operation", "Memory"])["Fidelity"].mean().reset_index()
    print(f"Mean Fidelity for {label} - Subject {subID}:\n{mean_fidelity}\n")
    return mean_fidelity


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

    qa_summary = []

    for subID in subject_ids:
        base_path = f"/Volumes/zbg_eHD/Zachary_Data/processed/sub-0{subID}/Representational_Changes_MNI_{roi}"
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

        qa_summary.append(
            {
                "subject": subID,
                "item_remembered_mean": calculate_and_log_mean_fidelity(
                    remembered_item_df, "Item Remembered", subID
                ),
                "item_forgot_mean": calculate_and_log_mean_fidelity(
                    forgot_item_df, "Item Forgot", subID
                ),
                "category_remembered_mean": calculate_and_log_mean_fidelity(
                    remembered_cate_df, "Category Remembered", subID
                ),
                "category_forgot_mean": calculate_and_log_mean_fidelity(
                    forgot_cate_df, "Category Forgot", subID
                ),
            }
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

    mixed_effects_result_item = perform_mixed_effects_model_condition_only(
        combined_paired_item_data, "Suppress", roi
    )
    mixed_effects_result_category = perform_mixed_effects_model_condition_only(
        combined_paired_cate_data, "Suppress", roi
    )

    mixed_effects_result_item = perform_mixed_effects_model_condition_only(
        combined_unpaired_item_data, "Suppress", roi
    )
    mixed_effects_result_category = perform_mixed_effects_model_condition_only(
        combined_unpaired_cate_data, "Suppress", roi
    )

    # Path to save group-level CSVs
    group_save_path = (
        f"/Volumes/zbg_eHD/Zachary_Data/processed/fmriprep/{roi}_group_level_data"
    )
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

    # # Perform t-tests for unpaired data
    # print("Performing Unpaired t-tests for Item Weighted Data:")
    # perform_t_tests(
    #     unpaired_all_remembered_item, unpaired_all_forgot_item, roi, "unpaired"
    # )
    # print("--------------------------------------------------")

    # print("Performing Unpaired t-tests for Category Weighted Data:")
    # perform_t_tests(
    #     unpaired_all_remembered_cate, unpaired_all_forgot_cate, roi, "unpaired"
    # )
    # print("--------------------------------------------------")

    # Define your dataframes here
    item_df = pd.concat([paired_agg_all_remembered_item, paired_agg_all_forgot_item])
    category_df = pd.concat(
        [paired_agg_all_remembered_cate, paired_agg_all_forgot_cate]
    )

    # Call the function with the item and category dataframes
    plot_title = "Suppress Operation Fidelity - Item vs. Category"
    save_path = "/Volumes/zbg_eHD/Zachary_Data/processed/figs/Suppress_Fidelity_Plot"
    plot_suppress_data(item_df, category_df, plot_title, save_path)

    un_item_df = pd.concat(
        [unpaired_agg_all_remembered_item, unpaired_agg_all_forgot_item]
    )
    un_category_df = pd.concat(
        [unpaired_agg_all_remembered_cate, unpaired_agg_all_forgot_cate]
    )

    # Call the function with the item and category dataframes
    plot_title = "Suppress Operation Fidelity - Item vs. Category _ unpaired"
    save_path = "/Volumes/zbg_eHD/Zachary_Data/processed/figs/Suppress_Fidelity_Plot_un"
    plot_suppress_data(un_item_df, un_category_df, plot_title, save_path)

    # plot_data(
    #     pd.concat([paired_agg_all_remembered_item, paired_agg_all_forgot_item]),
    #     f"Paired Item Weighted - {roi}",
    #     f"/Volumes/zbg_eHD/Zachary_Data/processed/figs/paired_item_summary_plot_{roi}.svg",
    # )
    # plot_data(
    #     pd.concat([paired_agg_all_remembered_cate, paired_agg_all_forgot_cate]),
    #     f"Paired Category Weighted - {roi}",
    #     f"/Volumes/zbg_eHD/Zachary_Data/processed/figs/paired_cate_summary_plot_{roi}.svg",
    # )

    # # Generate and save plots for unpaired data
    # plot_data(
    #     pd.concat([unpaired_all_remembered_item, unpaired_all_forgot_item]),
    #     f"Unpaired Item Weighted - {roi}",
    #     f"/Volumes/zbg_eHD/Zachary_Data/processed/figs/unpaired_item_summary_plot_{roi}.svg",
    # )
    # plot_data(
    #     pd.concat([unpaired_all_remembered_cate, unpaired_all_forgot_cate]),
    #     f"Unpaired Category Weighted - {roi}",
    #     f"/Volumes/zbg_eHD/Zachary_Data/processed/figs/unpaired_cate_summary_plot_{roi}.svg",
    # )


if __name__ == "__main__":
    for roi in [
        "VTC_mask",
    ]:
        main(roi)
