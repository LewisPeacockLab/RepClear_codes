import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os
from matplotlib.legend_handler import HandlerTuple


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
    agg_df = df.groupby("Operation")["Fidelity"].mean().reset_index()
    agg_df["Memory"] = memory_status
    return agg_df


def perform_t_tests(agg_remembered_df, agg_forgot_df):
    """
    Perform paired t-tests between remembered and forgotten data for each operation.
    """
    operations = agg_remembered_df["Operation"].unique()
    for op in operations:
        remembered_fidelity = agg_remembered_df.loc[
            agg_remembered_df["Operation"] == op, "Fidelity"
        ]
        forgot_fidelity = agg_forgot_df.loc[
            agg_forgot_df["Operation"] == op, "Fidelity"
        ]
        t_stat, p_val = stats.ttest_rel(remembered_fidelity, forgot_fidelity)
        print(f"Paired t-test for operation {op}: t = {t_stat}, p = {p_val}")


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
    fig.set_ylim([0, 0.05])

    fig.legend(
        handles=[tuple(bar_group) for bar_group in fig.containers],
        labels=[bar_group.get_label() for bar_group in fig.containers],
        title=fig.legend_.get_title().get_text(),
        handlelength=4,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.1)},
    )

    plt.savefig(save_path)
    plt.clf()


def main():
    # file paths
    remembered_csv_path = "path/to/remembered_csv"
    forgot_csv_path = "path/to/forgot_csv"

    # Load the data
    remembered_df = load_data(remembered_csv_path)
    forgot_df = load_data(forgot_csv_path)

    # Aggregate the data
    agg_remembered_df = aggregate_data(remembered_df, "Remembered")
    agg_forgot_df = aggregate_data(forgot_df, "Forgot")

    # Perform t-tests
    perform_t_tests(agg_remembered_df, agg_forgot_df)

    # Combine the data for plotting
    combined_df = pd.concat([agg_remembered_df, agg_forgot_df])

    # Generate and save plots
    plot_data(
        combined_df, "Item Weighted - Pre vs. Post RSA", "path/to/save/summary_plot.svg"
    )


if __name__ == "__main__":
    main()
