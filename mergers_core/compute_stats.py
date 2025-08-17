# coding: utf-8
import pandas as pd
import numpy as np

POPULATION_METRIC = "median_difference"


def compute_stats(df):
    dissim_rows = df[
        (df["dissimilarity_weight"] == 1) & (df["population_metric_weight"] == 0)
    ]
    pop_rows = df[
        (df["dissimilarity_weight"] == 0) & (df["population_metric_weight"] == 1)
    ]
    both_rows = df[
        (df["dissimilarity_weight"] == 1) & (df["population_metric_weight"] == 1)
    ]

    final_df = pd.DataFrame(
        np.zeros((3, 10), dtype=np.float64),
        columns=[
            "med %Δdissim",
            "avg %Δdissim",
            "med %Δpop (avg)",
            "avg %Δpop (avg)",
            "med %Δpop (med)",
            "avg %Δpop (med)",
            "med %Δpop (avgΔ)",
            "avg %Δpop (avgΔ)",
            "med %Δpop (medΔ)",
            "avg %Δpop (medΔ)",
        ],
    )
    for stat_index, row in enumerate([dissim_rows, pop_rows, both_rows]):
        pre_dissim = np.array(row["pre_dissim_bh_wa"])
        Δdissim = (np.array(row["post_dissim_bh_wa"]) - pre_dissim) / pre_dissim

        pre_pop_average = np.array(row["pre_population_average"])
        Δpop_average = (
            np.array(row["post_population_average"]) - pre_pop_average
        ) / pre_pop_average

        pre_pop_median = np.array(row["pre_population_median"])
        Δpop_median = (
            np.array(row["post_population_median"]) - pre_pop_median
        ) / pre_pop_median

        pre_pop_average_difference = np.array(row["pre_population_average_difference"])
        Δpop_average_difference = (
            np.array(row["post_population_average_difference"])
            - pre_pop_average_difference
        ) / pre_pop_average_difference

        pre_pop_median_difference = np.array(row["pre_population_median_difference"])
        Δpop_median_difference = (
            np.array(row["post_population_median_difference"])
            - pre_pop_median_difference
        ) / pre_pop_median_difference
        final_df.loc[stat_index] = [
            np.median(Δdissim) * 100,
            np.mean(Δdissim) * 100,
            np.median(Δpop_median) * 100,
            np.mean(Δpop_median) * 100,
            np.median(Δpop_average) * 100,
            np.mean(Δpop_average) * 100,
            np.median(Δpop_average_difference) * 100,
            np.mean(Δpop_average_difference) * 100,
            np.median(Δpop_median_difference) * 100,
            np.mean(Δpop_median_difference) * 100,
        ]
    final_df.index = ["dissim", "pop", "both"]
    return final_df


df = pd.read_csv("../data/results/top-200.csv")

print("runs with a capacity lower bound of 80%:")
print(compute_stats(df[df["school_decrease_threshold"] == 0.2]))

print("\nruns with a capacity lower bound of 0%:")
print(compute_stats(df[df["school_decrease_threshold"] == 1.0]))

print("\nall runs:")
print(compute_stats(df))
