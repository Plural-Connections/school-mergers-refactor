# coding: utf-8
import pandas as pd
import numpy as np

POPULATION_METRIC = "median_divergence"


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
            "m %Δdissim",
            "a %Δdissim",
            "m %Δpop (a)",
            "a %Δpop (a)",
            "m %Δpop (m)",
            "a %Δpop (m)",
            "m %Δpop (aΔ)",
            "a %Δpop (aΔ)",
            "m %Δpop (mΔ)",
            "a %Δpop (mΔ)",
        ],
    )
    for stat_index, row in enumerate([dissim_rows, pop_rows, both_rows]):
        pre_dissim = np.array(row["pre_dissim_bh_wa"])
        post_dissim = np.array(row["post_dissim_bh_wa"])
        Δdissim = (post_dissim - pre_dissim) / pre_dissim

        pre_pop_average = np.array(row["pre_population_average"])
        post_pop_average = np.array(row["post_population_average"])
        Δpop_average = (post_pop_average - pre_pop_average) / pre_pop_average

        pre_pop_median = np.array(row["pre_population_median"])
        post_pop_median = np.array(row["post_population_median"])
        Δpop_median = (post_pop_median - pre_pop_median) / pre_pop_median

        pre_pop_average_divergence = np.array(row["pre_population_average_divergence"])
        post_pop_average_divergence = np.array(
            row["post_population_average_divergence"]
        )
        Δpop_average_divergence = (
            post_pop_average_divergence - pre_pop_average_divergence
        ) / pre_pop_average_divergence

        pre_pop_median_divergence = np.array(row["pre_population_median_divergence"])
        post_pop_median_divergence = np.array(row["post_population_median_divergence"])
        Δpop_median_divergence = (
            post_pop_median_divergence - pre_pop_median_divergence
        ) / pre_pop_median_divergence

        final_df.loc[stat_index] = [
            np.median(Δdissim) * 100,
            np.mean(Δdissim) * 100,
            np.median(Δpop_median) * 100,
            np.mean(Δpop_median) * 100,
            np.median(Δpop_average) * 100,
            np.mean(Δpop_average) * 100,
            np.median(Δpop_average_divergence) * 100,
            np.mean(Δpop_average_divergence) * 100,
            np.median(Δpop_median_divergence) * 100,
            np.mean(Δpop_median_divergence) * 100,
        ]
    final_df.index = ["dissim", "pop", "both"]
    return final_df


print("m = median; a = average; SDT = school decrease threshold")

df = pd.read_csv("../data/results/top-200.csv")
df1 = df[df[""] == "bh_wa"]
stats = pd.concat(
    [
        compute_stats(df[df["school_decrease_threshold"] == 0.2]),
        compute_stats(df[df["school_decrease_threshold"] == 1.0]),
        compute_stats(df),
    ],
    keys=["0.2", "1.0", "all"],
)
stats.index.names = ["SDT", "objective"]

print(stats)
stats.to_csv("stats.csv", float_format="%.6f")
