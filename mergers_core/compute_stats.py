# coding: utf-8
import pandas as pd
import numpy as np


def compute_stats(df):
    dissim_rows = df[
        (df["dissimilarity_weight"] == 1) & (df["population_consistency_weight"] == 0)
    ]
    pop_rows = df[
        (df["dissimilarity_weight"] == 0) & (df["population_consistency_weight"] == 1)
    ]
    both_rows = df[
        (df["dissimilarity_weight"] == 1) & (df["population_consistency_weight"] == 1)
    ]

    final_df = pd.DataFrame(
        np.zeros((3, 4), dtype=np.float64),
        columns=[
            "median %Δdissim",
            "average %Δdissim",
            "median %Δpop",
            "average %Δpop",
        ],
    )
    for stat_index, row in enumerate([dissim_rows, pop_rows, both_rows]):
        pre_dissim = np.array(row["pre_dissim_bh_wa"])
        with np.errstate(divide="ignore", invalid="ignore"):
            Δdissim = (np.array(row["post_dissim_bh_wa"]) - pre_dissim) / pre_dissim
        pre_pop = np.array(row["pre_population_consistency"])
        with np.errstate(divide="ignore", invalid="ignore"):
            Δpop = (np.array(row["post_population_consistency"]) - pre_pop) / pre_pop
        final_df.loc[stat_index] = [
            np.median(Δdissim),
            np.mean(Δdissim),
            np.median(Δpop),
            np.mean(Δpop),
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
