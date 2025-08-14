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
        columns=["median Δdissim", "average Δdissim", "median Δpop", "average Δpop"],
    )
    for stat_index, row in enumerate([dissim_rows, pop_rows, both_rows]):
        Δdissim = np.array(row["post_dissim_bh_wa"]) - np.array(row["pre_dissim_bh_wa"])
        Δpop = np.array(row["post_population_consistency"]) - np.array(
            row["pre_population_consistency"]
        )
        final_df.loc[stat_index] = [
            np.median(Δdissim),
            np.mean(Δdissim),
            np.median(Δpop),
            np.mean(Δpop),
        ]
    final_df.index = ["dissim", "pop", "both"]
    return final_df


df = pd.read_csv("../data/results/top-200.csv")

print("school decrease threshold: 0.2")
print(compute_stats(df[df["school_decrease_threshold"] == 0.2]))
print()
print("school decrease threshold: 1.0")
print(compute_stats(df[df["school_decrease_threshold"] == 1.0]))
