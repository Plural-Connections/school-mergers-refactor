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

    results = pd.DataFrame(
        np.zeros((3, 8), dtype=np.float64),
        columns=[
            "m %Δdissim",
            "a %Δdissim",
            "m %Δpop (aΔ)",
            "a %Δpop (aΔ)",
            "prem dissim",
            "postm dissim",
            "prem pop (aΔ)",
            "postm pop (aΔ)",
        ],
    )
    for stat_index, row in enumerate([dissim_rows, pop_rows, both_rows]):
        pre_dissim = np.array(row["pre_dissim_bh_wa"])
        post_dissim = np.array(row["post_dissim_bh_wa"])
        Δdissim = (post_dissim - pre_dissim) / pre_dissim

        pre_pop_average_divergence = np.array(row["pre_population_average_divergence"])
        post_pop_average_divergence = np.array(
            row["post_population_average_divergence"]
        )
        Δpop_average_divergence = (
            post_pop_average_divergence - pre_pop_average_divergence
        ) / pre_pop_average_divergence

        results.loc[stat_index] = [
            np.median(Δdissim) * 100,
            np.mean(Δdissim) * 100,
            np.median(Δpop_average_divergence) * 100,
            np.mean(Δpop_average_divergence) * 100,
            np.median(pre_dissim),
            np.median(post_dissim),
            np.median(pre_pop_average_divergence),
            np.median(post_pop_average_divergence),
        ]
    results.index = ["dissim", "pop", "both"]
    return results


print("m = median; a = average; SDT = school decrease threshold")

df = pd.read_csv("data/results/results.csv")
print(compute_stats(df))
