# coding: utf-8
import pandas as pd
import numpy as np

POPULATION_METRIC = "median_divergence"


def compute_stats(df):
    df = df.sort_values(by="district").reset_index(drop=True)
    dissim_rows = df[
        (df["dissimilarity_weight"] == 1) & (df["population_metric_weight"] == 0)
    ]
    pop_rows = df[
        (df["dissimilarity_weight"] == 0) & (df["population_metric_weight"] == 1)
    ]
    both_rows = df[
        (df["dissimilarity_weight"] == 1) & (df["population_metric_weight"] == 1)
    ]

    print(f"{dissim_rows.shape=}, {pop_rows.shape=}, {both_rows.shape=}")
    relevant_columns = [
        "district",
        "pre_dissim_bh_wa",
        "dissimilarity_weight",
        "population_metric_weight",
    ]
    print(df[relevant_columns][38:])
    print(dissim_rows[relevant_columns][13:])
    print(pop_rows[relevant_columns][13:])
    print(both_rows[relevant_columns][13:])
    print()

    mask = dissim_rows["pre_dissim_bh_wa"].reset_index(drop=True) != pop_rows[
        "pre_dissim_bh_wa"
    ].reset_index(drop=True)
    print(mask[12:181])
    print(mask[13:180].all())
    print(mask[:13].any() or mask[180:].any())
    mask.index = dissim_rows.index
    # print(dissim_rows[mask]["pre_dissim_bh_wa"])
    mask.index = pop_rows.index
    # print(pop_rows[mask]["pre_dissim_bh_wa"])

    results = pd.DataFrame(
        np.zeros((3, 8), dtype=np.float64),
        columns=[
            "m %Δdissim",
            "a %Δdissim",
            "m %Δpop (aΔ)",
            "a %Δpop (aΔ)",
            "prem dissim",
            "prea dissim",
            "prem pop (aΔ)",
            "prea pop (aΔ)",
        ],
    )
    for stat_index, row in enumerate([dissim_rows, pop_rows, both_rows]):
        pre_dissim = np.array(row["pre_dissim_bh_wa"])
        post_dissim = np.array(row["post_dissim_bh_wa"])
        Δdissim = (post_dissim - pre_dissim) / pre_dissim

        pre_pop_aΔ = np.array(row["pre_population_average_divergence"])
        post_pop_aΔ = np.array(row["post_population_average_divergence"])
        Δpop_average_divergence = (post_pop_aΔ - pre_pop_aΔ) / pre_pop_aΔ

        results.loc[stat_index] = [
            np.median(Δdissim) * 100,
            np.mean(Δdissim) * 100,
            np.median(Δpop_average_divergence) * 100,
            np.mean(Δpop_average_divergence) * 100,
            np.median(pre_dissim),
            np.mean(pre_dissim),
            np.median(pre_pop_aΔ),
            np.mean(pre_pop_aΔ),
        ]
    results.index = ["dissim", "pop", "both"]
    return results


df = pd.read_csv("data/results/results.csv")
print(compute_stats(df))
