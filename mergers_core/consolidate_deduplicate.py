# Run by consolidate.sh

import pandas as pd


df = pd.read_csv("results.csv", dtype={"district": str})
columns_to_deduplicate = [
    "district",
    # "school_decrease_threshold",
    "dissimilarity_weight",
    "population_metric_weight",
    # "population_metric",
    # "dissimilarity_flavor",
]

duplicated_mask = df.duplicated(subset=columns_to_deduplicate, keep="first")
dropped_count = duplicated_mask.sum()

if dropped_count > 0:
    print(f"dropped {dropped_count} duplicates(s):")
    dropped_keys = df[duplicated_mask][columns_to_deduplicate].drop_duplicates()
    print(dropped_keys)
else:
    print("no duplicates")

df = df[~duplicated_mask]
df.to_csv("results.csv", index=False)
