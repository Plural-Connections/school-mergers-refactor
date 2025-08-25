# Run by consolidate.sh

import pandas as pd

df = pd.read_csv("${batchname}.csv", dtype={"district": str})
columns_to_deduplicate = [
    "district",
    "school_decrease_threshold",
    "dissimilarity_weight",
    "population_metric_weight",
    "population_metric",
    "dissimilarity_flavor",
]
print(df)
print(df[columns_to_deduplicate])

duplicated_mask = df.duplicated(subset=columns_to_deduplicate, keep="first")
print(duplicated_mask)
dropped_count = duplicated_mask.sum()

if dropped_count > 0:
    print(f"dropped {dropped_count} line(s):")
    dropped_keys = df[duplicated_mask][columns_to_deduplicate].drop_duplicates()
    print("Dropped keys:")
    print(dropped_keys)

df = df[~duplicated_mask]
df.to_csv("${batchname}.csv", index=False)
