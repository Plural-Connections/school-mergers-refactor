import pandas as pd
import glob

files = glob.glob("data/results/top-200/**/analytics.csv", recursive=True)

for f in files:
    print(f"Processing {f}")
    df = pd.read_csv(f)
    df["district"] = df["state"].astype(str) + "-" + df["district_id"].astype(str)
    df["population_metric"] = df["population_consistency_metric"]
    df["population_metric_weight"] = df["population_consistency_weight"]
    df.drop(
        columns=[
            "state",
            "district_id",
            "population_consistency_metric",
            "population_consistency_weight",
        ],
        inplace=True,
    )
    df.to_csv(f, index=False)
