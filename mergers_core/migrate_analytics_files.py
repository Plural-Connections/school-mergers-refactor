import pandas as pd
import glob
from tqdm.contrib.concurrent import process_map


def _process(file):
    df = pd.read_csv(file, dtype={"district_id": str})
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
    df.to_csv(file, index=False)


files = glob.glob("../data/results/top-200/**/analytics.csv", recursive=True)
process_map(_process, files, chunksize=1, desc="migrate")
