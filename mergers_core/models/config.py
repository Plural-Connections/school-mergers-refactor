import os
import sys
import typing
import pandas as pd
import itertools
from collections import namedtuple


def _load_district_list(
    filename: str,
    school_count_lower_bound: typing.Optional[int] = None,  # inclusive
    district_count: typing.Optional[int] = None,
):
    df = pd.read_csv(filename, dtype={"id": str})
    if not min:
        df = df[::-1]

    if school_count_lower_bound and "num_schools" in df:
        df = df[df["num_schools"] >= school_count_lower_bound]

    if district_count:
        df = df.head(district_count)

    return df[["id", "state"]].itertuples(index=False, name="District")


# Each Config object will have a member for each of the keys in possible_configs with
# its value in possible_configs[key].
class Config:
    possible_configs = {
        "district": list(_load_district_list("data/top_200_districts.csv")),
        "school_increase_threshold": [0.1],
        "school_decrease_threshold": [0.2, 1.0],
        "dissimilarity_weight": [0, 1],
        "population_consistency_weight": [0, 1],
        "population_consistency_metric": [
            # "median",
            "average_difference",
            # "median_difference",
        ],
        "minimize": [True],
        "dissimilarity_flavor": ["bh_wa", "wnw"],
        "interdistrict": [False],
        "write_to_s3": [False],
    }

    def __init__(self, configs_file: str, entry_index: int):
        self.__dict__.update(
            pd.read_csv(
                configs_file,
                header=None,
                names=self.possible_configs.keys(),
                dtype=self.possible_configs,
            )
            .iloc[entry_index]
            .to_dict()
        )

    # Get a default config and override any elements you'd like.
    @classmethod
    def custom_config(cls, **kwargs):
        config = cls.__new__(cls)
        config.__dict__.update({k: v[0] for k, v in cls.possible_configs.items()})
        config.__dict__.update(kwargs)
        return config

    @staticmethod
    def district(state: str, district_id: str):
        return namedtuple("District", ["state", "id"])(state, district_id)

    def to_dict(self):
        result = self.__dict__.copy()
        for k in list(result.keys()):
            if k not in self.possible_configs:
                print(f"warning: {k} not a possible config", file=sys.stderr)
                result.pop(k)
        return result


def generate_all_configs():
    result = pd.DataFrame(
        itertools.product(*Config.possible_configs.values()),
        columns=Config.possible_configs.keys(),
    ).drop(["district_id", "state"])
    result = result.join(
        result["district_id_and_state"].apply(lambda x: pd.Series(x._asdict()))
    ).drop("district_id_and_state", axis=1)

    os.makedirs("data/sweep_configs", exist_ok=True)
    pd.to_csv(result, "data/sweep_configs/configs.csv")
