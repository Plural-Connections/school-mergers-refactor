import os
import sys
import typing
import pandas as pd
import itertools
from collections import namedtuple


class District(namedtuple("District", ["state", "id"])):
    def __str__(self):
        return f"{self.state}-{self.id}"

    def __repr__(self):
        return f"District(state={self.state}, id={self.id})"

    @classmethod
    def from_string(cls, string: str):
        state, id = string.split("-")
        return cls(state=state, id=id)


def _load_district_list(
    filename: str,  # a csv file with a 'district' column
    school_count_lower_bound: typing.Optional[int] = None,  # inclusive
    district_count: typing.Optional[int] = None,
):

    df = pd.read_csv(
        filename,
        converters={"district": District.from_string},
        index_col="district",
    )

    if "num_schools" not in df:
        schools_per_district = pd.read_csv(
            "data/all_schools.csv",
            converters={"district": District.from_string},
            index_col="district",
        )
        df = df.join(schools_per_district).sort_values(by="num_schools")

    if school_count_lower_bound:
        df = df[df["num_schools"] >= school_count_lower_bound]

    if district_count:
        df = df.head(district_count)

    return df.index


# Each Config object will have a member for each of the keys in possible_configs with
# its value in possible_configs[key].
class Config:
    possible_configs = {
        "district": list(_load_district_list("data/top_200_districts.csv")),
        "school_increase_threshold": [0.1],
        "school_decrease_threshold": [1.0],
        "dissimilarity_weight": [0, 1],
        "population_metric_weight": [0, 1],
        "population_metric": [
            "average_divergence",
            # "median_divergence",
        ],
        "minimize": [True],
        "dissimilarity_flavor": ["wnw"],
        "interdistrict": [False],
        "write_to_s3": [False],
    }

    def __str__(self):
        return f"{self.__dict__}"

    def __repr__(self):
        return f"Config({self.__dict__!r})"

    def __init__(self, configs_file: str, entry_index: typing.Optional[int] = None):
        configs = pd.read_csv(
            configs_file, converters={"district": District.from_string}
        )
        if entry_index:
            config = configs.iloc[entry_index].to_dict()
        else:
            config = configs.sample(1).iloc[0].to_dict()
        self.__dict__.update(config)

    # Get a default config and override any elements you'd like.
    @classmethod
    def custom_config(cls, **kwargs):
        config = cls.__new__(cls)
        config.__dict__.update({k: v[0] for k, v in cls.possible_configs.items()})
        config.__dict__.update(kwargs)
        return config

    def to_dict(self):
        result = self.__dict__.copy()
        for k in list(result.keys()):
            if k not in self.possible_configs:
                print(f"warning: '{k}' is not a config key", file=sys.stderr)
                result.pop(k)
            if len(self.possible_configs[k]) == 1:
                result.pop(k)
        return result


def generate_all_configs(min_schools: typing.Optional[int] = 4):
    os.makedirs("data/sweep_configs", exist_ok=True)
    configs = pd.DataFrame(
        itertools.product(*Config.possible_configs.values()),
        columns=Config.possible_configs.keys(),
    )
    # must optimize for one metric
    configs = configs[
        (configs["dissimilarity_weight"] == 1)
        | (configs["population_metric_weight"] == 1)
    ]
    print(
        f"Generated {len(configs)} configs for "
        f"{len(Config.possible_configs["district"])} schools."
    )
    configs.to_csv("data/sweep_configs/configs.csv", index=False)
