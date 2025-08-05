import pandas as pd
import sys
from pathlib import Path
import itertools
import os

from mergers_core.models.merge_cp_sat import solve_and_output_results


CONFIG_ENTRY_TYPES = {
    "district_id": str,
    "state": str,
    "school_decrease_threshold": float,
    "school_increase_threshold": float,
    "dissimilarity_weight": int,
    "population_consistency_weight": int,
    "population_consistency_metric": str,
    "minimize": bool,
    "dissimilarity_flavor": str,
    "interdistrict": bool,
    "write_to_s3": bool,
}


def _get_district_and_state_ids(
    districts_to_process_file, min_elem_schools, dists_to_remove, n_schools
):
    df_districts = pd.read_csv(districts_to_process_file, dtype={"district_id": str})

    if min_elem_schools:
        df_districts = df_districts[
            df_districts["num_schools"] >= min_elem_schools
        ].reset_index(drop=True)

    if dists_to_remove:
        df_removed = pd.read_csv(dists_to_remove, dtype={"district_id": str})
        df_districts = df_districts[
            ~df_districts["district_id"].isin(df_removed["district_id"])
        ].reset_index(drop=True)

    sorted_districts = df_districts.sort_values(by="num_schools", ascending=False)
    if not n_schools:
        return sorted_districts[["district_id", "state"]]
    return sorted_districts.head(n_schools)[["district_id", "state"]]


def generate_year_state_sweep_configs(
    districts_to_process_file=os.path.join(
        "data", "solver_files", "2122", "out_sorted_states.csv"
    ),
    # One of these options is respected. If they're both None, a different file is used.
    min_schools=None,  # Districts with strictly fewer schools are not processed
    n_districts=None,  # The n districts with the most schools are chosen
    batch_root="min_elem_{}_{}_{}",
    dists_to_remove=None,
    output_dir=os.path.join("data", "sweep_configs", "{}"),
):
    # exclude parameters from the itertools.product shenanigans
    exclude = list(locals().keys())
    exclude.append("exclude")

    if not min_schools and not n_districts:
        district_id_and_state = (
            pd.read_csv(
                os.path.join("data", "top_200_districts.csv"),
                dtype={"district_id": str},
            )
            .sample(50)
            .itertuples()
        )
    else:
        district_id_and_state = _get_district_and_state_ids(
            districts_to_process_file, min_schools, dists_to_remove, n_districts
        ).itertuples()

    school_increase_threshold = [0.1]
    school_decrease_threshold = [0.2, 1.0]
    dissimilarity_weight = [0, 1]
    population_consistency_weight = [0, 1]
    population_consistency_metric = [
        # "median",
        "average_difference",
        # "median_difference",
    ]
    minimize = [True]

    dissimilarity_flavor = ["bh_wa", "wnw"]
    interdistrict = [False]
    write_to_s3 = [False]

    to_product = {key: value for key, value in locals().items() if key not in exclude}
    configurations = pd.DataFrame(
        itertools.product(*to_product.values()),
        columns=to_product.keys(),
    )

    configurations["district_id"] = [
        t.district_id for t in configurations["district_id_and_state"]
    ]
    configurations["state"] = [t.state for t in configurations["district_id_and_state"]]

    configurations = configurations.drop("district_id_and_state", axis=1)[
        ~(
            (configurations["population_consistency_weight"] == 0)
            & (configurations["dissimilarity_weight"] == 0)
        )
    ]

    configurations["batch"] = [
        batch_root.format(
            min_schools or n_districts or "top-200",
            row.dissimilarity_flavor,
            row.population_consistency_metric,
        )
        for row in configurations.itertuples()
    ]

    print(f"Generated {len(configurations)} configurations.")

    for batch_name in configurations["batch"].unique():
        output_path = Path(output_dir.format(batch_name))
        output_path.mkdir(parents=True, exist_ok=True)

        batch_configurations = configurations[
            configurations["batch"] == batch_name
        ].drop("batch", axis=1)

        batch_configurations[list(CONFIG_ENTRY_TYPES.keys())].to_csv(
            output_path / "configs.csv", index=False, header=False
        )


def run_entry(
    entry_index: int,
    configs_file: str,
    batch_name: str,
    solver_function=solve_and_output_results,
):
    config = pd.read_csv(
        configs_file,
        header=None,
        names=CONFIG_ENTRY_TYPES.keys(),
        dtype=CONFIG_ENTRY_TYPES,
    )[entry_index].to_dict()

    config["batch"] = batch_name

    try:
        solver_function(**config)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python simulation_sweeps.py <entry_index> <configs_file> <batch_name>"
        )
        sys.exit(1)
    run_entry(int(sys.argv[1]), *sys.argv[2:])
