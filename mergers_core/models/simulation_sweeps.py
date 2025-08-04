import pandas as pd
import sys
import numpy as np
from pathlib import Path
import itertools
import os

from mergers_core.models.merge_cp_sat import solve_and_output_results
from mergers_core.models.constants import MAX_SOLVER_TIME


def _get_district_and_state_ids(
    districts_to_process_file, min_elem_schools, dists_to_remove, n_schools
):
    df_districts = pd.read_csv(districts_to_process_file, dtype={"district_id": str})
    df_districts = df_districts[
        df_districts["num_schools"] >= min_elem_schools
    ].reset_index(drop=True)

    if dists_to_remove:
        df_removed = pd.read_csv(dists_to_remove, dtype={"district_id": str})
        df_districts = df_districts[
            ~df_districts["district_id"].isin(df_removed["district_id"])
        ].reset_index(drop=True)

    return df_districts.sort_values(by="num_schools").head(n_schools)[
        ["district_id", "state"]
    ]


def generate_year_state_sweep_configs(
    districts_to_process_file=os.path.join(
        "data", "solver_files", "2122", "out_sorted_states.csv"
    ),
    max_cluster_node_time=43200,
    total_cluster_tasks_per_group=500,
    min_schools=6,  # Districts with fewer schools are not processed
    n_schools=120,
    batch_root="min_elem_{}_constrained_{}_{}",
    dists_to_remove=None,
    output_dir=os.path.join("data", "sweep_configs", "{}"),
):
    # exclude parameters from the itertools.product shenanigans
    exclude = list(locals().keys())
    exclude.append("exclude")

    district_id_and_state = _get_district_and_state_ids(
        districts_to_process_file, min_schools, dists_to_remove, n_schools
    ).itertuples()
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
    configurations = configurations.drop("district_id_and_state", axis=1)

    configurations = configurations[
        ~(
            (configurations["population_consistency_weight"] == 0)
            & (configurations["dissimilarity_weight"] == 0)
        )
    ]

    configurations["batch"] = [
        batch_root.format(
            min_schools,
            row.dissimilarity_flavor,
            row.population_consistency_metric,
        )
        for row in configurations.itertuples()
    ]

    print(f"Generated {len(configurations)} configurations.")

    num_jobs_per_group = int(
        np.floor(max_cluster_node_time / MAX_SOLVER_TIME)
        * total_cluster_tasks_per_group
    )
    num_cluster_groups_per_batch = int(
        np.ceil(
            len(configurations)
            / len(configurations["batch"].unique())
            / num_jobs_per_group
        )
    )
    print(
        f"{num_jobs_per_group} jobs per group, making {num_cluster_groups_per_batch} groups per batch."
    )

    for batch_name in configurations["batch"].unique():
        output_path = Path(output_dir.format(batch_name))
        output_path.mkdir(parents=True, exist_ok=True)

        batch_configurations = configurations[
            configurations["batch"] == batch_name
        ].drop("batch", axis=1)

        for i in range(num_cluster_groups_per_batch):
            df_chunk = batch_configurations.iloc[
                i * num_jobs_per_group : (i + 1) * num_jobs_per_group
            ]
            df_chunk.to_csv(output_path / f"{i}.csv", index=False)


def run_sweep_for_chunk(
    chunk_id,
    num_total_chunks,
    group_id,
    batch_name,
    solver_function=solve_and_output_results,
):
    sweeps_dir = os.path.join("data", "sweep_configs", batch_name)

    df_configs = pd.read_csv(
        os.path.join(sweeps_dir, f"{group_id}.csv"),
        dtype={
            "district_id": str,
            "state": str,
            "school_decrease_threshold": float,
            "dissimilarity_weight": int,
            "population_consistency_weight": int,
            "population_consistency_metric": str,
            "minimize": bool,
            "dissimilarity_flavor": str,
            "interdistrict": bool,
            "write_to_s3": bool,
        },
    )

    configs = []
    for _, row in df_configs.iterrows():
        config_dict = row.to_dict()
        configs.append(config_dict)

    chunk_size = len(df_configs) // num_total_chunks
    remainder = len(df_configs) % num_total_chunks

    start_index = chunk_id * chunk_size + min(chunk_id, remainder)
    end_index = start_index + chunk_size + (1 if chunk_id < remainder else 0)

    configs_to_compute = configs[start_index:end_index]

    for config in configs_to_compute:
        config["batch"] = batch_name

        try:
            solver_function(**config)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(
            "Usage: python simulation_sweeps.py <chunk_id> <num_total_chunks> <group_id> <batch_name>"
        )
        sys.exit(1)
    run_sweep_for_chunk(
        int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    )
