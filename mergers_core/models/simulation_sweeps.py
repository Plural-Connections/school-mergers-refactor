import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

from mergers_core.models.merge_cp_sat import solve_and_output_results
from mergers_core.models.constants import MAX_SOLVER_TIME


def generate_year_state_sweep_configs(
    districts_to_process_file=os.path.join("data", "school_data", "all_districts.csv"),
    max_cluster_node_time=43200,
    total_cluster_tasks_per_group=500,
    min_elem_schools=4,
    batch_root="min_num_elem_{}_constrained_bh_wa",
    dists_to_remove=None,
    output_dir=os.path.join("data", "sweep_configs", "{}"),
):
    interdistrict_options = [False]
    objective_options = ["bh_wa"]
    school_decrease_threshold_options = [0.2]
    batch_name = batch_root.format(min_elem_schools)
    write_to_s3 = True

    df_districts = pd.read_csv(districts_to_process_file, dtype={"district_id": str})
    df_districts = df_districts[
        df_districts["num_schools"] >= min_elem_schools
    ].reset_index(drop=True)

    if dists_to_remove:
        df_removed = pd.read_csv(dists_to_remove, dtype={"district_id": str})
        df_districts = df_districts[
            ~df_districts["district_id"].isin(df_removed["district_id"])
        ].reset_index(drop=True)

    sweep_configs = {
        "state": [],
        "district_id": [],
        "school_decrease_threshold": [],
        "interdistrict": [],
        "objective": [],
        "batch": [],
        "write_to_s3": [],
    }

    output_path = Path(output_dir.format(batch_name))
    output_path.mkdir(parents=True, exist_ok=True)

    for _, district in df_districts.iterrows():
        for interdistrict in interdistrict_options:
            for threshold in school_decrease_threshold_options:
                for objective in objective_options:
                    sweep_configs["state"].append(district["state"])
                    sweep_configs["district_id"].append(district["district_id"])
                    sweep_configs["school_decrease_threshold"].append(threshold)
                    sweep_configs["interdistrict"].append(interdistrict)
                    sweep_configs["objective"].append(objective)
                    sweep_configs["batch"].append(batch_name)
                    sweep_configs["write_to_s3"].append(write_to_s3)

    df_out = pd.DataFrame(data=sweep_configs).sample(frac=1)
    num_jobs_per_group = int(
        np.floor(max_cluster_node_time / MAX_SOLVER_TIME)
        * total_cluster_tasks_per_group
    )
    num_cluster_groups = int(np.ceil(len(df_out) / num_jobs_per_group))
    for i in range(num_cluster_groups):
        df_chunk = df_out.iloc[i * num_jobs_per_group : (i + 1) * num_jobs_per_group]
        df_chunk.to_csv(output_path / f"{i}.csv", index=False)


def run_sweep_for_chunk(
    chunk_id,
    num_total_chunks,
    group_id,
    solver_function=solve_and_output_results,
    sweeps_dir=os.path.join(
        "data", "sweep_configs", "min_num_elem_4_constrained_bh_wa"
    ),
):
    df_configs = pd.read_csv(
        os.path.join(sweeps_dir, f"{group_id}.csv"), dtype={"district_id": str}
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
        try:
            solver_function(**config)
        except Exception as e:
            print(e)
            pass


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Error: need to specify <chunk_ID>, <num_chunks>, and <group_id> for cluster run"
        )
        sys.exit(1)
    run_sweep_for_chunk(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
