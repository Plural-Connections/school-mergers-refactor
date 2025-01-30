from mergers_core.utils.header import *

from mergers_core.models.merge_cp_sat import solve_and_output_results
from mergers_core.models.constants import MAX_SOLVER_TIME


def generate_year_state_sweep_configs(
    districts_to_process_file="data/school_data/all_districts.csv",
    max_cluster_node_time=43200,
    total_cluster_tasks_per_group=500,
    min_elem_schools=4,
    batch_root="min_num_elem_{}_constrained_bh_wa",
    # dists_to_remove="data/results/min_num_elem_schools_4_bottomless/consolidated_simulation_results_min_num_elem_schools_4_bottomless.csv",
    dists_to_remove=None,
    output_dir="data/sweep_configs/{}/",
):
    interdistrict = [False]
    objective = ["bh_wa"]
    school_decrease_threshold = [0.2]
    batch = batch_root.format(min_elem_schools)
    write_to_s3 = True

    # TODO: comment out for entire state or country-level sims
    df_d = pd.read_csv(districts_to_process_file, dtype={"district_id": str})
    df_d = df_d[df_d["num_schools"] >= min_elem_schools].reset_index(drop=True)

    if dists_to_remove:
        df_r = pd.read_csv(dists_to_remove, dtype={"district_id": str})
        df_d = df_d[~df_d["district_id"].isin(df_r["district_id"])].reset_index(
            drop=True
        )
    sweeps = {
        "state": [],
        "district_id": [],
        "school_decrease_threshold": [],
        "interdistrict": [],
        "objective": [],
        "batch": [],
        "write_to_s3": [],
    }

    # Make directory for simulation sweeps
    Path(output_dir.format(batch)).mkdir(parents=True, exist_ok=True)

    # Iterate through params for sweeps
    for i in range(0, len(df_d)):
        for d in interdistrict:
            for t in school_decrease_threshold:
                for o in objective:
                    sweeps["state"].append(df_d["state"][i])
                    sweeps["district_id"].append(df_d["district_id"][i])
                    sweeps["school_decrease_threshold"].append(t)
                    sweeps["interdistrict"].append(d)
                    sweeps["objective"].append(o)
                    sweeps["batch"].append(batch)
                    sweeps["write_to_s3"].append(write_to_s3)

    # Create groups to run on cluster
    df_out = pd.DataFrame(data=sweeps).sample(frac=1)
    num_jobs_per_group = int(
        np.floor(max_cluster_node_time / MAX_SOLVER_TIME)
        * total_cluster_tasks_per_group
    )
    num_cluster_groups = int(np.ceil(len(df_out) / num_jobs_per_group))
    for i in range(0, num_cluster_groups):
        df_curr = df_out.iloc[(i * num_jobs_per_group) : ((i + 1) * num_jobs_per_group)]
        df_curr.to_csv(output_dir.format(batch) + str(i) + ".csv", index=False)


def run_sweep_for_chunk(
    chunk_ID,
    num_total_chunks,
    group_ID,
    solver_function=solve_and_output_results,
    sweeps_dir="data/sweep_configs/min_num_elem_4_constrained_bh_wa/",
):
    df = pd.read_csv(sweeps_dir + str(group_ID) + ".csv", dtype={"district_id": str})

    configs = []
    for i in range(0, len(df)):
        config_dict = {
            "state": df["state"][i],
            "district_id": df["district_id"][i],
            "school_decrease_threshold": df["school_decrease_threshold"][i],
            "interdistrict": df["interdistrict"][i],
            "batch": df["batch"][i],
            "write_to_s3": df["write_to_s3"][i],
        }
        if "objective" in df.columns:
            config_dict["objective"] = df["objective"][i]

        configs.append(config_dict)

    remainder = len(df) % num_total_chunks
    chunk_size = max(1, int(np.floor(len(df) / num_total_chunks)))
    if remainder > 0 and len(df) > num_total_chunks:
        chunk_size += 1

    configs_to_compute = configs[
        (chunk_ID * chunk_size) : ((chunk_ID + 1) * chunk_size)
    ]

    for curr_config in configs_to_compute:
        try:
            solver_function(**curr_config)
        except Exception as e:
            print(e)
            pass


if __name__ == "__main__":
    # generate_year_state_sweep_configs()
    if len(sys.argv) < 4:
        print(
            "Error: need to specify <chunk_ID>, <num_chunks>, and <group_id> for cluster run"
        )
        exit()
    run_sweep_for_chunk(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
