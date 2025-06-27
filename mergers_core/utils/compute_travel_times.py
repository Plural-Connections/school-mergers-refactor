from mergers_core.utils.header import *
from mergers_core.utils.distances_and_times import (
    get_distance_for_coord_pair,
    get_travel_time_for_coord_pair,
)
import us


def compute_travel_time_matrices(
    state,
    input_file="data/solver_files/2122/{}/between_within_district_allowed_mergers.json",
    lat_longs_file="data/school_data/nces_21_22_lat_longs.csv",
    blocks_file="data/attendance_boundaries/2122/{}/blocks_to_elementary.csv",
    output_dir="data/travel_times_files/2122/{}/",
):
    school_ids = read_json(input_file.format(state))
    df_locs = pd.read_csv(lat_longs_file, dtype={"nces_id": str})
    df_blocks = pd.read_csv(
        blocks_file.format(state), dtype={"ncessch": str, "GEOID20": str}
    )

    travel_times = defaultdict(dict)
    for i, s in enumerate(school_ids):
        print(i / len(school_ids))
        curr_blocks = df_blocks[df_blocks["ncessch"] == s].reset_index(drop=True)
        for s2 in school_ids[s]:
            lat_long = df_locs[df_locs["nces_id"] == s2].iloc[0]
            for i in range(0, len(curr_blocks)):
                block_lat = curr_blocks["block_centroid_lat"][i]
                block_long = curr_blocks["block_centroid_long"][i]
                if not s2 in travel_times[curr_blocks["GEOID20"][i]]:
                    # travel_times[curr_blocks["GEOID20"][i]][
                    #     s2
                    # ] = get_distance_for_coord_pair(
                    #     [(block_lat, block_long), (lat_long["lat"], lat_long["long"])]
                    # )

                    travel_times[curr_blocks["GEOID20"][i]][s2] = (
                        get_travel_time_for_coord_pair(
                            [
                                (block_long, block_lat),
                                (lat_long["long"], lat_long["lat"]),
                            ]
                        )
                    )
                else:
                    continue

    Path(output_dir.format(state)).mkdir(parents=True, exist_ok=True)
    write_dict(
        os.path.join(output_dir.format(state), "block_to_school_driving_times.json"),
        travel_times,
    )


def compute_travel_time_matrices_parallel():
    N_THREADS = 10
    state_abbrev = []
    for s in us.states.STATES:
        state_abbrev.append(s.abbr)

    print("Starting parallel processing ...")
    print(len(state_abbrev))

    # state_abbrev = ["VA"]
    from multiprocessing import Pool

    p = Pool(N_THREADS)
    p.map(compute_travel_time_matrices, state_abbrev)

    p.terminate()
    p.join()


if __name__ == "__main__":
    # compute_travel_time_matrices("VA")
    compute_travel_time_matrices_parallel()
