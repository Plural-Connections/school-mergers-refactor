from mergers_core.utils.header import *
from mergers_core.utils.produce_files_for_solver import GRADE_KEYS
import us


def block_allocation(curr_school, curr_blocks, r, block_students_by_cat, cat_keys):
    total_students_to_allocate = curr_school[cat_keys[r]]

    key = cat_keys[r]
    perc_key = "curr_percent_{}".format(key)
    curr_blocks[perc_key] = curr_blocks[key] / curr_blocks[key].sum()
    curr_blocks = curr_blocks.sort_values(by=perc_key, ascending=False).reset_index()
    num_students_remaining = total_students_to_allocate

    # If we have a high fraction of the given category of
    # students at the school, it's possible our
    # census-based block-level counts are off.  So
    # instead, simply assume the number of students in that
    # category hailing from each block to the school
    # is proportional to the total number of students
    # we estimate to be living in that block
    perc_total_key = "percent_total_pop_in_block"
    curr_blocks[perc_total_key] = (
        curr_blocks["num_total"] / curr_blocks["num_total"].sum()
    )

    perc_key_to_use = perc_key
    if curr_school[r] > 0.5:
        perc_key_to_use = perc_total_key

    num_allocated = 0
    for i in range(0, len(curr_blocks)):
        if num_students_remaining <= 0:
            num_to_allocate = 0
        elif np.isnan(num_students_remaining):
            num_to_allocate = float("nan")
        else:
            num_to_allocate = int(
                min(
                    num_students_remaining,
                    np.ceil(
                        np.nan_to_num(
                            total_students_to_allocate * curr_blocks[perc_key_to_use][i]
                        )
                    ),
                )
            )
        block_students_by_cat[curr_blocks["block_id"][i]][
            cat_keys[r]
        ] += num_to_allocate
        num_allocated += num_to_allocate
        num_students_remaining -= num_to_allocate

    # print(r, total_students_to_allocate, num_allocated)
    return num_allocated


def allocate_students_to_blocks(curr_school, curr_blocks, all_cat_keys):
    block_students_by_cat = defaultdict(Counter)
    for k in all_cat_keys:
        block_allocation(
            curr_school,
            curr_blocks,
            k,
            block_students_by_cat,
            all_cat_keys,
        )

    return block_students_by_cat


def load_and_prep_block_racial_demos(
    block_demos_all_file,
    block_demos_over_18_file,
    census_block_mapping_file="/Users/ngillani/OneDrive - Northeastern University/neu/rezoning-schools/data/census_block_covariates/census_block_mapping_2020_2010.csv",
):
    print("\tLoad mapping file ...")
    # Map a given 2020 block to its highest-weighted block from 2010
    df_mapping = (
        pd.read_csv(census_block_mapping_file, encoding="ISO-8859-1", dtype=str)
        .sort_values(by="WEIGHT", ascending=False)
        .groupby(["GEOID20"])
        .agg({"GEOID10": "first"})
    )

    # Start with block demographics because these don't change regardless of the year of the attendance boundaries we are considering
    print("\tLoading block demos ...")
    df_block_demos_all = pd.read_csv(
        block_demos_all_file, encoding="ISO-8859-1", dtype=str
    )
    df_block_demos_over_18 = pd.read_csv(
        block_demos_over_18_file, encoding="ISO-8859-1", dtype=str
    )

    print("\tCreating block group ID and block ID fields ...")

    df_block_demos_all["block_id"] = df_block_demos_all[
        ["STATEA", "COUNTYA", "TRACTA", "BLOCKA"]
    ].agg("".join, axis=1)

    df_block_demos_all = pd.merge(
        df_block_demos_all,
        df_mapping,
        left_on="block_id",
        right_on="GEOID20",
        how="inner",
    )

    df_block_demos_all["block_group_id"] = df_block_demos_all["GEOID10"].apply(
        lambda x: x[:-3]
    )

    df_block_demos_over_18["block_id"] = df_block_demos_over_18[
        ["STATEA", "COUNTYA", "TRACTA", "BLOCKA"]
    ].agg("".join, axis=1)

    print(
        "\tStore number of kids under 18 (proxy for school-going aged children) per race, per block..."
    )

    df_block_demos = pd.merge(
        df_block_demos_all, df_block_demos_over_18, on="block_id", how="inner"
    )

    df_block_demos["num_total"] = pd.to_numeric(
        df_block_demos_all["U7B001"]
    ) - pd.to_numeric(df_block_demos_over_18["U7D001"])

    df_block_demos["num_white"] = pd.to_numeric(
        df_block_demos_all["U7B003"]
    ) - pd.to_numeric(df_block_demos_over_18["U7D003"])

    df_block_demos["num_black"] = pd.to_numeric(
        df_block_demos_all["U7B004"]
    ) - pd.to_numeric(df_block_demos_over_18["U7D004"])

    df_block_demos["num_native"] = pd.to_numeric(
        df_block_demos_all["U7B005"]
    ) - pd.to_numeric(df_block_demos_over_18["U7D005"])

    df_block_demos["num_asian"] = pd.to_numeric(
        df_block_demos_all["U7B006"]
    ) - pd.to_numeric(df_block_demos_over_18["U7D006"])

    df_block_demos["num_hispanic"] = pd.to_numeric(
        df_block_demos_all["U7C002"]
    ) - pd.to_numeric(df_block_demos_over_18["U7E002"])

    return df_block_demos


def output_block_level_census_data(
    state,
    input_file="data/attendance_boundaries/2122/{}/blocks_to_elementary.csv",
    block_demos_all_file_name="/Users/ngillani/OneDrive - Northeastern University/neu/rezoning-schools/data/census_block_covariates/2020_census_race_hisp_all_by_block/by_state/{}/racial_demos_by_block.csv",
    block_demos_over_18_file_name="/Users/ngillani/OneDrive - Northeastern University/neu/rezoning-schools/data/census_block_covariates/2020_census_race_hisp_over_18_by_block/by_state/{}/racial_demos_by_block.csv",
    output_file="data/attendance_boundaries/2122/{}/census_data_for_blocks_data.csv",
):
    print("\tLoading block demos for state {}...".format(state))
    df_block_demos = load_and_prep_block_racial_demos(
        block_demos_all_file_name.format(state),
        block_demos_over_18_file_name.format(state),
    )

    print("\tLoading zoned blocks for state {} ...".format(state))
    df_block_zones = pd.read_csv(input_file.format(state), dtype=str)

    print("\tJoining zoned blocks on block demos ...")
    df = pd.merge(
        df_block_demos,
        df_block_zones,
        left_on="block_id",
        right_on="GEOID20",
        how="inner",
    )

    print("\tSubsetting columns and outputting csv for {}".format(state))
    df = df[
        [
            "block_id",
            "block_group_id",
            "block_centroid_lat",
            "block_centroid_long",
            "ncessch",
            "num_total",
            "num_white",
            "num_black",
            "num_native",
            "num_asian",
            "num_hispanic",
        ]
    ]

    df.to_csv(output_file.format(state), index=False)


def output_block_level_census_data_parallel():
    N_THREADS = 10
    state_abbrev = []
    for s in us.states.STATES:
        state_abbrev.append(s.abbr)

    print("Starting parallel processing ...")
    print(len(state_abbrev))

    # state_abbrev = ["VA"]
    from multiprocessing import Pool

    p = Pool(N_THREADS)
    p.map(output_block_level_census_data, state_abbrev)

    p.terminate()
    p.join()


def estimate_students_per_block(
    state,
    input_file="data/attendance_boundaries/2122/{}/census_data_for_blocks_data.csv",
    input_schools_file="data/solver_files/2122/{}/school_enrollments.csv",
    output_file="data/attendance_boundaries/2122/{}/estimated_student_counts_per_block.csv",
    all_cat_keys={
        "perwht": "num_white",
        "perblk": "num_black",
        "perasn": "num_asian",
        "pernam": "num_native",
        "perhsp": "num_hispanic",
        "pertotal": "num_total",
    },
):
    print(state)
    df_blocks = pd.read_csv(
        input_file.format(state), encoding="ISO-8859-1", dtype={"block_id": str}
    )
    df_schools = pd.read_csv(
        input_schools_file.format(state), encoding="ISO-8859-1", dtype={"NCESSCH": str}
    )

    # Add columns in representing total number of students in school, and what percentage they account for
    for c in all_cat_keys:
        curr_cols = [k for k in df_schools.keys() if k.startswith(all_cat_keys[c])]
        df_schools[all_cat_keys[c]] = df_schools[curr_cols].sum(axis=1)

    for c in all_cat_keys:
        df_schools[c] = df_schools[all_cat_keys[c]] / df_schools["num_total"]

    block_students_by_cat = {}

    for i in range(0, len(df_schools)):
        # print(i / len(df_schools))
        curr_school = df_schools.iloc[i]

        curr_blocks = df_blocks[
            df_blocks["ncessch"] == curr_school["NCESSCH"]
        ].reset_index(drop=True)

        blocks_for_curr_school = allocate_students_to_blocks(
            curr_school, curr_blocks, all_cat_keys
        )
        block_students_by_cat.update(blocks_for_curr_school)

        # Check to make sure the sum of students per category from each block is always <= the school's total enrollment
        for val in all_cat_keys.values():
            cat_total_to_school = 0
            school_total = 0
            for b in blocks_for_curr_school:
                cat_total_to_school += blocks_for_curr_school[b][val]
                school_total += blocks_for_curr_school[b]["num_total"]

            # print(curr_school["NCESSCH"], val, cat_total_to_school, school_total)
            assert np.isnan(cat_total_to_school) or cat_total_to_school <= school_total

    # Initialize dict and create dataframe
    data_allocations = {cat: [] for cat in all_cat_keys.values()}
    data_allocations["block_id"] = []

    for b in block_students_by_cat:
        data_allocations["block_id"].append(b)
        for cat in all_cat_keys.values():
            data_allocations[cat].append(block_students_by_cat[b][cat])

    df_allocations = pd.DataFrame(data=data_allocations)
    df_blocks = df_blocks[
        df_blocks["block_id"].isin(list(block_students_by_cat.keys()))
    ].reset_index(drop=True)[
        ["block_id", "block_centroid_lat", "block_centroid_long", "ncessch"]
    ]
    df_blocks = pd.merge(df_blocks, df_allocations, how="left", on="block_id")

    df_blocks.to_csv(output_file.format(state), index=False)


def estimate_students_per_block_parallel():
    N_THREADS = 10
    state_abbrev = []
    for s in us.states.STATES:
        state_abbrev.append(s.abbr)

    print("Starting parallel processing ...")
    print(len(state_abbrev))

    # state_abbrev = ["VA"]
    from multiprocessing import Pool

    p = Pool(N_THREADS)
    p.map(estimate_students_per_block, state_abbrev)

    p.terminate()
    p.join()


if __name__ == "__main__":
    # output_block_level_census_data_parallel()
    estimate_students_per_block_parallel()
