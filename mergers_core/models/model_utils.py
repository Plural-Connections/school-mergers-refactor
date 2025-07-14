import mergers_core.utils.header as header
import mergers_core.models.constants as constants
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import os
import json
from pathlib import Path
import glob
import shutil

DF_BLOCKS = None
TRAVEL_TIMES = None


def estimate_travel_time_impacts(
    school_cluster_lists,
    df_current_grades,
    df_schools_in_play,
):
    category_columns = [col for col in DF_BLOCKS.keys() if col.startswith("num_")]

    # Compute status quo total driving times per category
    status_quo_total_driving_times_per_cat = Counter()
    for _, school_in_play in df_schools_in_play.iterrows():
        nces_id = school_in_play["NCESSCH"]
        df_blocks_school = DF_BLOCKS[DF_BLOCKS["ncessch"] == nces_id].reset_index(
            drop=True
        )
        for col in category_columns:
            for _, block in df_blocks_school.iterrows():
                driving_time = TRAVEL_TIMES[block["block_id"]][nces_id]
                if driving_time and not np.isnan(driving_time):
                    status_quo_total_driving_times_per_cat[
                        f"all_status_quo_time_{col}"
                    ] += (driving_time * block[col])

    # Compute how many students we expect to switch from a given school A to a given
    # school B, per category and compute the estimated travel times for those students
    num_students_switching_per_school_per_cat = defaultdict(dict)
    clusters = [c.split(", ") for c in school_cluster_lists]
    merged_schools = filter(lambda x: len(x) > 1, clusters)
    schools_serving_each_grade_per_cluster = defaultdict(dict)

    race_keys = list(constants.RACE_KEYS.values())
    for cluster in merged_schools:
        cluster_key = ", ".join(cluster)

        # Determine which grades are served by which schools in a given cluster
        for school in cluster:
            school_grades = df_current_grades[
                df_current_grades["NCESSCH"] == school
            ].iloc[0]
            for grade in constants.GRADE_TO_INDEX:
                if school_grades[grade]:
                    schools_serving_each_grade_per_cluster[cluster_key][grade] = school
        for school in cluster:
            school_enrollments = df_schools_in_play[
                df_schools_in_play["NCESSCH"] == school
            ].iloc[0]
            for grade in constants.GRADE_TO_INDEX:
                school_serving_grade = schools_serving_each_grade_per_cluster[
                    cluster_key
                ][grade]
                if school != school_serving_grade:
                    if (
                        school_serving_grade
                        not in num_students_switching_per_school_per_cat[school]
                    ):
                        num_students_switching_per_school_per_cat[school][
                            school_serving_grade
                        ] = Counter()
                    for race in race_keys:
                        num_students_switching_per_school_per_cat[school][
                            school_serving_grade
                        ][race] += school_enrollments[f"{race}_{grade}"]

    # Estimate changes in driving times for students switching schools
    current_total_switcher_driving_times = Counter()
    current_total_switcher_driving_times_per_school = defaultdict(Counter)
    new_total_switcher_driving_times = Counter()
    new_total_switcher_driving_times_per_school = defaultdict(Counter)
    for school_1, school_2_data in num_students_switching_per_school_per_cat.items():
        df_blocks_school = DF_BLOCKS[DF_BLOCKS["ncessch"] == school_1].reset_index(
            drop=True
        )
        for school_2, switcher_data in school_2_data.items():
            switchers_per_block_and_cat = defaultdict(Counter)
            for col in category_columns:
                # Determine est number of switchers per block (decimal students are fine)
                df_blocks_school[f"percent_of_total_{col}"] = (
                    df_blocks_school[col] / df_blocks_school[col].sum()
                )
                df_blocks_school = df_blocks_school.fillna(0)
                for _, block in df_blocks_school.iterrows():
                    switchers_per_block_and_cat[block["block_id"]][col] += (
                        block[f"percent_of_total_{col}"] * switcher_data[col]
                    )
                    # Estimate status quo and new travel times (# per block x travel time per block)
                    if TRAVEL_TIMES[block["block_id"]][school_1] and not np.isnan(
                        TRAVEL_TIMES[block["block_id"]][school_1]
                    ):
                        current_total_switcher_driving_times[
                            f"switcher_status_quo_time_{col}"
                        ] += (
                            switchers_per_block_and_cat[block["block_id"]][col]
                            * TRAVEL_TIMES[block["block_id"]][school_1]
                        )
                        current_total_switcher_driving_times_per_school[school_1][
                            f"switcher_status_quo_time_{col}"
                        ] += (
                            switchers_per_block_and_cat[block["block_id"]][col]
                            * TRAVEL_TIMES[block["block_id"]][school_1]
                        )
                    if TRAVEL_TIMES[block["block_id"]][school_2] and not np.isnan(
                        TRAVEL_TIMES[block["block_id"]][school_2]
                    ):
                        new_total_switcher_driving_times[
                            f"switcher_new_time_{col}"
                        ] += (
                            switchers_per_block_and_cat[block["block_id"]][col]
                            * TRAVEL_TIMES[block["block_id"]][school_2]
                        )
                        new_total_switcher_driving_times_per_school[school_1][
                            f"switcher_new_time_{col}"
                        ] += (
                            switchers_per_block_and_cat[block["block_id"]][col]
                            * TRAVEL_TIMES[block["block_id"]][school_2]
                        )

    return (
        status_quo_total_driving_times_per_cat,
        current_total_switcher_driving_times,
        new_total_switcher_driving_times,
        current_total_switcher_driving_times_per_school,
        new_total_switcher_driving_times_per_school,
    )


def check_solution_validity_and_compute_outcomes(
    df_mergers_g, df_grades, df_schools_in_play, state
):
    race_keys = list(constants.RACE_KEYS.values())
    df_mergers_curr = df_mergers_g.copy(deep=True)
    df_grades_curr = df_grades.copy(deep=True)

    # Make a dataframe based on which grades are offered by which schools
    school_cluster_lists = df_mergers_curr["school_cluster"].tolist()

    grades_served_per_cluster = defaultdict(set)
    school_clusters = defaultdict(list)
    for cluster in school_cluster_lists:
        schools = cluster.split(", ")
        for school in schools:
            school_clusters[school] = schools
            school_grades = df_grades_curr[df_grades_curr["NCESSCH"] == school].iloc[0]
            for grade in constants.GRADE_TO_INDEX:
                if school_grades[grade]:
                    grades_served_per_cluster[cluster].add(grade)

    num_per_cat_per_school = defaultdict(Counter)
    num_per_school_per_grade_per_cat = {}
    for school in school_clusters:
        school_grades = df_grades_curr[df_grades_curr["NCESSCH"] == school].iloc[0]
        num_per_school_per_grade_per_cat[school] = {r: Counter() for r in race_keys}
        for school_2 in school_clusters[school]:
            school_2_enrollments = df_schools_in_play[
                df_schools_in_play["NCESSCH"] == school_2
            ].iloc[0]
            for grade in constants.GRADE_TO_INDEX:
                if school_grades[grade]:
                    for race in race_keys:
                        num_per_cat_per_school[race][school] += school_2_enrollments[
                            f"{race}_{grade}"
                        ]
                        num_per_school_per_grade_per_cat[school][race][
                            grade
                        ] += school_2_enrollments[f"{race}_{grade}"]

    # Solution validity checking
    total_cols = [f"num_total_{g}" for g in constants.GRADE_TO_INDEX]
    total_students_dict = sum(num_per_cat_per_school["num_total"].values())
    total_students_df = df_schools_in_play[total_cols].sum(axis=1).sum()

    for cluster in grades_served_per_cluster:
        if len(grades_served_per_cluster[cluster]) != len(constants.GRADE_TO_INDEX):
            raise Exception(
                f"Only {len(grades_served_per_cluster[cluster])} of {len(constants.GRADE_TO_INDEX)} grades represented across cluster {cluster}"
            )

    if total_students_dict != total_students_df:
        raise Exception("All students not accounted for in re-assignment")

    for _, row in df_grades_curr.iterrows():
        curr_grade_seq = row[list(constants.GRADE_TO_INDEX.keys())].tolist()
        start_grade = None
        end_grade = None
        for i, g in enumerate(curr_grade_seq):
            if g and not start_grade:
                start_grade = g
                continue
            if start_grade and not g:
                end_grade = curr_grade_seq[i - 1]
                continue
            if start_grade and end_grade and g:
                raise Exception(
                    f"Grade levels schools are serving are not contiguous: {row['NCESSCH']}, {', '.join(curr_grade_seq)}"
                )

    # Compute dissimilarity values for white/non-white
    dissim_vals = []
    for school in school_clusters:
        dissim_vals.append(
            np.abs(
                (
                    num_per_cat_per_school["num_white"][school]
                    / sum(num_per_cat_per_school["num_white"].values())
                )
                - (
                    (
                        num_per_cat_per_school["num_total"][school]
                        - num_per_cat_per_school["num_white"][school]
                    )
                    / (
                        sum(num_per_cat_per_school["num_total"].values())
                        - sum(num_per_cat_per_school["num_white"].values())
                    )
                )
            )
        )

    dissim_val = 0.5 * np.sum(dissim_vals)

    # Compute dissimilarity values for black-hispanic and white-asian
    bh_wa_dissim_vals = []
    for school in school_clusters:
        bh_wa_dissim_vals.append(
            np.abs(
                (
                    (
                        num_per_cat_per_school["num_black"][school]
                        + num_per_cat_per_school["num_hispanic"][school]
                    )
                    / (
                        sum(num_per_cat_per_school["num_black"].values())
                        + sum(num_per_cat_per_school["num_hispanic"].values())
                    )
                )
                - (
                    (
                        num_per_cat_per_school["num_white"][school]
                        + num_per_cat_per_school["num_asian"][school]
                    )
                    / (
                        sum(num_per_cat_per_school["num_white"].values())
                        + sum(num_per_cat_per_school["num_asian"].values())
                    )
                )
            )
        )

    bh_wa_dissim_val = 0.5 * np.sum(bh_wa_dissim_vals)

    # Compute the number of students per group who will switch schools
    clusters = [c.split(", ") for c in school_cluster_lists]
    num_students_switching = {f"{r}_switched": 0 for r in race_keys}
    num_students_switching_per_school = {}
    num_total_students = {f"{r}_all": 0 for r in race_keys}
    schools = []
    for cluster in clusters:
        for school in cluster:
            schools.append(school)
            school_grades = df_grades_curr[df_grades_curr["NCESSCH"] == school].iloc[0]
            school_enrollments = df_schools_in_play[
                df_schools_in_play["NCESSCH"] == school
            ].iloc[0]
            num_students_switching_per_school[school] = {
                f"{r}_switched": 0 for r in race_keys
            }
            for grade in constants.GRADE_TO_INDEX:
                for race in race_keys:
                    num_total_students[f"{race}_all"] += school_enrollments[
                        f"{race}_{grade}"
                    ]
                    if not school_grades[grade]:
                        num_students_switching[
                            f"{race}_switched"
                        ] += school_enrollments[f"{race}_{grade}"]
                        num_students_switching_per_school[school][
                            f"{race}_switched"
                        ] += school_enrollments[f"{race}_{grade}"]

    # Estimate impacts on travel times
    (
        status_quo_total_driving_times_per_cat,
        status_quo_total_driving_times_for_switchers_per_cat,
        new_total_driving_times_for_switchers_per_cat,
        status_quo_total_driving_times_for_switchers_per_school_per_cat,
        new_total_driving_times_for_switchers_per_school_per_cat,
    ) = estimate_travel_time_impacts(
        school_cluster_lists,
        df_grades_curr,
        df_schools_in_play,
    )

    return (
        dissim_val,
        bh_wa_dissim_val,
        num_per_cat_per_school,
        num_per_school_per_grade_per_cat,
        num_total_students,
        num_students_switching,
        num_students_switching_per_school,
        status_quo_total_driving_times_per_cat,
        status_quo_total_driving_times_for_switchers_per_cat,
        new_total_driving_times_for_switchers_per_cat,
        status_quo_total_driving_times_for_switchers_per_school_per_cat,
        new_total_driving_times_for_switchers_per_school_per_cat,
    )


def maybe_load_large_files():
    global DF_BLOCKS, TRAVEL_TIMES
    if not DF_BLOCKS and not TRAVEL_TIMES:
        DF_BLOCKS = pd.read_csv(
            constants.BLOCKS_FILE.format(constants.STATE),
            dtype={"ncessch": str, "block_id": str},
        )


def output_solver_solution(
    solver,
    matches,
    grades_interval_binary,
    state,
    district_id,
    school_decrease_threshold,
    interdistrict,
    df_schools_in_play,
    output_dir,
    s3_bucket,
    write_to_s3,
    mergers_file_name,
    grades_served_file_name,
    schools_in_play_file_name,
    results_file_name="analytics.csv",
):
    maybe_load_large_files()

    # Extract solver variables
    match_data = {"school_1": [], "school_2": []}
    grades_served_data = {"NCESSCH": []} | {
        g: [] for g in constants.GRADE_TO_INDEX.keys()
    }
    for school in matches:
        for school_2 in matches[school]:
            val = solver.BooleanValue(matches[school][school_2])
            if val:
                match_data["school_1"].append(school)
                match_data["school_2"].append(school_2)

        grades_served_data["NCESSCH"].append(school)
        for idx, grade in enumerate(constants.GRADE_TO_INDEX.keys()):
            val = solver.BooleanValue(grades_interval_binary[school][idx])
            grades_served_data[grade].append(val)

    if write_to_s3:
        output_dir = s3_bucket + output_dir
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    df_mergers = pd.DataFrame(match_data)
    df_mergers_g = (
        df_mergers.groupby("school_1", as_index=False)
        .agg({"school_2": ", ".join})
        .drop_duplicates(subset="school_2")
        .drop(columns=["school_1"])
        .rename(columns={"school_2": "school_cluster"})
    )
    df_grades = pd.DataFrame(grades_served_data)
    df_mergers_g.to_csv(os.path.join(output_dir, mergers_file_name), index=False)
    df_grades.to_csv(os.path.join(output_dir, grades_served_file_name), index=False)
    df_schools_in_play.to_csv(
        os.path.join(output_dir, schools_in_play_file_name), index=False
    )

    # Compute pre/post dissim and other outcomes of interest
    try:
        df_mergers_pre = pd.DataFrame({"school_cluster": df_grades["NCESSCH"].tolist()})
        df_grades_pre = pd.DataFrame(
            {"NCESSCH": df_grades["NCESSCH"].tolist()}
            | {id: [True] * df_grades.shape[0] for id in constants.GRADE_TO_INDEX}
        )
        pre_dissim, pre_dissim_bh_wa, _, _, _, _, _, _, _, _, _, _ = (
            check_solution_validity_and_compute_outcomes(
                df_mergers_pre, df_grades_pre, df_schools_in_play, state
            )
        )

        (
            post_dissim,
            post_dissim_bh_wa,
            students_per_group_per_school_post_merger,
            students_per_grade_per_group_per_school_post_merger,
            num_total_students,
            num_students_switching,
            students_switching_per_group_per_school,
            status_quo_total_driving_times_per_cat,
            status_quo_total_driving_times_for_switchers_per_cat,
            new_total_driving_times_for_switchers_per_cat,
            status_quo_total_driving_times_for_switchers_per_school_per_cat,
            new_total_driving_times_for_switchers_per_school_per_cat,
        ) = check_solution_validity_and_compute_outcomes(
            df_mergers_g, df_grades, df_schools_in_play, state
        )

    except Exception as e:
        print(f"ERROR!!!! {e}")
        errors = {"error_message": str(e)}
        header.write_json(os.path.join(output_dir, "errors.json"), errors)
        return

    # Output results
    data_to_output = {
        "state": state,
        "district_id": district_id,
        "school_decrease_threshold": school_decrease_threshold,
        "interdistrict": bool(interdistrict),
        "pre_dissim": pre_dissim,
        "post_dissim": post_dissim,
        "pre_dissim_bh_wa": pre_dissim_bh_wa,
        "post_dissim_bh_wa": post_dissim_bh_wa,
    }
    data_to_output.update(num_total_students)
    data_to_output.update(num_students_switching)
    data_to_output.update(status_quo_total_driving_times_per_cat)
    data_to_output.update(status_quo_total_driving_times_for_switchers_per_cat)
    data_to_output.update(new_total_driving_times_for_switchers_per_cat)

    print(
        f"Pre dissim: {pre_dissim}\n",
        f"Post dissim: {post_dissim}\n",
        f"Pre bh-wa dissim: {pre_dissim_bh_wa}\n",
        f"Post bh-wa dissim: {post_dissim_bh_wa}\n",
    )
    try:
        print(
            f"Percent switchers: {num_students_switching['num_total_switched'] / num_total_students['num_total_all']}\n",
            f"SQ avg. travel time - all: {status_quo_total_driving_times_per_cat['all_status_quo_time_num_total'] / num_total_students['num_total_all']/ 60}\n",
            f"SQ avg. travel time - switchers: {status_quo_total_driving_times_for_switchers_per_cat['switcher_status_quo_time_num_total'] / num_students_switching['num_total_switched'] / 60}\n",
            f"New avg. travel time - switchers: {new_total_driving_times_for_switchers_per_cat['switcher_new_time_num_total'] / num_students_switching['num_total_switched'] / 60}\n",
        )
    except Exception as e:
        pass

    pd.DataFrame(data_to_output, index=[0]).to_csv(
        os.path.join(output_dir, results_file_name), index=False
    )

    variables_to_dump = [
        "students_switching_per_group_per_school",
        "students_per_group_per_school_post_merger",
        "students_per_grade_per_group_per_school_post_merger",
        "status_quo_total_driving_times_for_switchers_per_school_per_cat",
        "new_total_driving_times_for_switchers_per_school_per_cat",
    ]

    # Output number of students per race, per school
    if write_to_s3:
        from s3fs import S3FileSystem

        for variable in variables_to_dump:
            s3 = S3FileSystem()
            with s3.open(os.path.join(output_dir, variable + ".json"), "w") as file:
                json.dump(locals()[variable], file)
    else:
        for variable in variables_to_dump:
            with open(os.path.join(output_dir, variable + ".json"), "w") as file:
                json.dump(locals()[variable], file)


"""
Produces post solver / analysis files in case the jobs running on the server failed to produce some
"""


def produce_post_solver_files(
    df_mergers_g,
    df_grades,
    df_schools_in_play,
    state,
    district_id,
    school_decrease_threshold,
    interdistrict,
    output_dir,
    results_file_name="analytics.csv",
    students_switching_per_group_per_school_file="students_switching_per_group_per_school.json",
    students_per_group_per_school_post_merger_file="students_per_group_per_school_post_merger.json",
    students_per_grade_per_group_per_school_post_merger_file="students_per_grade_per_group_per_school_post_merger.json",
    status_quo_total_driving_times_for_switchers_per_school_per_cat_file="status_quo_total_driving_times_for_switchers_per_school_per_cat.json",
    new_total_driving_times_for_switchers_per_school_per_cat_file="new_total_driving_times_for_switchers_per_school_per_cat.json",
):
    # Compute pre/post dissim and other outcomes of interest
    try:
        pre_dissim, pre_dissim_bh_wa, _, _, _, _, _, _, _, _, _, _ = (
            check_solution_validity_and_compute_outcomes(
                df_mergers_g, df_grades, df_schools_in_play, state, pre_or_post="pre"
            )
        )

        (
            post_dissim,
            post_dissim_bh_wa,
            num_per_cat_per_school,
            num_per_school_per_grade_per_cat,
            num_total_students,
            num_students_switching,
            num_students_switching_per_school,
            status_quo_total_driving_times_per_cat,
            status_quo_total_driving_times_for_switchers_per_cat,
            new_total_driving_times_for_switchers_per_cat,
            status_quo_total_driving_times_for_switchers_per_school_per_cat,
            new_total_driving_times_for_switchers_per_school_per_cat,
        ) = check_solution_validity_and_compute_outcomes(
            df_mergers_g, df_grades, df_schools_in_play, state, pre_or_post="post"
        )

    except Exception as e:
        print(f"ERROR!!!! {e}")
        errors = {"error_message": str(e)}
        header.write_json(os.path.join(output_dir, "errors.json"), errors)
        return

    # Output results
    data_to_output = {
        "state": state,
        "district_id": district_id,
        "school_decrease_threshold": school_decrease_threshold,
        "interdistrict": bool(interdistrict),
        "pre_dissim": pre_dissim,
        "post_dissim": post_dissim,
        "pre_dissim_bh_wa": pre_dissim_bh_wa,
        "post_dissim_bh_wa": post_dissim_bh_wa,
    }
    data_to_output.update(num_total_students)
    data_to_output.update(num_students_switching)
    data_to_output.update(status_quo_total_driving_times_per_cat)
    data_to_output.update(status_quo_total_driving_times_for_switchers_per_cat)
    data_to_output.update(new_total_driving_times_for_switchers_per_cat)

    print(
        f"Pre dissim: {pre_dissim}\n",
        f"Post dissim: {post_dissim}\n",
        f"Pre bh-wa dissim: {pre_dissim_bh_wa}\n",
        f"Post bh-wa dissim: {post_dissim_bh_wa}\n",
    )
    try:
        print(
            f"Percent switchers: {num_students_switching['num_total_switched'] / num_total_students['num_total_all']}\n",
            f"SQ avg. travel time - all: {status_quo_total_driving_times_per_cat['all_status_quo_time_num_total']/ num_total_students['num_total_all']/ 60}\n",
            f"SQ avg. travel time - switchers: {status_quo_total_driving_times_for_switchers_per_cat['switcher_status_quo_time_num_total']/ num_students_switching['num_total_switched']/ 60}\n",
            f"New avg. travel time - switchers: {new_total_driving_times_for_switchers_per_cat['switcher_new_time_num_total']/num_students_switching['num_total_switched']/ 60}\n",
        )
    except Exception as e:
        pass

    try:
        pd.DataFrame(data_to_output, index=[0]).to_csv(
            os.path.join(output_dir, results_file_name), index=False
        )

        header.write_dict(
            os.path.join(output_dir, students_switching_per_group_per_school_file),
            num_students_switching_per_school,
        )

        header.write_dict(
            os.path.join(output_dir, students_per_group_per_school_post_merger_file),
            num_per_cat_per_school,
        )

        header.write_dict(
            os.path.join(
                output_dir, students_per_grade_per_group_per_school_post_merger_file
            ),
            num_per_school_per_grade_per_cat,
        )

        header.write_dict(
            os.path.join(
                output_dir,
                status_quo_total_driving_times_for_switchers_per_school_per_cat_file,
            ),
            status_quo_total_driving_times_for_switchers_per_school_per_cat,
        )

        header.write_dict(
            os.path.join(
                output_dir,
                new_total_driving_times_for_switchers_per_school_per_cat_file,
            ),
            new_total_driving_times_for_switchers_per_school_per_cat,
        )

    except Exception as e:
        pass


def produce_post_solver_files_parallel(
    batch="min_elem_4_interdistrict_bottom_sensitivity",
    solutions_dir="data/results/{}/",
):
    # Compute pre/post dissim and other outcomes of interest
    all_jobs = []
    for state in os.listdir(solutions_dir.format(batch)):
        if "consolidated" in state:
            continue
        for district_id in os.listdir(os.path.join(solutions_dir.format(batch), state)):
            try:
                curr_dir = os.path.join(
                    solutions_dir.format(batch),
                    state,
                    district_id,
                )
                soln_dirs = os.listdir(curr_dir)
                for dir in soln_dirs:
                    if ".html" in dir:
                        continue
                    this_dir = os.path.join(curr_dir, dir)
                    df_mergers_g = pd.read_csv(
                        glob.glob(
                            os.path.join(
                                this_dir,
                                "**/" + "school_mergers.csv",
                            ),
                            recursive=True,
                        )[0],
                        dtype=str,
                    )
                    df_grades = pd.read_csv(
                        glob.glob(
                            os.path.join(
                                this_dir,
                                "**/" + "grades_served.csv",
                            ),
                            recursive=True,
                        )[0],
                        dtype={"NCESSCH": str},
                    )
                    df_schools_in_play = pd.read_csv(
                        glob.glob(
                            os.path.join(
                                this_dir,
                                "**/" + "schools_in_play.csv",
                            ),
                            recursive=True,
                        )[0],
                        dtype={"NCESSCH": str},
                    )
                    this_dir_root = this_dir.split("/")[-1].split("_")
                    interdistrict = False if this_dir_root[0] == "False" else True
                    school_decrease_threshold = float(this_dir_root[1])

                    all_jobs.append(
                        (
                            df_mergers_g,
                            df_grades,
                            df_schools_in_play,
                            state,
                            district_id,
                            school_decrease_threshold,
                            interdistrict,
                            os.path.join(curr_dir, dir, ""),
                        )
                    )

            except Exception as e:
                print(f"Exception {state}, {district_id}, {e}")
                pass

    print("Starting parallel processing ...")
    from multiprocessing import Pool

    N_THREADS = 10
    p = Pool(N_THREADS)
    p.starmap(produce_post_solver_files, all_jobs)

    p.terminate()
    p.join()


def consolidate_results_files(
    batch="min_elem_4_interdistrict_bottom_sensitivity",
    batch_dir="data/results/{}",
    output_file="data/results/{}/consolidated_simulation_results_{}_{}_{}.csv",
):

    analytics_files = glob.glob(
        os.path.join(batch_dir.format(batch), "**/" + "analytics.csv"), recursive=True
    )
    all_dfs = []
    results_folder = []
    for i, f in enumerate(analytics_files):
        all_dfs.append(pd.read_csv(f, dtype={"NCESSCH": str, "district_id": str}))
        results_folder.append(f.split("/")[-2])

    df = pd.concat(all_dfs)
    df["results_folder"] = results_folder
    df_f = df.drop_duplicates(
        subset=["district_id", "school_decrease_threshold", "interdistrict"]
    )
    df_duplicates = df[
        ~df["results_folder"].isin(df_f["results_folder"].tolist())
    ].reset_index(drop=True)

    print("Num duplicates: ", len(df_duplicates))
    # Delete duplicate results' folders
    for i in range(len(df_duplicates)):
        shutil.rmtree(
            os.path.join(
                batch_dir.format(batch),
                df_duplicates["state"][i],
                df_duplicates["district_id"][i],
                df_duplicates["results_folder"][i],
            )
        )
    print("Num results: ", len(df_f))

    # Create different files for diff interdistrict, bottom constraint unique combos
    df_param_combos = df_f.groupby(
        ["school_decrease_threshold", "interdistrict"], as_index=False
    ).agg({"state": "count"})
    for i in range(len(df_param_combos)):
        school_decrease_threshold = df_param_combos["school_decrease_threshold"][i]
        interdistrict = df_param_combos["interdistrict"][i]
        df_curr = df_f[
            (df_f["school_decrease_threshold"] == school_decrease_threshold)
            & (df_f["interdistrict"] == interdistrict)
        ].reset_index(drop=True)
        df_curr.to_csv(
            output_file.format(batch, batch, school_decrease_threshold, interdistrict),
            index=False,
        )


def compare_batch_totals(
    batch_1="min_num_elem_schools_4_constrained",
    batch_2="min_num_elem_schools_4_bottomless",
    file_root="data/results/{}/consolidated_simulation_results_{}.csv",
):
    df_1 = pd.read_csv(file_root.format(batch_1, batch_1), dtype={"district_id": str})[
        ["district_id", "num_total_all"]
    ]
    df_2 = pd.read_csv(file_root.format(batch_2, batch_2), dtype={"district_id": str})[
        ["district_id", "num_total_all"]
    ].rename(columns={"num_total_all": "num_total_all_2"})
    df = pd.merge(df_1, df_2, on="district_id", how="inner")
    df["diff_totals"] = df["num_total_all"] != df["num_total_all_2"]
    print(df["diff_totals"].sum())
    df_diff = df[df["diff_totals"] == True].reset_index(drop=True)
    print(len(df_diff) / len(df))
    print(df_diff.head(10))


if __name__ == "__main__":
    produce_post_solver_files_parallel()
    consolidate_results_files()
