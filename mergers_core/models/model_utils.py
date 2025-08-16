import mergers_core.utils.header as header
import traceback
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


def _calculate_status_quo_driving_times(df_schools_in_play, category_columns):
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
    return status_quo_total_driving_times_per_cat


def _calculate_student_switching_data(
    school_cluster_lists, df_current_grades, df_schools_in_play
):
    num_students_switching_per_school_per_cat = defaultdict(dict)
    clusters = [c.split(", ") for c in school_cluster_lists]
    merged_schools = filter(lambda x: len(x) > 1, clusters)
    schools_serving_each_grade_per_cluster = defaultdict(dict)
    race_keys = list(constants.RACE_KEYS.values())

    for cluster in merged_schools:
        cluster_key = ", ".join(cluster)
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
                if school == school_serving_grade:
                    continue

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
    return num_students_switching_per_school_per_cat


def _estimate_switcher_driving_times(
    num_students_switching_per_school_per_cat, category_columns
):
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
                df_blocks_school[f"percent_of_total_{col}"] = (
                    df_blocks_school[col] / df_blocks_school[col].sum()
                )
                df_blocks_school = df_blocks_school.fillna(0)
                for _, block in df_blocks_school.iterrows():
                    switchers_per_block_and_cat[block["block_id"]][col] += (
                        block[f"percent_of_total_{col}"] * switcher_data[col]
                    )
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
        current_total_switcher_driving_times,
        current_total_switcher_driving_times_per_school,
        new_total_switcher_driving_times,
        new_total_switcher_driving_times_per_school,
    )


def estimate_travel_time_impacts(
    school_cluster_lists,
    df_current_grades,
    df_schools_in_play,
):
    category_columns = [col for col in DF_BLOCKS.keys() if col.startswith("num_")]

    status_quo_total_driving_times_per_cat = _calculate_status_quo_driving_times(
        df_schools_in_play, category_columns
    )

    num_students_switching_per_school_per_cat = _calculate_student_switching_data(
        school_cluster_lists, df_current_grades, df_schools_in_play
    )

    (
        current_total_switcher_driving_times,
        current_total_switcher_driving_times_per_school,
        new_total_switcher_driving_times,
        new_total_switcher_driving_times_per_school,
    ) = _estimate_switcher_driving_times(
        num_students_switching_per_school_per_cat, category_columns
    )

    return {
        "status_quo_total_driving_times_per_cat": status_quo_total_driving_times_per_cat,
        "current_total_switcher_driving_times": current_total_switcher_driving_times,
        "new_total_switcher_driving_times": new_total_switcher_driving_times,
        "current_total_switcher_driving_times_per_school": current_total_switcher_driving_times_per_school,
        "new_total_switcher_driving_times_per_school": new_total_switcher_driving_times_per_school,
    }


def _calculate_student_distributions(school_clusters, df_grades, df_schools_in_play):
    race_keys = list(constants.RACE_KEYS.values())
    num_per_cat_per_school = defaultdict(Counter)
    num_per_school_per_grade_per_cat = {}

    for school in school_clusters:
        school_grades = df_grades[df_grades["NCESSCH"] == school].iloc[0]
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
    return num_per_cat_per_school, num_per_school_per_grade_per_cat


def _validate_solution(
    grades_served_per_cluster, num_per_cat_per_school, df_schools_in_play, df_grades
):
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

    for _, row in df_grades.iterrows():
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
                    f"Grade levels schools are serving are not contiguous: {row['NCESSCH']}, {', '.join(map(str, curr_grade_seq))}"
                )


def compute_dissimilarity_metrics(school_clusters, num_per_cat_per_school):
    dissim_vals = []
    total_white = sum(num_per_cat_per_school["num_white"].values())
    total_students = sum(num_per_cat_per_school["num_total"].values())
    total_non_white = total_students - total_white

    if total_white == 0 or total_non_white == 0:
        dissim_val = 0
    else:
        for school in school_clusters:
            dissim_vals.append(
                np.abs(
                    (num_per_cat_per_school["num_white"][school] / total_white)
                    - (
                        (
                            num_per_cat_per_school["num_total"][school]
                            - num_per_cat_per_school["num_white"][school]
                        )
                        / total_non_white
                    )
                )
            )
        dissim_val = 0.5 * np.sum(dissim_vals)

    bh_wa_dissim_vals = []
    total_black_hispanic = sum(num_per_cat_per_school["num_black"].values()) + sum(
        num_per_cat_per_school["num_hispanic"].values()
    )
    total_white_asian = sum(num_per_cat_per_school["num_white"].values()) + sum(
        num_per_cat_per_school["num_asian"].values()
    )

    if total_black_hispanic == 0 or total_white_asian == 0:
        bh_wa_dissim_val = 0
    else:
        for school in school_clusters:
            bh_wa_dissim_vals.append(
                np.abs(
                    (
                        (
                            num_per_cat_per_school["num_black"][school]
                            + num_per_cat_per_school["num_hispanic"][school]
                        )
                        / total_black_hispanic
                    )
                    - (
                        (
                            num_per_cat_per_school["num_white"][school]
                            + num_per_cat_per_school["num_asian"][school]
                        )
                        / total_white_asian
                    )
                )
            )
        bh_wa_dissim_val = 0.5 * np.sum(bh_wa_dissim_vals)

    return dissim_val, bh_wa_dissim_val


def compute_population_metrics(df_schools_in_play, num_per_cat_per_school):
    school_capacities = df_schools_in_play.set_index("NCESSCH")[
        "student_capacity"
    ].to_dict()
    school_populations = num_per_cat_per_school["num_total"]
    percentages = {}
    for school, population in school_populations.items():
        capacity = school_capacities.get(school)
        if capacity and capacity > 0:
            percentages[school] = population / capacity

    if not percentages:
        return {"median": 0, "average_difference": 0, "median_difference": 0}

    list_percentages = list(percentages.values())
    average_percentage = np.mean(list_percentages)
    differences = [np.abs(p - average_percentage) for p in list_percentages]
    average_difference = np.mean(differences)

    return {
        "median": np.median(list_percentages),
        "average_difference": average_difference,
        "median_difference": np.median(differences),
    }


def _count_switching_students(school_cluster_lists, df_grades, df_schools_in_play):
    race_keys = list(constants.RACE_KEYS.values())
    clusters = [c.split(", ") for c in school_cluster_lists]
    num_students_switching = {f"{r}_switched": 0 for r in race_keys}
    num_students_switching_per_school = {}
    num_total_students = {f"{r}_all": 0 for r in race_keys}

    for cluster in clusters:
        for school in cluster:
            school_grades = df_grades[df_grades["NCESSCH"] == school].iloc[0]
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
    return (
        num_total_students,
        num_students_switching,
        num_students_switching_per_school,
    )


def check_solution_validity_and_compute_outcomes(
    df_mergers_, df_grades_, df_schools_in_play, state
):
    df_mergers = df_mergers_.copy(deep=True)
    df_grades = df_grades_.copy(deep=True)

    school_cluster_lists = df_mergers["school_cluster"].tolist()
    grades_served_per_cluster = defaultdict(set)
    school_clusters = defaultdict(list)
    for cluster in school_cluster_lists:
        schools = cluster.split(", ")
        for school in schools:
            school_clusters[school] = schools
            school_grades = df_grades[df_grades["NCESSCH"] == school].iloc[0]
            for grade in constants.GRADE_TO_INDEX:
                if school_grades[grade]:
                    grades_served_per_cluster[cluster].add(grade)

    (
        num_per_cat_per_school,
        num_per_school_per_grade_per_cat,
    ) = _calculate_student_distributions(school_clusters, df_grades, df_schools_in_play)

    _validate_solution(
        grades_served_per_cluster,
        num_per_cat_per_school,
        df_schools_in_play,
        df_grades,
    )

    dissim_val, bh_wa_dissim_val = compute_dissimilarity_metrics(
        school_clusters, num_per_cat_per_school
    )

    population_metrics = compute_population_metrics(
        df_schools_in_play, num_per_cat_per_school
    )

    (
        num_total_students,
        num_students_switching,
        num_students_switching_per_school,
    ) = _count_switching_students(school_cluster_lists, df_grades, df_schools_in_play)

    travel_time_impacts = estimate_travel_time_impacts(
        school_cluster_lists,
        df_grades,
        df_schools_in_play,
    )

    return (
        dissim_val,
        bh_wa_dissim_val,
        population_metrics,
        num_per_cat_per_school,
        num_per_school_per_grade_per_cat,
        num_total_students,
        num_students_switching,
        num_students_switching_per_school,
        travel_time_impacts,
    )


def maybe_load_large_files(state):
    global DF_BLOCKS, TRAVEL_TIMES
    if DF_BLOCKS is None and TRAVEL_TIMES is None:
        DF_BLOCKS = pd.read_csv(
            constants.BLOCKS_FILE.format(state),
            dtype={"ncessch": str, "block_id": str},
        )
        TRAVEL_TIMES = header.read_json(constants.TRAVEL_TIMES_FILE.format(state))

        print("Loaded large files.")


def output_solver_solution(
    *,
    config,
    solver,
    matches,
    grades_interval_binary,
    df_schools_in_play,
    output_dir,
    s3_bucket,
    pre_dissim_wnw,
    pre_dissim_bh_wa,
    pre_population_metrics,
):
    maybe_load_large_files(config.district.state)

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

    if config.write_to_s3:
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
    df_mergers_g.to_csv(os.path.join(output_dir, "school_mergers.csv"), index=False)
    df_grades.to_csv(os.path.join(output_dir, "grades_served.csv"), index=False)
    df_schools_in_play.to_csv(
        os.path.join(output_dir, "schools_in_play.csv"), index=False
    )

    # Compute post-solver outcomes of interest
    try:
        (
            post_dissim_wnw,
            post_dissim_bh_wa,
            post_population_metrics,
            num_per_cat_per_school,
            num_per_school_per_grade_per_cat,
            num_total_students,
            num_students_switching,
            num_students_switching_per_school,
            travel_time_impacts,
        ) = check_solution_validity_and_compute_outcomes(
            df_mergers_g, df_grades, df_schools_in_play, config.district.state
        )

    except Exception as e:
        print(f"ERROR!!!! {e}")
        traceback.print_exc()
        errors = {"error_message": str(e)}
        header.write_json(os.path.join(output_dir, "errors.json"), errors)
        return

    pre_population_metric = pre_population_metrics[config.population_metric]
    post_population_metric = post_population_metrics[config.population_metric]

    # Output results
    data_to_output = {
        name: locals()[name]
        for name in [
            "pre_dissim_wnw",
            "post_dissim_wnw",
            "pre_dissim_bh_wa",
            "post_dissim_bh_wa",
        ]
    }
    data_to_output.update(config.to_dict())
    data_to_output.update(num_total_students)
    data_to_output.update(num_students_switching)
    data_to_output.update(travel_time_impacts["status_quo_total_driving_times_per_cat"])
    data_to_output.update(travel_time_impacts["current_total_switcher_driving_times"])
    data_to_output.update(travel_time_impacts["new_total_switcher_driving_times"])

    present_stat = (
        lambda pre, post: f"{pre:.4f} -> {post:.4f}"
        f" ({(post - pre) / pre * 100:+06.2f}%)"
    )

    print()
    print(f"dissim{' ' * 14}wnw: {present_stat(pre_dissim_wnw, post_dissim_wnw)}")
    print(f"dissim{' ' * 12}bh/wa: {present_stat(pre_dissim_bh_wa, post_dissim_bh_wa)}")

    for metric in pre_population_metrics.keys():
        stats = present_stat(
            pre_population_metrics[metric], post_population_metrics[metric]
        )
        print(f"pop{(20 - len(metric)) * ' '}{metric}: {stats}")

    print()

    try:
        print(
            f"Percent switchers: {num_students_switching['num_total_switched'] / num_total_students['num_total_all']}\n",
            f"SQ avg. travel time - all: {travel_time_impacts['status_quo_total_driving_times_per_cat']['all_status_quo_time_num_total'] / num_total_students['num_total_all'] / 60}\n",
            f"SQ avg. travel time - switchers: {travel_time_impacts['current_total_switcher_driving_times']['switcher_status_quo_time_num_total'] / num_students_switching['num_total_switched'] / 60}",
            f"New avg. travel time - switchers: {travel_time_impacts['new_total_switcher_driving_times']['switcher_new_time_num_total'] / num_students_switching['num_total_switched'] / 60}",
        )
    except (KeyError, ZeroDivisionError) as e:
        print(f"Could not print travel time stats: {e}")
        traceback.print_exc()

    # We need to do this dictionary comprehension because otherwise pandas sees the
    # district namedtuple and decides that the dataframe must have two rows, which we
    # don't want.
    pd.DataFrame({k: [v] for k, v in data_to_output.items()}, index=[0]).to_csv(
        os.path.join(output_dir, "analytics.csv"), index=False
    )

    impacts = travel_time_impacts | {
        name: locals()[name]
        for name in [
            "num_per_school_per_grade_per_cat",
            "num_per_cat_per_school",
            "num_students_switching_per_school",
        ]
    }

    if config.write_to_s3:
        from s3fs import S3FileSystem

        for name, thing in impacts.items():
            s3 = S3FileSystem()
            with s3.open(os.path.join(output_dir, name + ".json"), "w") as file:
                json.dump(thing, file)

    else:
        for name, thing in impacts.items():
            with open(os.path.join(output_dir, name + ".json"), "w") as file:
                json.dump(thing, file)


def produce_post_solver_files_parallel(
    batch="min_elem_4_interdistrict_bottom_sensitivity",
    solutions_dir="data/results/{}/",
):
    # Compute pre/post dissim and other outcomes of interest
    all_jobs = []
    for state in os.listdir(solutions_dir.format(batch)):
        if "consolidated" in state:
            continue
        for district_id in os.path.join(solutions_dir.format(batch), state):
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
                                "**/**" + "school_mergers.csv",
                            ),
                            recursive=True,
                        )[0],
                        dtype=str,
                    )
                    df_grades = pd.read_csv(
                        glob.glob(
                            os.path.join(
                                this_dir,
                                "**/**" + "grades_served.csv",
                            ),
                            recursive=True,
                        )[0],
                        dtype={"NCESSCH": str},
                    )
                    df_schools_in_play = pd.read_csv(
                        glob.glob(
                            os.path.join(
                                this_dir,
                                "**/**" + "schools_in_play.csv",
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
    p.starmap(output_solver_solution, all_jobs)

    p.terminate()
    p.join()


def consolidate_results_files(
    batch="min_elem_4_interdistrict_bottom_sensitivity",
    batch_dir="data/results/{}",
    output_file="data/results/{}/consolidated_simulation_results_{}_{}_{}.csv",
):

    analytics_files = glob.glob(
        os.path.join(batch_dir.format(batch), "**/**" + "analytics.csv"), recursive=True
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
    batch_1="min_elem_4_interdistrict_bottom_sensitivity",
    batch_2="min_elem_4_bottomless",
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
