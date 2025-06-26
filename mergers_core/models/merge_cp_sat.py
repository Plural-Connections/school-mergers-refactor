from mergers_core.utils.header import *
from ortools.sat.python import cp_model
from mergers_core.models.constants import *
from mergers_core.models.model_utils import output_solver_solution


def load_and_process_data(state, district_id, interdistrict):
    df_schools = pd.read_csv(
        f"data/solver_files/2122/{state}/school_enrollments.csv", dtype={"NCESSCH": str}
    )
    df_schools["leaid"] = df_schools["NCESSCH"].str[:7]
    df_schools_curr_d = df_schools[df_schools["leaid"] == district_id].reset_index(
        drop=True
    )
    district_schools = list(set(df_schools_curr_d["NCESSCH"].tolist()))
    permissible_matches = {}
    all_districts_involved = set()
    if interdistrict:
        permissible_matches = read_dict(
            f"data/solver_files/2122/{state}/between_within_district_allowed_mergers.json"
        )
        for s in district_schools:
            all_districts_involved.update([s2[:7] for s2 in permissible_matches[s]])
    else:
        permissible_matches = read_dict(
            f"data/solver_files/2122/{state}/within_district_allowed_mergers.json"
        )
        all_districts_involved.add(district_id)

    all_districts_involved = list(all_districts_involved)

    df_capacities = pd.read_csv(
        "data/school_data/21_22_school_capacities.csv", dtype={"NCESSCH": str}
    )
    df_capacities["leaid"] = df_capacities["NCESSCH"].str[:7]
    df_capacities = df_capacities[
        df_capacities["leaid"].isin(all_districts_involved)
    ].reset_index(drop=True)
    school_capacities = {
        df_capacities["NCESSCH"][i]: int(df_capacities["student_capacity"][i])
        for i in range(0, len(df_capacities))
    }
    df_schools_in_play = df_schools[
        df_schools["leaid"].isin(all_districts_involved)
    ].reset_index(drop=True)

    curr_schools = df_schools_in_play["NCESSCH"].tolist()
    total_per_grade_per_school = defaultdict(dict)
    total_pop_per_cat_across_schools = Counter()
    for i in range(0, len(df_schools_in_play)):
        for r in RACE_KEYS.values():
            total_per_grade_per_school[df_schools_in_play["NCESSCH"][i]][r] = [
                int(df_schools_in_play[f"{r}_{g}"][i]) for g in GRADE_TO_IND
            ]

            total_pop_per_cat_across_schools[r] += sum(
                [int(df_schools_in_play[f"{r}_{g}"][i]) for g in GRADE_TO_IND]
            )

    return (
        school_capacities,
        permissible_matches,
        total_per_grade_per_school,
        total_pop_per_cat_across_schools,
        df_schools_in_play,
    )


def initialize_variables(model, df_schools_in_play):
    # Variable to track which schools are matched with which schools
    nces_ids = set(df_schools_in_play["NCESSCH"].tolist())
    matches = {
        school: {
            school_2: model.NewBoolVar(f"{school},{school_2}_matched")
            for school_2 in nces_ids
        }
        for school in nces_ids
    }

    # Variable to track which grades are offered by which schools
    grades_start = {}
    grades_end = {}
    grades_duration = {}
    grades_interval = {}
    grades_interval_binary = {}
    all_grades = list(GRADE_TO_IND.values())
    for school in nces_ids:
        grades_start[school] = model.NewIntVar(
            all_grades[0], all_grades[-1], f"{school}_start_grade"
        )
        grades_end[school] = model.NewIntVar(
            all_grades[0], all_grades[-1], f"{school}_end_grade"
        )
        grades_duration[school] = model.NewIntVar(
            0, all_grades[-1], f"{school}_grade_duration"
        )
        grades_interval[school] = model.NewIntervalVar(
            grades_start[school],
            grades_duration[school],
            grades_end[school],
            # is_present,
            f"{school}_grade_interval",
        )
        grades_interval_binary[school] = [
            model.NewIntVar(0, 1, "{},{}".format(school, i))
            for i in GRADE_TO_IND.values()
        ]

        for i in range(0, len(grades_interval_binary[school])):
            i_less_than = model.NewBoolVar("{},{}_less".format(school, i))
            model.Add(i <= grades_end[school]).OnlyEnforceIf(i_less_than)
            model.Add(i > grades_end[school]).OnlyEnforceIf(i_less_than.Not())
            i_greater_than = model.NewBoolVar(f"{school},{i}_greater")
            model.Add(i >= grades_start[school]).OnlyEnforceIf(i_greater_than)
            model.Add(i < grades_start[school]).OnlyEnforceIf(i_greater_than.Not())
            i_in_range = model.NewBoolVar("{},{}_in_range".format(school, i))
            model.AddMultiplicationEquality(i_in_range, [i_less_than, i_greater_than])
            model.Add(grades_interval_binary[school][i] == 1).OnlyEnforceIf(i_in_range)
            model.Add(grades_interval_binary[school][i] == 0).OnlyEnforceIf(
                i_in_range.Not()
            )

    # Add in hints to reflect the status quo
    for school in nces_ids:
        # Initialization: each school serves all grades
        for i in range(0, len(grades_interval_binary[school])):
            model.AddHint(grades_interval_binary[school][i], 1)

        # Initialization: each school is only "matched" to itself
        for s2 in nces_ids:
            if school == s2:
                model.AddHint(matches[school][s2], True)
            else:
                model.AddHint(matches[school][s2], False)

    return (
        matches,
        grades_interval_binary,
    )


def set_constraints(
    model,
    school_capacities,
    school_decrease_threshold,
    total_per_grade_per_school,
    permissible_matches,
    matches,
    grades_interval_binary,
):
    # CONSTRAINT: define symmetry of matches between any 3 schools
    # NOTE: we are hard coding this because we only allow 3 schools to be merged together.  Allowing more schools will require
    # more nested loops

    # TODO(ng): not sure we need all of the onlyenforceif clauses (some might be redundant)
    for s in matches:
        for s2 in matches:
            model.Add(matches[s2][s] == True).OnlyEnforceIf(matches[s][s2])
            model.Add(matches[s][s2] == True).OnlyEnforceIf(matches[s2][s])
            for s3 in matches:
                s2_s3_matched_to_s1 = model.NewBoolVar(
                    f"{s2}_{s3}_should_be_matched_to_{s}"
                )
                model.AddMultiplicationEquality(
                    s2_s3_matched_to_s1, [matches[s][s2], matches[s2][s3]]
                )
                model.Add(matches[s][s3] == True).OnlyEnforceIf(s2_s3_matched_to_s1)
                model.Add(matches[s3][s] == True).OnlyEnforceIf(s2_s3_matched_to_s1)

    student_count_sums = defaultdict(list)
    grades_represented_sums = defaultdict(list)
    for s in matches:
        # CONSTRAINT: A school can only be matched to 1 (itself), 2, or 3 other schools
        model.Add(sum([matches[s][s2] for s2 in matches]) >= 1)
        model.Add(sum([matches[s][s2] for s2 in matches]) <= 3)

        # For the constraint in the inner loop that tracks the total number of students assigned to a schoool
        sum_s = model.NewIntVar(0, MAX_TOTAL_STUDENTS, f"{s}_total_students")
        model.Add(
            sum_s
            == sum(
                [
                    total_per_grade_per_school[s]["num_total"][i]
                    * grades_interval_binary[s][i]
                    for i in GRADE_TO_IND.values()
                ]
            )
        )
        student_count_sums[s].append(sum_s)

        # For the constraint in the inner loop that tracks the grades that are assigned to merged schools
        num_grades_s = model.NewIntVar(0, len(GRADE_TO_IND), f"{s}_num_grade_levels")
        model.Add(num_grades_s == sum(grades_interval_binary[s]))
        grades_represented_sums[s].append(num_grades_s)

        for s2 in matches[s]:
            # CONSTRAINT: Schools can only be matched to those schools they are bordering ("permissible matches")
            # if s != s2:
            if s2 not in permissible_matches[s]:
                model.Add(matches[s][s2] == False)

            # CONSTRAINT: For schools that are matched, do not allow grade offerings to overlap

            if s != s2:
                # Attempt at a homegrown no overlap constraint — basically, if two schools are matched together,
                # they can't serve the same grades
                for i in range(0, len(list(GRADE_TO_IND.values()))):
                    curr_prod = model.NewIntVar(0, 1, f"grade_int_prod_{s}_{s2}")
                    model.AddMultiplicationEquality(
                        curr_prod,
                        [grades_interval_binary[s][i], grades_interval_binary[s2][i]],
                    )
                    model.Add(curr_prod == 0).OnlyEnforceIf(matches[s][s2])

            # CONSTRAINT: School totals must be less than capacity -
            # this part is just summing up student counts per grade for merged schools
            sum_s2 = model.NewIntVar(0, MAX_TOTAL_STUDENTS, f"{s}_{s2}_total_students")

            # Here, we add on the number of students per grade from s2 in the grades that school s will be serving
            # if s and s2 are the same school, set the sum of s2 to 0
            if s != s2:
                model.Add(
                    sum_s2
                    == sum(
                        [
                            total_per_grade_per_school[s2]["num_total"][i]
                            * grades_interval_binary[s][i]
                            for i in GRADE_TO_IND.values()
                        ]
                    )
                ).OnlyEnforceIf(matches[s][s2])
                model.Add(sum_s2 == 0).OnlyEnforceIf(matches[s][s2].Not())
            else:
                model.Add(sum_s2 == 0)

            student_count_sums[s].append(sum_s2)

            # CONSTRAINT: the grade levels assigned to merged schools should add up to the total number of grade levels
            num_grades_s2 = model.NewIntVar(
                0, len(GRADE_TO_IND), f"{s}_{s2}_num_grade_levels"
            )

            if s != s2:
                model.Add(
                    num_grades_s2 == sum(grades_interval_binary[s2])
                ).OnlyEnforceIf(matches[s][s2])
                model.Add(num_grades_s2 == 0).OnlyEnforceIf(matches[s][s2].Not())
            else:
                model.Add(num_grades_s2 == 0)

            grades_represented_sums[s].append(num_grades_s2)

        # CONSTRAINT: make sure all grades are represented across merged schools collectively
        model.Add(sum(grades_represented_sums[s]) == len(GRADE_TO_IND))

        # CONSTRAINT: School totals must be less than capacity
        model.Add(sum(student_count_sums[s]) <= school_capacities[s])

        # CONSTRAINT: School totals must not be less than school_decrease_threshold % of current enrollment
        school_cap_lower_bound = int(
            SCALING[0]
            * np.round(
                (1 - school_decrease_threshold)
                * sum(
                    [
                        total_per_grade_per_school[s]["num_total"][i]
                        for i in GRADE_TO_IND.values()
                    ]
                ),
                decimals=SCALING[1],
            )
        )
        model.Add(SCALING[0] * sum(student_count_sums[s]) >= school_cap_lower_bound)

        max_grade_served_s = model.NewIntVar(0, 20, f"{s}_grade_max")
        all_grades_served_s = []
        for g in GRADE_TO_IND.values():
            all_grades_served_s.append(grades_interval_binary[s][g] * g)
        model.AddMaxEquality(max_grade_served_s, all_grades_served_s)
        for s2 in matches[s]:
            if s == s2:
                continue
            # CONSTRAINT: School s totals must be less than capacity of whatever school s2 it is matched to
            # as long as s2 serves grades later than s

            max_grades_served_s_s2 = model.NewIntVar(0, 20, f"{s}_{s2}_grade_max")
            all_grades_served_s2 = []
            for g in GRADE_TO_IND.values():
                all_grades_served_s2.append(grades_interval_binary[s2][g] * g)
            model.AddMaxEquality(max_grades_served_s_s2, all_grades_served_s2)
            s2_serving_higher_grades_than_s = model.NewBoolVar(f"{s}_{s2}_higher_grade")
            model.Add(max_grades_served_s_s2 > max_grade_served_s).OnlyEnforceIf(
                s2_serving_higher_grades_than_s
            )
            model.Add(max_grades_served_s_s2 <= max_grade_served_s).OnlyEnforceIf(
                s2_serving_higher_grades_than_s.Not()
            )
            condition = model.NewBoolVar(f"{s}_{s2}_condition")
            model.AddMultiplicationEquality(
                condition, [s2_serving_higher_grades_than_s, matches[s][s2]]
            )

            model.Add(
                sum(student_count_sums[s]) <= school_capacities[s2]
            ).OnlyEnforceIf(condition)

            # CONSTRAINT: School totals must not be less than school_decrease_threshold % of current enrollment
            school_cap_lower_bound = int(
                SCALING[0]
                * np.round(
                    (1 - school_decrease_threshold)
                    * sum(
                        [
                            total_per_grade_per_school[s2]["num_total"][i]
                            for i in GRADE_TO_IND.values()
                        ]
                    ),
                    decimals=SCALING[1],
                )
            )
            model.Add(
                SCALING[0] * sum(student_count_sums[s]) >= school_cap_lower_bound
            ).OnlyEnforceIf(condition)


def set_objective_white_nonwhite_dissimilarity(
    model,
    dissim_weight,
    total_per_grade_per_school,
    total_pop_per_cat_across_schools,
    matches,
    grades_interval_binary,
):
    dissim_objective_terms = []
    students_switching_terms = []
    for s in matches:
        # For the constraint in the inner loop that tracks the total number of students assigned to a schoool
        sum_s_white = model.NewIntVar(0, MAX_TOTAL_STUDENTS, f"{s}_white_students")
        sum_s_total = model.NewIntVar(0, MAX_TOTAL_STUDENTS, f"{s}_all_students")
        model.Add(
            sum_s_white
            == sum(
                [
                    total_per_grade_per_school[s]["num_white"][i]
                    * grades_interval_binary[s][i]
                    for i in GRADE_TO_IND.values()
                ]
            )
        )
        model.Add(
            sum_s_total
            == sum(
                [
                    total_per_grade_per_school[s]["num_total"][i]
                    * grades_interval_binary[s][i]
                    for i in GRADE_TO_IND.values()
                ]
            )
        )

        total_cat_students_at_school = [sum_s_white]
        total_students_at_school = [sum_s_total]
        for s2 in matches[s]:
            if s == s2:
                continue
            sum_s2_white = model.NewIntVar(
                0, MAX_TOTAL_STUDENTS, f"{s}_{s2}_white_students".format(s, s2)
            )
            sum_s2_total = model.NewIntVar(
                0, MAX_TOTAL_STUDENTS, f"{s}_{s2}_all_students".format(s, s2)
            )

            # Here, we add on the number of students per grade from s2 in the grades that school s will be serving
            model.Add(
                sum_s2_white
                == sum(
                    [
                        total_per_grade_per_school[s2]["num_white"][i]
                        * grades_interval_binary[s][i]
                        for i in GRADE_TO_IND.values()
                    ]
                )
            ).OnlyEnforceIf(matches[s][s2])
            model.Add(sum_s2_white == 0).OnlyEnforceIf(matches[s][s2].Not())

            total_cat_students_at_school.append(sum_s2_white)

            model.Add(
                sum_s2_total
                == sum(
                    [
                        total_per_grade_per_school[s2]["num_total"][i]
                        * grades_interval_binary[s][i]
                        for i in GRADE_TO_IND.values()
                    ]
                )
            ).OnlyEnforceIf(matches[s][s2])
            model.Add(sum_s2_total == 0).OnlyEnforceIf(matches[s][s2].Not())

            total_students_at_school.append(sum_s2_total)

            ### Track students who are switching schools ###
            # curr_students_switching = model.NewIntVar(
            #     0,
            #     MAX_TOTAL_STUDENTS,
            #     f"{s}_{s2}_all_students_switching".format(s, s2),
            # )

            # model.Add(curr_students_switching == sum_s2_total).OnlyEnforceIf(
            #     matches[s][s2]
            # )
            # model.Add(curr_students_switching == 0).OnlyEnforceIf(matches[s][s2].Not())
            # students_switching_terms.append(curr_students_switching)

        # Scaling and prepping to apply division equality constraint,
        # required to do divisions in CP-SAT
        scaled_total_cat_students_at_school = model.NewIntVar(0, SCALING[0] ** 2, "")

        model.Add(
            scaled_total_cat_students_at_school
            == SCALING[0] * sum(total_cat_students_at_school)
        )

        scaled_total_non_cat_students_at_school = model.NewIntVar(
            0, SCALING[0] ** 2, ""
        )
        model.Add(
            scaled_total_non_cat_students_at_school
            == SCALING[0]
            * (sum(total_students_at_school) - sum(total_cat_students_at_school))
        )

        # Fraction of cat students across the district that are at this school
        cat_ratio_at_school = model.NewIntVar(0, SCALING[0] ** 2, "")
        model.AddDivisionEquality(
            cat_ratio_at_school,
            scaled_total_cat_students_at_school,
            total_pop_per_cat_across_schools["num_white"],
        )

        # Fraction of non-cat students that are at this school
        non_cat_ratio_at_school = model.NewIntVar(0, SCALING[0] ** 2, "")
        model.AddDivisionEquality(
            non_cat_ratio_at_school,
            scaled_total_non_cat_students_at_school,
            total_pop_per_cat_across_schools["num_total"]
            - total_pop_per_cat_across_schools["num_white"],
        )

        # Computing dissimilarity index
        diff_val = model.NewIntVar(-(SCALING[0] ** 2), SCALING[0] ** 2, "")
        model.Add(diff_val == cat_ratio_at_school - non_cat_ratio_at_school)
        obj_term_to_add = model.NewIntVar(0, SCALING[0] ** 2, "")
        model.AddAbsEquality(
            obj_term_to_add,
            diff_val,
        )

        dissim_objective_terms.append(obj_term_to_add)

    # Compute ratio of students switching schools
    # scaled_num_students_switching_schools = model.NewIntVar(0, SCALING[0] ** 2, "")
    # model.Add(
    #     scaled_num_students_switching_schools
    #     == SCALING[0] * sum(students_switching_terms)
    # )

    # ratio_students_switching_schools = model.NewIntVar(0, SCALING[0] ** 2, "")
    # model.AddDivisionEquality(
    #     ratio_students_switching_schools,
    #     scaled_num_students_switching_schools,
    #     total_pop_per_cat_across_schools["num_total"],
    # )

    ## Full objective function to minimize
    dissim_val = model.NewIntVar(0, MAX_TOTAL_STUDENTS * SCALING[0], "")
    model.Add(dissim_val == sum(dissim_objective_terms))
    model.Minimize(dissim_val)
    # model.Minimize(
    #     int(SCALING[0] * dissim_weight) * dissim_val
    #     + int(SCALING[0] * (1 - dissim_weight)) * 2 * ratio_students_switching_schools
    # )


def set_objective_bh_wa_dissimilarity(
    model,
    dissim_weight,
    total_per_grade_per_school,
    total_pop_per_cat_across_schools,
    matches,
    grades_interval_binary,
):
    dissim_objective_terms = []
    students_switching_terms = []
    for s in matches:
        # For the constraint in the inner loop that tracks the total number of students assigned to a schoool
        sum_s_bh = model.NewIntVar(0, MAX_TOTAL_STUDENTS, f"{s}_bh_students".format(s))
        sum_s_wa = model.NewIntVar(0, MAX_TOTAL_STUDENTS, f"{s}_wa_students".format(s))
        model.Add(
            sum_s_bh
            == sum(
                [
                    (
                        total_per_grade_per_school[s]["num_black"][i]
                        + total_per_grade_per_school[s]["num_hispanic"][i]
                    )
                    * grades_interval_binary[s][i]
                    for i in GRADE_TO_IND.values()
                ]
            )
        )
        model.Add(
            sum_s_wa
            == sum(
                [
                    (
                        total_per_grade_per_school[s]["num_white"][i]
                        + total_per_grade_per_school[s]["num_asian"][i]
                    )
                    * grades_interval_binary[s][i]
                    for i in GRADE_TO_IND.values()
                ]
            )
        )

        total_bh_students_at_school = [sum_s_bh]
        total_wa_students_at_school = [sum_s_wa]
        for s2 in matches[s]:
            if s == s2:
                continue
            sum_s2_bh = model.NewIntVar(
                0, MAX_TOTAL_STUDENTS, f"{s}_{s2}_bh_students".format(s, s2)
            )
            sum_s2_wa = model.NewIntVar(
                0, MAX_TOTAL_STUDENTS, f"{s}_{s2}_wa_students".format(s, s2)
            )

            # Here, we add on the number of students per grade from s2 in the grades that school s will be serving
            model.Add(
                sum_s2_bh
                == sum(
                    [
                        (
                            total_per_grade_per_school[s2]["num_black"][i]
                            + total_per_grade_per_school[s2]["num_hispanic"][i]
                        )
                        * grades_interval_binary[s][i]
                        for i in GRADE_TO_IND.values()
                    ]
                )
            ).OnlyEnforceIf(matches[s][s2])
            model.Add(sum_s2_bh == 0).OnlyEnforceIf(matches[s][s2].Not())

            total_bh_students_at_school.append(sum_s2_bh)

            model.Add(
                sum_s2_wa
                == sum(
                    [
                        (
                            total_per_grade_per_school[s2]["num_white"][i]
                            + total_per_grade_per_school[s2]["num_asian"][i]
                        )
                        * grades_interval_binary[s][i]
                        for i in GRADE_TO_IND.values()
                    ]
                )
            ).OnlyEnforceIf(matches[s][s2])
            model.Add(sum_s2_wa == 0).OnlyEnforceIf(matches[s][s2].Not())

            total_wa_students_at_school.append(sum_s2_wa)

        # Scaling and prepping to apply division equality constraint,
        # required to do divisions in CP-SAT
        scaled_total_bh_students_at_school = model.NewIntVar(0, SCALING[0] ** 2, "")

        model.Add(
            scaled_total_bh_students_at_school
            == SCALING[0] * sum(total_bh_students_at_school)
        )

        scaled_total_wa_students_at_school = model.NewIntVar(0, SCALING[0] ** 2, "")
        model.Add(
            scaled_total_wa_students_at_school
            == SCALING[0] * sum(total_wa_students_at_school)
        )

        # Fraction of bh students across the district that are at this school
        bh_ratio_at_school = model.NewIntVar(0, SCALING[0] ** 2, "")
        model.AddDivisionEquality(
            bh_ratio_at_school,
            scaled_total_bh_students_at_school,
            total_pop_per_cat_across_schools["num_black"]
            + total_pop_per_cat_across_schools["num_hispanic"],
        )

        # Fraction of wa students that are at this school
        wa_ratio_at_school = model.NewIntVar(0, SCALING[0] ** 2, "")
        model.AddDivisionEquality(
            wa_ratio_at_school,
            scaled_total_wa_students_at_school,
            total_pop_per_cat_across_schools["num_white"]
            + total_pop_per_cat_across_schools["num_asian"],
        )

        # Computing dissimilarity index
        diff_val = model.NewIntVar(-(SCALING[0] ** 2), SCALING[0] ** 2, "")
        model.Add(diff_val == bh_ratio_at_school - wa_ratio_at_school)
        obj_term_to_add = model.NewIntVar(0, SCALING[0] ** 2, "")
        model.AddAbsEquality(
            obj_term_to_add,
            diff_val,
        )

        dissim_objective_terms.append(obj_term_to_add)

    # Compute ratio of students switching schools
    # scaled_num_students_switching_schools = model.NewIntVar(0, SCALING[0] ** 2, "")
    # model.Add(
    #     scaled_num_students_switching_schools
    #     == SCALING[0] * sum(students_switching_terms)
    # )

    # ratio_students_switching_schools = model.NewIntVar(0, SCALING[0] ** 2, "")
    # model.AddDivisionEquality(
    #     ratio_students_switching_schools,
    #     scaled_num_students_switching_schools,
    #     total_pop_per_cat_across_schools["num_total"],
    # )

    ## Full objective function to minimize
    dissim_val = model.NewIntVar(0, MAX_TOTAL_STUDENTS * SCALING[0], "")
    model.Add(dissim_val == sum(dissim_objective_terms))
    model.Minimize(dissim_val)
    # model.Minimize(
    #     int(SCALING[0] * dissim_weight) * dissim_val
    #     + int(SCALING[0] * (1 - dissim_weight)) * 2 * ratio_students_switching_schools
    # )


def solve_and_output_results(
    state="NC",
    district_id="3701500",
    school_decrease_threshold=0.2,
    dissim_weight=1,
    interdistrict=False,
    objective="bh_wa",
    # objective="white_nonwhite",
    batch="testing",
    output_dir="data/results/{}/{}/{}/{}_{}_{}_{}_{}_{}/",
    mergers_file_name="school_mergers.csv",
    grades_served_file_name="grades_served.csv",
    schools_in_play_file_name="schools_in_play.csv",
    s3_bucket="s3://school-mergers/",
    write_to_s3=False,
):
    print(f"Loading and processing data for {state} {district_id} ...")
    (
        school_capacities,
        permissible_matches,
        total_per_grade_per_school,
        total_pop_per_cat_across_schools,
        df_schools_in_play,
    ) = load_and_process_data(state, district_id, interdistrict)

    # Create the cp model
    model = cp_model.CpModel()

    print("Initializing variables ...")
    (
        matches,
        grades_interval_binary,
    ) = initialize_variables(model, df_schools_in_play)

    print("Setting constraints ...")
    set_constraints(
        model,
        school_capacities,
        school_decrease_threshold,
        total_per_grade_per_school,
        permissible_matches,
        matches,
        grades_interval_binary,
    )

    if objective == "bh_wa":
        print("Setting objective function bh/wa ...")
        set_objective_bh_wa_dissimilarity(
            model,
            dissim_weight,
            total_per_grade_per_school,
            total_pop_per_cat_across_schools,
            matches,
            grades_interval_binary,
        )
    else:
        print("Setting objective function white/non-white...")
        set_objective_white_nonwhite_dissimilarity(
            model,
            dissim_weight,
            total_per_grade_per_school,
            total_pop_per_cat_across_schools,
            matches,
            grades_interval_binary,
        )

    print("Solving ...")
    solver = cp_model.CpSolver()

    # Sets a time limit for solver
    solver.parameters.max_time_in_seconds = MAX_SOLVER_TIME

    # Adding parallelism
    solver.parameters.num_search_workers = NUM_SOLVER_THREADS

    status = solver.Solve(model)

    curr_output_dir = output_dir.format(
        batch,
        state,
        district_id,
        interdistrict,
        school_decrease_threshold,
        status,
        solver.WallTime(),
        solver.NumBranches(),
        solver.NumConflicts(),
    )
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Status is: ", status)
        print("Outputting solution ...")
        output_solver_solution(
            solver,
            matches,
            grades_interval_binary,
            state,
            district_id,
            school_decrease_threshold,
            interdistrict,
            df_schools_in_play,
            curr_output_dir,
            s3_bucket,
            write_to_s3,
            mergers_file_name,
            grades_served_file_name,
            schools_in_play_file_name,
        )

    else:
        print("Status: ", status)
        print(model.Validate())
        result = {"status": status}
        df_r = pd.DataFrame(result, index=[0])
        if write_to_s3:
            df_r.to_csv(s3_bucket + curr_output_dir, index=False)
        else:
            Path(curr_output_dir).mkdir(parents=True, exist_ok=True)
            write_dict(os.path.join(curr_output_dir, "output.json"), result)


if __name__ == "__main__":
    solve_and_output_results()
