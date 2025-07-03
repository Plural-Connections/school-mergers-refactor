from ortools.sat.python import cp_model
import mergers_core.utils.header as header
import mergers_core.models.constants as constants
from mergers_core.models.model_utils import output_solver_solution
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
import os


def load_and_process_data(state, district_id, interdistrict):
    """
    Loads and preprocesses all necessary data for the CP-SAT model.

    This function reads school enrollment data, capacity data, and permissible
    merger information from CSV and JSON files. It filters this data for the
    specified district and other districts involved in potential interdistrict
    mergers. It then calculates and aggregates student counts by race and grade.

    Args:
        state (str): The two-letter state abbreviation.
        district_id (str): The unique identifier for the school district.
        interdistrict (bool): Flag indicating whether to consider mergers
            between different districts.

    Returns:
        tuple: A tuple containing:
            - school_capacities (dict): Maps school ID to its student capacity.
            - permissible_matches (dict): Maps each school ID to a list of
              other school IDs it is allowed to merge with.
            - total_per_grade_per_school (defaultdict): Nested dictionary
              mapping school ID and race to a list of student counts per grade.
            - total_pop_per_cat_across_schools (Counter): Counts of total
              students per racial category across all involved schools.
            - df_schools_in_play (pd.DataFrame): DataFrame containing detailed
              enrollment and demographic data for all schools involved in the
              potential mergers.
    """
    # Load school enrollment data for the specified state.
    df_schools = pd.read_csv(
        f"data/solver_files/2122/{state}/school_enrollments.csv", dtype={"NCESSCH": str}
    )
    df_schools["leaid"] = df_schools["NCESSCH"].str[:7]
    # Filter for schools within the target district.
    df_schools_curr_d = df_schools[df_schools["leaid"] == district_id].reset_index(
        drop=True
    )
    district_schools = list(set(df_schools_curr_d["NCESSCH"].tolist()))
    permissible_matches = {}
    all_districts_involved = set()

    # Load permissible merger data based on whether the scenario is interdistrict.
    if interdistrict:
        permissible_matches = header.read_json(
            f"data/solver_files/2122/{state}/between_within_district_allowed_mergers.json"
        )
        # Identify all districts that are involved in the potential mergers.
        for school in district_schools:
            all_districts_involved.update(
                [school_2[:7] for school_2 in permissible_matches[school]]
            )
    else:
        permissible_matches = header.read_json(
            f"data/solver_files/2122/{state}/within_district_allowed_mergers.json"
        )
        all_districts_involved.add(district_id)

    all_districts_involved = list(all_districts_involved)

    # Load school capacity data and filter for the involved districts.
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
    # Filter the main schools DataFrame to include all schools "in play".
    df_schools_in_play = df_schools[
        df_schools["leaid"].isin(all_districts_involved)
    ].reset_index(drop=True)

    # Aggregate student counts by grade and race for each school.
    total_per_grade_per_school = defaultdict(dict)
    total_pop_per_cat_across_schools = Counter()
    for i in range(0, len(df_schools_in_play)):
        for race in constants.RACE_KEYS.values():
            total_per_grade_per_school[df_schools_in_play["NCESSCH"][i]][race] = [
                int(df_schools_in_play[f"{race}_{grade}"][i])
                for grade in constants.GRADE_TO_INDEX
            ]

            # Sum up the total population for each racial category across all schools.
            total_pop_per_cat_across_schools[race] += sum(
                [
                    int(df_schools_in_play[f"{race}_{grade}"][i])
                    for grade in constants.GRADE_TO_INDEX
                ]
            )

    return (
        school_capacities,
        permissible_matches,
        total_per_grade_per_school,
        total_pop_per_cat_across_schools,
        df_schools_in_play,
    )


def initialize_variables(model, df_schools_in_play):
    """
    Initializes the core variables for the CP-SAT model.

    This function creates the decision variables that the solver will manipulate
    to find an optimal solution. These include variables for school matches and
    grade assignments.

    Args:
        model (cp_model.CpModel): The CP-SAT model instance.
        df_schools_in_play (pd.DataFrame): DataFrame containing data for all
            schools to be considered in the model.

    Returns:
        tuple: A tuple containing:
            - matches (dict): A nested dictionary of boolean variables, where
              matches[s1][s2] is true if school s1 and s2 are merged.
            - grades_interval_binary (dict): A dictionary mapping each school ID
              to a list of binary variables, one for each grade level,
              indicating if the school serves that grade.
    """
    # Create a set of unique school IDs from the input DataFrame.
    nces_ids = set(df_schools_in_play["NCESSCH"].tolist())

    # Create a symmetric matrix of boolean variables to represent matches.
    # matches[s1][s2] is true if school s1 and school s2 are merged.
    matches = {
        school: {
            school_2: model.NewBoolVar(f"{school},{school_2}_matched")
            for school_2 in nces_ids
        }
        for school in nces_ids
    }

    # --- Grade Assignment Variables ---
    # These variables determine the grade range (e.g., K-5, 6-8) for each school.
    grades_start = {}
    grades_end = {}
    grades_duration = {}
    grades_interval = {}
    grades_interval_binary = {}  # Binary representation of grades served.
    all_grades = list(constants.GRADE_TO_INDEX.values())

    for school in nces_ids:
        # Variables for the start, end, and duration of the grade sequence.
        grades_start[school] = model.NewIntVar(
            all_grades[0], all_grades[-1], f"{school}_start_grade"
        )
        grades_end[school] = model.NewIntVar(
            all_grades[0], all_grades[-1], f"{school}_end_grade"
        )
        grades_duration[school] = model.NewIntVar(
            0, all_grades[-1], f"{school}_grade_duration"
        )
        # Interval variable representing the continuous block of grades.
        grades_interval[school] = model.NewIntervalVar(
            grades_start[school],
            grades_duration[school],
            grades_end[school],
            f"{school}_grade_interval",
        )
        # Create a binary variable for each grade to indicate if it's served.
        grades_interval_binary[school] = [
            model.NewBoolVar(f"{school},{i}") for i in constants.GRADE_TO_INDEX.values()
        ]

        # Link the interval variables (start, end) to the binary grade indicators.
        # A grade's binary indicator is 1 if and only if the grade falls
        # within the [grades_start, grades_end] range of the school.
        for i in range(0, len(grades_interval_binary[school])):
            i_less_than = model.NewBoolVar(f"{school},{i}_less")
            model.Add(i <= grades_end[school]).OnlyEnforceIf(i_less_than)
            model.Add(i > grades_end[school]).OnlyEnforceIf(i_less_than.Not())
            i_greater_than = model.NewBoolVar(f"{school},{i}_greater")
            model.Add(i >= grades_start[school]).OnlyEnforceIf(i_greater_than)
            model.Add(i < grades_start[school]).OnlyEnforceIf(i_greater_than.Not())
            i_in_range = model.NewBoolVar(f"{school},{i}_in_range")
            model.AddMultiplicationEquality(i_in_range, [i_less_than, i_greater_than])
            model.Add(grades_interval_binary[school][i] == 1).OnlyEnforceIf(i_in_range)
            model.Add(grades_interval_binary[school][i] == 0).OnlyEnforceIf(
                i_in_range.Not()
            )

    # --- Solver Hints ---
    # Provide the solver with an initial solution (the status quo) to speed up
    # the search process.
    for school in nces_ids:
        # Hint: Initially, assume each school serves all its current grades.
        for i in range(0, len(grades_interval_binary[school])):
            model.AddHint(grades_interval_binary[school][i], 1)

        # Hint: Initially, assume each school is only "matched" with itself.
        for school_2 in nces_ids:
            if school == school_2:
                model.AddHint(matches[school][school_2], True)
            else:
                model.AddHint(matches[school][school_2], False)

    return (
        matches,
        grades_interval_binary,
    )


def _get_students_at_school(
    model, matches, grades_at_school, school, students_per_grade_per_school
):
    """
    Calculates the number of students that will be assigned to a school building.

    This function determines the total student population for a given school
    building ('school') based on a potential merger scenario. It calculates this
    by summing two groups of students:
    1.  Students from the original 'school' who are in the grade levels that
        the school will serve after the merger.
    2.  Students from a merged school ('school2') who are in the grade levels
        that 'school' will serve. This is calculated for each potential merger.

    Args:
        model (cp_model.CpModel): The CP-SAT model instance.
        matches (dict): A nested dictionary of boolean variables, where
            matches[s1][s2] is true if school s1 and s2 are merged.
        grades_at_school (dict): A dictionary mapping each school ID to a list
            of binary variables, one for each grade level, indicating if the
            school serves that grade.
        school (str): The NCESSCH ID of the school building for which to
            calculate the student population.
        students_per_grade_per_school (defaultdict): A nested dictionary
            containing the number of students for each grade and racial
            category within each school.

    Returns:
        list: A list of CP-SAT integer variables representing the different
        groups of students that will make up the new population of the school
        building. The sum of this list would represent the total enrollment.
    """
    # Calculate the base number of students this school will serve from its
    # original student body based on its new grade assignments.
    students_at_school = sum(
        [
            students_per_grade_per_school[school]["num_total"][grade]
            * grades_at_school[school][grade]
            for grade in constants.GRADE_TO_INDEX.values()
        ]
    )
    model.Add(students_at_school >= 0)
    model.Add(students_at_school <= constants.MAX_TOTAL_STUDENTS)

    results: list = [students_at_school]

    # Calculate the number of students school would receive from a merger.
    for school2 in matches[school]:
        transfer_from_school2 = model.NewIntVar(
            0, constants.MAX_TOTAL_STUDENTS, f"{school}_{school2}_total_students"
        )
        if school != school2:
            # Sum students from s2 for the grades that s will now serve.
            model.Add(
                transfer_from_school2
                == sum(
                    [
                        students_per_grade_per_school[school2]["num_total"][i]
                        * grades_at_school[school][i]
                        for i in constants.GRADE_TO_INDEX.values()
                    ]
                )
            ).OnlyEnforceIf(matches[school][school2])
            model.Add(transfer_from_school2 == 0).OnlyEnforceIf(
                matches[school][school2].Not()
            )
        else:
            model.Add(
                transfer_from_school2 == 0
            )  # No students transfer if it's the same school.

        results.append(transfer_from_school2)

    return results


def set_constraints(
    model,
    school_capacities,
    school_decrease_threshold,
    students_per_grade_per_school,
    permissible_matches,
    matches,
    grades_at_school,
):
    """
    Defines the set of rules and limitations for the school merger problem.

    The main constraints, expressed mathematically, are:

        Given two sets S and G, where S is the set of elementary schools in the
        district, and G is the set of grades served across S.

        M ∈ {0,1}^|S|×|S|, where S is the set of elementary schools in the
        district. Mij = 1 implies that schools i and j should be merged. A
        school is always considered as “merged” with itself, i.e. Mii = 1 for
        all i. Further, R ∈ {0,1}^|S|×|S|. Rij = 1 implies that school i serves
        grade level j. The entries of these two matrices represent the key
        decision variables for the algorithm.
        E is the matrix of people enrolled in a certain grade at a certain school.
        g_s^{start,end} is the first or last grade at school s.
        'capacity bounds' refers to the requirement that a school is not permitted to
        admit more than its capacity of students or less than p_min *
        current_enrollment.

        Mergers are symmetric and transitive:
        Ms,s′=1 ⇒ Ms′,s=1              ∀ s,s′∈S
        Ms,s′=1 ∧ Ms′,s′′=1 ⇒ Ms,s′′=1 ∀ s,s′,s′′∈S

        Schools can only be paired, tripled, or left unchanged:
        ∑_s′∈S(Ms,s′) ∈ {1,2,3} ∀ s∈S

        The grade span served by any school must be contiguous:
        Rs,g=1 IF (g_s^start≤g<g_s^end); 0 OTHERWISE ∀ s∈S ∀ g∈G
        (enforced by the grades interval being constructed from an IntervalVar)

        Each school's enrollment must be within its capacity bounds:
        p_min⋅∑_g∈G(Es,g) ≤ ∑_g∈G(∑_s′∈S(Ms,s′))⋅Rs,g⋅Es,g ≤ Capacity(s) ∀ s∈S

        Each school's future enrollment (the enrollment expected after students have
        moved up grades) must stay within the enrollment bounds.
        g_s′^end>g_s^end ∧ Ms,s′=1 ⇒
            p_min⋅∑_g∈G(Es′,g)
            ≤ ∑_g∈G(∑_s′′∈S(Ms,s′′))⋅Rs,g⋅Es′′,g ≤
            Capacity(s′)
        ∀ s,s′∈S s≠s′

    Arguments:
        model (cp_model.CpModel): The CP-SAT model instance.
        school_capacities (dict): Maps school ID to its student capacity.
        school_decrease_threshold (float): The maximum allowable percentage
            decrease in a school's enrollment.
        students_per_grade_per_school (defaultdict): Student counts by grade,
            school, and race.
        permissible_matches (dict): Defines which schools are allowed to merge.
        matches (dict): Dictionary of boolean match variables.
        grades_at_school (dict): Dictionary of binary grade variables.
    """

    # --- Symmetry and transitivity ---
    # Ensures that if A is matched with B, B is matched with A.
    # Also enforces transitivity for 3-way mergers: if A-B and B-C, then A-C.
    # This creates cohesive merged groups.
    # NOTE: This is hardcoded for a maximum of 3 schools per merger.
    for school1 in matches:
        for school2 in matches:
            # Symmetry: If s1 is matched with s2, s2 must be matched with s1.
            model.AddImplication(matches[school1][school2], matches[school2][school1])

            # Transitivity for 3-school merges: A-B and B-C, then A-C
            for school3 in matches:
                ab_and_bc = model.NewBoolVar(
                    f"{school1}-{school2}-{school3}_transitivity"
                )
                model.AddMultiplicationEquality(  # Multiplication parallels AND
                    ab_and_bc, [matches[school1][school2], matches[school2][school3]]
                )
                model.AddImplication(
                    ab_and_bc,
                    matches[school1][school3],
                )

    for school1 in matches:
        # --- Each school can only be paired, tripled, or left unchanged ---
        num_matches = sum([matches[school1][school2] for school2 in matches])
        model.Add(num_matches >= 1)
        model.Add(num_matches <= 3)

    for school1 in matches:
        # --- Enrollment must be within a specified minimum and maximum capacity ---
        students_at_this_school = _get_students_at_school(
            model, matches, grades_at_school, school1, students_per_grade_per_school
        )
        model.Add(sum(students_at_this_school) <= school_capacities[school1])

        school_current_population = sum(
            [
                students_per_grade_per_school[school1]["num_total"][i]
                for i in constants.GRADE_TO_INDEX.values()
            ]
        )
        enrollment_lower_bound = int(
            constants.SCALING[0]
            * np.round(
                (1 - school_decrease_threshold) * school_current_population,
                decimals=constants.SCALING[1],
            )
        )
        model.Add(
            constants.SCALING[0] * sum(students_at_this_school)
            >= enrollment_lower_bound
        )

    for school1 in matches:
        for school2 in matches[school1]:
            # --- Permissible Match Constraint ---
            # A school can only be matched with schools from its pre-approved list.
            # Primarily used for adjacency checking.
            model.Add(matches[school1][school2] is False).OnlyEnforceIf(
                school2 not in permissible_matches[school1]
            )

            # --- No Grade Overlap Constraint ---
            # If two schools are merged, they cannot serve the same grade levels.
            if school1 != school2:
                for i in range(len(constants.GRADE_TO_INDEX)):
                    model.Add(
                        grades_at_school[school1][i] + grades_at_school[school2][i] == 1
                    ).OnlyEnforceIf(matches[school1][school2])

    for school1 in matches:
        # --- Grade Completeness Constraint ---
        # Ensure that a group of merged schools combined serve a full set of grades.

        # Calculate the number of grade levels this school will offer postmerger.
        num_grades_in_school = sum(grades_at_school[school1])
        model.Add(num_grades_in_school >= 0)
        model.Add(num_grades_in_school <= len(constants.GRADE_TO_INDEX))
        num_grades_represented = [num_grades_in_school]
        for school2 in matches[school1]:
            # Number of grades school2 serves after a merger with school1.
            num_grades_s2 = model.NewIntVar(
                0,
                len(constants.GRADE_TO_INDEX),
                f"{school1}_{school2}_num_grade_levels",
            )
            if school1 != school2:
                model.Add(
                    num_grades_s2 == sum(grades_at_school[school2])
                ).OnlyEnforceIf(matches[school1][school2])
                model.Add(num_grades_s2 == 0).OnlyEnforceIf(
                    matches[school1][school2].Not()
                )
            else:
                model.Add(num_grades_s2 == 0)

            num_grades_represented.append(num_grades_s2)

        model.Add(sum(num_grades_represented) == len(constants.GRADE_TO_INDEX))

    for school1 in matches:
        # --- Feeder Pattern Capacity Constraints ---
        # If school s and school s' are merged AND s' serves higher grades than s, then
        # the total number of students assigned to s must not exceed the
        # capacity of s'.
        max_grade_served_s = model.NewIntVar(0, 20, f"{school1}_grade_max")
        all_grades_served_s = []
        for grade in constants.GRADE_TO_INDEX.values():
            all_grades_served_s.append(grades_at_school[school1][grade] * grade)
        model.AddMaxEquality(max_grade_served_s, all_grades_served_s)

        for school2 in matches[school1]:
            if school1 == school2:
                continue

            # Determine if s2 serves higher grades than s.
            max_grade_served_s2 = model.NewIntVar(
                0, 20, f"{school1}_{school2}_grade_max"
            )
            all_grades_served_s2 = []
            for grade in constants.GRADE_TO_INDEX.values():
                all_grades_served_s2.append(grades_at_school[school2][grade] * grade)
            model.AddMaxEquality(max_grade_served_s2, all_grades_served_s2)

            s2_serving_higher_grades_than_s = model.NewBoolVar(
                f"{school1}_{school2}_higher_grade"
            )
            model.Add(max_grade_served_s2 > max_grade_served_s).OnlyEnforceIf(
                s2_serving_higher_grades_than_s
            )
            model.Add(max_grade_served_s2 <= max_grade_served_s).OnlyEnforceIf(
                s2_serving_higher_grades_than_s.Not()
            )

            # True if s and s2 are matched AND s2 serves higher grades.
            matched_and_s2_higher_grade = model.NewBoolVar(
                f"{school1}_{school2}_condition"
            )
            model.AddMultiplicationEquality(
                matched_and_s2_higher_grade,
                [s2_serving_higher_grades_than_s, matches[school1][school2]],
            )

            # If the condition is met, the students assigned to school1 must
            # fit within the capacity of school2.
            model.Add(
                sum(students_at_this_school) <= school_capacities[school2]
            ).OnlyEnforceIf(matched_and_s2_higher_grade)

            # The enrollment floor constraint also applies to the feeder school.
            this_school_enrollment_lower_bound = int(
                constants.SCALING[0]
                * np.round(
                    (1 - school_decrease_threshold)
                    * sum(
                        [
                            students_per_grade_per_school[school2]["num_total"][i]
                            for i in constants.GRADE_TO_INDEX.values()
                        ]
                    ),
                    decimals=constants.SCALING[1],
                )
            )
            model.Add(
                constants.SCALING[0] * sum(students_at_this_school)
                >= this_school_enrollment_lower_bound
            ).OnlyEnforceIf(matched_and_s2_higher_grade)


def set_objective_white_nonwhite_dissimilarity(
    model,
    dissim_weight,
    students_per_grade_per_school,
    total_pop_per_cat_across_schools,
    matches,
    grades_offered,
):
    """
    Sets the objective function to minimize racial dissimilarity between white
    and non-white students.

    The function calculates the dissimilarity index, which measures how unevenly
    student populations are distributed across schools. The goal is to make the
    racial composition of each school as close as possible to the district-wide
    composition.

    Args:
        model (cp_model.CpModel): The CP-SAT model instance.
        dissim_weight (float): Weight for the dissimilarity component of the
            objective function (currently not used in favor of a pure
            dissimilarity objective).
        total_per_grade_per_school (defaultdict): Student counts by grade,
            school, and race.
        total_pop_per_cat_across_schools (Counter): Total student counts by race.
        matches (dict): Dictionary of boolean match variables.
        grades_interval_binary (dict): Dictionary of binary grade variables.
    """
    dissim_objective_terms = []
    for school in matches:
        # --- Calculate Student Counts for the New School Configuration ---
        # Sum the number of white students and total students that will be
        # assigned to the building of 'school'.
        sum_s_white = model.NewIntVar(
            0, constants.MAX_TOTAL_STUDENTS, f"{school}_white_students"
        )
        sum_s_total = model.NewIntVar(
            0, constants.MAX_TOTAL_STUDENTS, f"{school}_all_students"
        )
        model.Add(
            sum_s_white
            == sum(
                [
                    students_per_grade_per_school[school]["num_white"][i]
                    * grades_offered[school][i]
                    for i in constants.GRADE_TO_INDEX.values()
                ]
            )
        )
        model.Add(
            sum_s_total
            == sum(
                [
                    students_per_grade_per_school[school]["num_total"][i]
                    * grades_offered[school][i]
                    for i in constants.GRADE_TO_INDEX.values()
                ]
            )
        )

        total_cat_students_at_school = [sum_s_white]
        total_students_at_school = [sum_s_total]

        # Add students from any matched schools (school_2).
        for school_2 in matches[school]:
            if school == school_2:
                continue
            sum_s2_white = model.NewIntVar(
                0, constants.MAX_TOTAL_STUDENTS, f"{school}_{school_2}_white_students"
            )
            sum_s2_total = model.NewIntVar(
                0, constants.MAX_TOTAL_STUDENTS, f"{school}_{school_2}_all_students"
            )

            # Add white students from school_2 for the grades served by school.
            model.Add(
                sum_s2_white
                == sum(
                    [
                        students_per_grade_per_school[school_2]["num_white"][i]
                        * grades_offered[school][i]
                        for i in constants.GRADE_TO_INDEX.values()
                    ]
                )
            ).OnlyEnforceIf(matches[school][school_2])
            model.Add(sum_s2_white == 0).OnlyEnforceIf(matches[school][school_2].Not())
            total_cat_students_at_school.append(sum_s2_white)

            # Add total students from school_2 for the grades served by school.
            model.Add(
                sum_s2_total
                == sum(
                    [
                        students_per_grade_per_school[school_2]["num_total"][i]
                        * grades_offered[school][i]
                        for i in constants.GRADE_TO_INDEX.values()
                    ]
                )
            ).OnlyEnforceIf(matches[school][school_2])
            model.Add(sum_s2_total == 0).OnlyEnforceIf(matches[school][school_2].Not())
            total_students_at_school.append(sum_s2_total)

        # --- Calculate Dissimilarity Index Term for the School ---
        # The dissimilarity index is sum(|group_i/group_total - other_i/other_total|).
        # We need to use integer arithmetic, so we scale values to avoid floats.

        # Scaled total white students at the school.
        scaled_total_cat_students_at_school = model.NewIntVar(
            0, constants.SCALING[0] ** 2, ""
        )
        model.Add(
            scaled_total_cat_students_at_school
            == constants.SCALING[0] * sum(total_cat_students_at_school)
        )

        # Scaled total non-white students at the school.
        scaled_total_non_cat_students_at_school = model.NewIntVar(
            0, constants.SCALING[0] ** 2, ""
        )
        model.Add(
            scaled_total_non_cat_students_at_school
            == constants.SCALING[0]
            * (sum(total_students_at_school) - sum(total_cat_students_at_school))
        )

        # (white students at school / total white students in district)
        cat_ratio_at_school = model.NewIntVar(0, constants.SCALING[0] ** 2, "")
        model.AddDivisionEquality(
            cat_ratio_at_school,
            scaled_total_cat_students_at_school,
            total_pop_per_cat_across_schools["num_white"],
        )

        # (non-white students at school / total non-white students in district)
        non_cat_ratio_at_school = model.NewIntVar(0, constants.SCALING[0] ** 2, "")
        model.AddDivisionEquality(
            non_cat_ratio_at_school,
            scaled_total_non_cat_students_at_school,
            total_pop_per_cat_across_schools["num_total"]
            - total_pop_per_cat_across_schools["num_white"],
        )

        # Absolute difference of the two ratios.
        diff_val = model.NewIntVar(
            -(constants.SCALING[0] ** 2), constants.SCALING[0] ** 2, ""
        )
        model.Add(diff_val == cat_ratio_at_school - non_cat_ratio_at_school)
        obj_term_to_add = model.NewIntVar(0, constants.SCALING[0] ** 2, "")
        model.AddAbsEquality(
            obj_term_to_add,
            diff_val,
        )
        dissim_objective_terms.append(obj_term_to_add)

    # --- Set Final Objective ---
    # The overall objective is to minimize the sum of the absolute differences
    # across all schools, which is equivalent to minimizing the dissimilarity index.
    dissim_val = model.NewIntVar(
        0, constants.MAX_TOTAL_STUDENTS * constants.SCALING[0], ""
    )
    model.Add(dissim_val == sum(dissim_objective_terms))
    model.Minimize(dissim_val)


def set_objective_bh_wa_dissimilarity(
    model,
    dissim_weight,
    total_per_grade_per_school,
    total_pop_per_cat_across_schools,
    matches,
    grades_interval_binary,
):
    """
    Sets the objective function to minimize racial dissimilarity between
    Black/Hispanic (BH) and White/Asian (WA) students.

    This function is analogous to set_objective_white_nonwhite_dissimilarity,
    but it uses different racial groupings (BH vs. WA) to calculate the
    dissimilarity index.

    Args:
        model (cp_model.CpModel): The CP-SAT model instance.
        dissim_weight (float): Weight for the dissimilarity component of the
            objective function.
        total_per_grade_per_school (defaultdict): Student counts by grade,
            school, and race.
        total_pop_per_cat_across_schools (Counter): Total student counts by race.
        matches (dict): Dictionary of boolean match variables.
        grades_interval_binary (dict): Dictionary of binary grade variables.
    """
    dissim_objective_terms = []
    students_switching_terms = []
    for school in matches:
        # --- Calculate Student Counts for the New School Configuration ---
        # Sum the number of BH and WA students that will be assigned to the
        # building of 'school'.
        sum_s_bh = sum(
            [
                (
                    total_per_grade_per_school[school]["num_black"][i]
                    + total_per_grade_per_school[school]["num_hispanic"][i]
                )
                * grades_interval_binary[school][i]
                for i in constants.GRADE_TO_INDEX.values()
            ]
        )
        sum_s_wa = sum(
            [
                (
                    total_per_grade_per_school[school]["num_white"][i]
                    + total_per_grade_per_school[school]["num_asian"][i]
                )
                * grades_interval_binary[school][i]
                for i in constants.GRADE_TO_INDEX.values()
            ]
        )

        total_bh_students_at_school = [sum_s_bh]
        total_wa_students_at_school = [sum_s_wa]

        # Add students from any matched schools (school_2).
        for school_2 in matches[school]:
            if school == school_2:
                continue

            sum_s2_bh = (
                sum(
                    [
                        (
                            total_per_grade_per_school[school_2]["num_black"][i]
                            + total_per_grade_per_school[school_2]["num_hispanic"][i]
                        )
                        * grades_interval_binary[school][i]
                        for i in constants.GRADE_TO_INDEX.values()
                    ]
                )
                * matches[school][school_2]
            )
            total_bh_students_at_school.append(sum_s2_bh)

            sum_s2_wa = (
                sum(
                    [
                        (
                            total_per_grade_per_school[school_2]["num_white"][i]
                            + total_per_grade_per_school[school_2]["num_asian"][i]
                        )
                        * grades_interval_binary[school][i]
                        for i in constants.GRADE_TO_INDEX.values()
                    ]
                )
                * matches[school][school_2]
            )
            total_wa_students_at_school.append(sum_s2_wa)

        # --- Calculate Dissimilarity Index Term for the School ---
        # Scaled total BH students at the school.
        scaled_total_bh_students_at_school = model.NewIntVar(
            0, constants.SCALING[0] ** 2, ""
        )
        model.Add(
            scaled_total_bh_students_at_school
            == constants.SCALING[0] * sum(total_bh_students_at_school)
        )

        # Scaled total WA students at the school.
        scaled_total_wa_students_at_school = model.NewIntVar(
            0, constants.SCALING[0] ** 2, ""
        )
        model.Add(
            scaled_total_wa_students_at_school
            == constants.SCALING[0] * sum(total_wa_students_at_school)
        )

        # (BH students at school / total BH students in district)
        bh_ratio_at_school = model.NewIntVar(0, constants.SCALING[0] ** 2, "")
        model.AddDivisionEquality(
            bh_ratio_at_school,
            scaled_total_bh_students_at_school,
            total_pop_per_cat_across_schools["num_black"]
            + total_pop_per_cat_across_schools["num_hispanic"],
        )

        # (WA students at school / total WA students in district)
        wa_ratio_at_school = model.NewIntVar(0, constants.SCALING[0] ** 2, "")
        model.AddDivisionEquality(
            wa_ratio_at_school,
            scaled_total_wa_students_at_school,
            total_pop_per_cat_across_schools["num_white"]
            + total_pop_per_cat_across_schools["num_asian"],
        )

        # Absolute difference of the two ratios.
        diff_val = model.NewIntVar(
            -(constants.SCALING[0] ** 2), constants.SCALING[0] ** 2, ""
        )
        model.Add(diff_val == bh_ratio_at_school - wa_ratio_at_school)
        obj_term_to_add = model.NewIntVar(0, constants.SCALING[0] ** 2, "")
        model.AddAbsEquality(
            obj_term_to_add,
            diff_val,
        )
        dissim_objective_terms.append(obj_term_to_add)

    # --- Set Final Objective ---
    # Minimize the sum of the absolute differences (the dissimilarity index).
    dissim_val = model.NewIntVar(
        0, constants.MAX_TOTAL_STUDENTS * constants.SCALING[0], ""
    )
    model.Add(dissim_val == sum(dissim_objective_terms))
    model.Minimize(dissim_val)


def solve_and_output_results(
    state="CA",
    district_id="0602160",
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
    print(f"Matches:\n{matches}\n\n\ngrades_interval_binary: {grades_interval_binary}")

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
    solver.parameters.max_time_in_seconds = constants.MAX_SOLVER_TIME

    # Adding parallelism
    solver.parameters.num_search_workers = constants.NUM_SOLVER_THREADS

    status = solver.Solve(model)

    curr_output_dir = f"data/results/{batch}/{state}/{district_id}/{interdistrict}_{school_decrease_threshold}_{status}_{solver.WallTime()}_{solver.NumBranches()}_{solver.NumConflicts()}"
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
            header.write_json(os.path.join(curr_output_dir, "output.json"), result)


if __name__ == "__main__":
    solve_and_output_results()
