from ortools.sat.python import cp_model
import mergers_core.utils.header as header
import mergers_core.models.constants as constants
from mergers_core.models.model_utils import (
    output_solver_solution,
    compute_dissimilarity_metrics,
    compute_population_consistencies,
)
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
import os
from fractions import Fraction
import mergers_core.models.config as config


def _load_and_filter_nces_schools(
    filename: os.PathLike, districts: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load NCES school data from a CSV and filter by district.

    Args:
        filename: Path to the NCES school data CSV file.
        districts: A list of district IDs to filter the schools by.

    Returns:
        A tuple containing two DataFrames:
        - The filtered DataFrame containing only schools from the specified districts.
        - The original, unfiltered DataFrame.
    """
    dataframe = pd.read_csv(filename, dtype={"NCESSCH": str})
    # District ID is the first 7 characters of the NCES school ID
    dataframe["leaid"] = dataframe["NCESSCH"].str[:7]

    filtered = dataframe[dataframe["leaid"].isin(districts)]

    filtered = filtered.reset_index(drop=True)

    return filtered, dataframe


def load_and_process_data(
    config: config.Config,
) -> tuple[
    dict[str, int],
    dict[str, list[str]],
    dict[str, dict[str, list[int]]],
    Counter[str],
    pd.DataFrame,
]:
    """Loads and preprocesses all necessary data for the CP-SAT model.

    This function reads school enrollment data, capacity data, and permissible
    merger information from CSV and JSON files. It filters this data for the
    specified district and other districts involved in potential interdistrict
    mergers. It then calculates and aggregates student counts by race and grade.

    Args:
        config: The configuration for this run.

    Returns:
        A tuple containing:
            - school_capacities: Maps school ID to its student capacity.
            - permissible_matches: Maps each school ID to a list of
              other school IDs it is allowed to merge with.
            - students_per_grade_per_school: Nested dictionary
              mapping school ID and race to a list of student counts per grade.
            - total_across_schools_by_category: Counts of total
              students per racial category across all involved schools.
            - df_schools_in_play: DataFrame containing detailed
              enrollment and demographic data for all schools involved in the
              potential mergers.
    """
    # Load school enrollment data
    enrollment_file = os.path.join(
        "data", "solver_files", "2122", config.district.state, "school_enrollments.csv"
    )
    df_schools, _ = _load_and_filter_nces_schools(enrollment_file, [config.district.id])
    print(f"Loaded {len(df_schools)} schools.")

    unique_schools = list(set(df_schools["NCESSCH"].tolist()))

    # Load permissible merger data
    if config.interdistrict:
        merger_file = os.path.join(
            "data",
            "solver_files",
            "2122",
            config.district.state,
            "between_within_district_allowed_mergers.json",
        )
        permissible_matches = header.read_json(merger_file)
        districts_involved = {
            school[:7]
            for school in unique_schools
            for school_2 in permissible_matches.get(school, [])
        }
    else:
        merger_file = os.path.join(
            "data",
            "solver_files",
            "2122",
            config.district.state,
            "within_district_allowed_mergers.json",
        )
        permissible_matches = header.read_json(merger_file)
        districts_involved = {config.district.id}

    # Load and merge capacity data
    capacity_file = "data/school_data/21_22_school_capacities.csv"
    df_capacities, _ = _load_and_filter_nces_schools(
        capacity_file, list(districts_involved)
    )

    df_schools_in_play = pd.merge(
        df_schools,
        df_capacities[["NCESSCH", "student_capacity"]],
        on="NCESSCH",
        how="left",
    )
    school_capacities = (
        pd.Series(df_capacities.student_capacity.values, index=df_capacities.NCESSCH)
        .astype(int)
        .to_dict()
    )

    # Vectorized aggregation of student counts
    students_per_grade_per_school = defaultdict(lambda: defaultdict(list))
    total_across_schools_by_category = Counter()

    for race in constants.RACE_KEYS.values():
        grade_columns = [f"{race}_{grade}" for grade in constants.GRADE_TO_INDEX]

        # Sum population for each category across all schools
        total_across_schools_by_category[race] = int(
            df_schools_in_play[grade_columns].sum().sum()
        )

        # Group by school and create the nested dictionary
        for ncessch, row in df_schools_in_play.iterrows():
            students_per_grade_per_school[row["NCESSCH"]][race] = (
                row[grade_columns].astype(int).tolist()
            )

    return (
        school_capacities,
        permissible_matches,
        students_per_grade_per_school,
        total_across_schools_by_category,
        df_schools_in_play,
    )


def initialize_variables(
    model: cp_model.CpModel, df_schools_in_play: pd.DataFrame
) -> tuple[dict[str, dict[str, cp_model.IntVar]], dict[str, list[cp_model.IntVar]]]:
    """Initializes the core variables for the CP-SAT model.

    This function creates the decision variables that the solver will manipulate
    to find an optimal solution. These include variables for school matches and
    grade assignments.

    Args:
        model: The CP-SAT model instance.
        df_schools_in_play: DataFrame containing data for all
            schools to be considered in the model.

    Returns:
        A tuple containing:
            - matches: A nested dictionary of boolean variables, where
              matches[s1][s2] is true if school s1 and s2 are merged.
            - grades_interval_binary: A dictionary mapping each school ID
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
    grades_start = dict()
    grades_end = dict()
    grades_duration = dict()
    grades_interval = dict()
    grades_interval_binary = dict()  # Binary representation of grades served.
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
            model.AddHint(matches[school][school_2], school == school_2)

    return (
        matches,
        grades_interval_binary,
    )


def _get_students_at_school(
    model: cp_model.CpModel,
    matches: dict[str, dict[str, cp_model.IntVar]],
    grades_interval_binary: dict[str, list[cp_model.IntVar]],
    school: str,
    students_per_grade_per_school: dict[str, dict[str, list[int]]],
) -> list[cp_model.IntVar]:
    """Calculates the number of students that will be assigned to a school building.

    This function determines the total student population for a given school
    building ('school') based on a potential merger scenario. It calculates this
    by summing two groups of students:
    1.  Students from the original 'school' who are in the grade levels that
        the school will serve after the merger.
    2.  Students from a merged school ('school2') who are in the grade levels
        that 'school' will serve. This is calculated for each potential merger.

    Args:
        model: The CP-SAT model instance.
        matches: A nested dictionary of boolean variables, where
            matches[s1][s2] is true if school s1 and s2 are merged.
        grades_interval_binary: A dictionary mapping each school ID to a list
            of binary variables, one for each grade level, indicating if the
            school serves that grade.
        school: The NCESSCH ID of the school building for which to
            calculate the student population.
        students_per_grade_per_school: A nested dictionary
            containing the number of students for each grade and racial
            category within each school.

    Returns:
        A list of CP-SAT integer variables representing the different
        groups of students that will make up the new population of the school
        building. The sum of this list would represent the total enrollment.
    """
    # Calculate the base number of students this school will serve from its
    # original student body based on its new grade assignments.
    students_at_school = sum(
        [
            students_per_grade_per_school[school]["num_total"][grade]
            * grades_interval_binary[school][grade]
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
                        * grades_interval_binary[school][i]
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
    model: cp_model.CpModel,
    config: config.Config,
    school_capacities: dict[str, int],
    students_per_grade_per_school: dict[str, dict[str, list[int]]],
    permissible_matches: dict[str, list[str]],
    matches: dict[str, dict[str, cp_model.IntVar]],
    grades_interval_binary: dict[str, list[cp_model.IntVar]],
) -> None:
    """Defines the set of rules and limitations for the school merger problem.

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
        moved up grades) must stay within the enrollment bounds:
        g_s′^end>g_s^end ∧ Ms,s′=1 ⇒
            p_min⋅∑_g∈G(Es′,g)
            ≤ ∑_g∈G(∑_s′′∈S(Ms,s′′))⋅Rs,g⋅Es′′,g ≤
            Capacity(s′)
        ∀ s,s′∈S s≠s′

        Every grade must be served by one school in every merger:
        ∑s'∈S (Ms,s' * ∑g∈G Rs',g) = |G| ∀ s∈S

    Args:
        model: The CP-SAT model instance.
        config: The configuration for this run.
        school_capacities: Maps school ID to its student capacity.
        students_per_grade_per_school: Student counts by grade,
            school, and race.
        permissible_matches: Defines which schools are allowed to merge.
        matches: Dictionary of boolean match variables.
        grades_interval_binary: Dictionary of binary grade variables.
    """

    students_at_each_school = {
        school: sum(
            _get_students_at_school(
                model,
                matches,
                grades_interval_binary,
                school,
                students_per_grade_per_school,
            )
        )
        for school in matches
    }

    # --- Symmetry and transitivity ---
    # Ensures that if A is matched with B, B is matched with A.
    # Also enforces transitivity for 3-way mergers: if A-B and B-C, then A-C.
    # This creates cohesive merged groups.
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
        model.Add(
            students_at_each_school[school1]
            <= round(
                (1 + config.school_increase_threshold) * school_capacities[school1]
            )
        )

        school_current_population = sum(
            [
                students_per_grade_per_school[school1]["num_total"][i]
                for i in constants.GRADE_TO_INDEX.values()
            ]
        )
        enrollment_lower_bound = int(
            constants.SCALING[0]
            * np.round(
                (1 - config.school_decrease_threshold) * school_current_population,
                decimals=constants.SCALING[1],
            )
        )
        model.Add(
            constants.SCALING[0] * students_at_each_school[school1]
            >= enrollment_lower_bound
        )

    for school1 in matches:
        for school2 in matches[school1]:
            # --- Permissible Match Constraint ---
            # A school can only be matched with schools from its pre-approved list.
            # Primarily used for adjacency checking.

            # Ignore the linter warning: using `is False` causes shenanigans that break
            # everything. Oh, python...
            model.Add(matches[school1][school2] == False).OnlyEnforceIf(
                school2 not in permissible_matches[school1]
            )

            # --- No Grade Overlap Constraint ---
            # If two schools are merged, they cannot serve the same grade levels.
            # It is permitted for neither school to serve a grade level because of
            # triple mergers. In those cases, the grade completeness constriant ensures
            # that every grade is served.
            if school1 != school2:
                for i in range(len(constants.GRADE_TO_INDEX)):
                    model.Add(
                        grades_interval_binary[school1][i]
                        + grades_interval_binary[school2][i]
                        <= 1
                    ).OnlyEnforceIf(matches[school1][school2])

    for school1 in matches:
        # --- Grade Completeness Constraint ---
        # Ensure that a group of merged schools combined serve a full set of grades.

        # Calculate the number of grade levels this school will offer postmerger.
        num_grades_in_school = sum(grades_interval_binary[school1])
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
                    num_grades_s2 == sum(grades_interval_binary[school2])
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
        # If school s (feeder) and school s' (receiving) are merged AND s' serves higher
        # grades than s, then the total number of students assigned to s must not exceed
        # the capacity of s'.
        max_grade_served_s = model.NewIntVar(
            0, max(constants.GRADE_TO_INDEX.values()), f"{school1}_grade_max"
        )
        all_grades_served_s = []
        for grade in constants.GRADE_TO_INDEX.values():
            all_grades_served_s.append(grades_interval_binary[school1][grade] * grade)
        model.AddMaxEquality(max_grade_served_s, all_grades_served_s)

        for school2 in matches[school1]:
            if school1 == school2:
                continue

            # Determine if s2 serves higher grades than s.
            max_grade_served_s2 = model.NewIntVar(
                0,
                max(constants.GRADE_TO_INDEX.values()),
                f"{school1}_{school2}_grade_max",
            )
            all_grades_served_s2 = []
            for grade in constants.GRADE_TO_INDEX.values():
                all_grades_served_s2.append(
                    grades_interval_binary[school2][grade] * grade
                )
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
                students_at_each_school[school1]
                <= round(
                    (1 + config.school_increase_threshold) * school_capacities[school2]
                )
            ).OnlyEnforceIf(matched_and_s2_higher_grade)

            # The enrollment floor constraint also applies to the feeder school.
            feeder_school_enrollment_lower_bound = int(
                constants.SCALING[0]
                * np.round(
                    (1 - config.school_decrease_threshold)
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
                constants.SCALING[0] * students_at_each_school[school1]
                >= feeder_school_enrollment_lower_bound
            ).OnlyEnforceIf(matched_and_s2_higher_grade)


def calculate_dissimilarity(
    model: cp_model.CpModel,
    students_per_grade_per_school: dict[str, dict[str, list[int]]],
    total_across_schools_by_category: Counter[str],
    matches: dict[str, dict[str, cp_model.IntVar]],
    grades_interval_binary: dict[str, list[cp_model.IntVar]],
    groups_a: list[str],
    groups_b: list[str],
) -> cp_model.LinearExpr:
    """Returns a LinearExpr that calculates the dissimilarity index between two groups.

    The dissimilarity index is the sum of this absolute difference for all schools:
    ┃ group_school   non_group_school ┃
    ┃ ———————————— - ———————————————— ┃
    ┃ group_total     non_group_total ┃

    Args:
        model: The CP-SAT model instance.
        students_per_grade_per_school: Student counts by grade,
            school, and race.
        total_across_schools_by_category: Total student counts by race.
        matches: Dictionary of boolean match variables.
        grades_interval_binary: Dictionary of binary grade variables.
        groups_a: Groups of students for the first half of a dissimilarity term.
        groups_b: Groups of students for the second half of a dissimilarity term.

    Returns:
        A LinearExpr representing the dissimilarity index.
    """

    for idx in range(len(groups_a)):
        groups_a[idx] = "num_" + groups_a[idx]
    for idx in range(len(groups_b)):
        groups_b[idx] = "num_" + groups_b[idx]

    dissimilarity_terms = []
    for school in matches:
        # --- Calculate Student Counts for the New School config.Configuration ---
        # Sum the number of A and B students that will be assigned to the
        # building of 'school'.
        total_a_students_at_school = []
        for school_2 in matches[school]:
            students_at_s2_a = sum(
                sum(
                    students_per_grade_per_school[school_2][group][i]
                    for group in groups_a
                )
                * grades_interval_binary[school][i]
                for i in constants.GRADE_TO_INDEX.values()
            )

            sum_school2_a = model.NewIntVar(
                0,
                constants.MAX_TOTAL_STUDENTS,
                f"sum_s2_group_a_{school},{school_2}",
            )
            model.Add(sum_school2_a == students_at_s2_a).OnlyEnforceIf(
                matches[school][school_2]
            )
            model.Add(sum_school2_a == 0).OnlyEnforceIf(matches[school][school_2].Not())
            total_a_students_at_school.append(sum_school2_a)

        total_b_students_at_school = []
        for school_2 in matches[school]:
            students_at_s2_b = sum(
                sum(
                    students_per_grade_per_school[school_2][group][i]
                    for group in groups_b
                )
                * grades_interval_binary[school][i]
                for i in constants.GRADE_TO_INDEX.values()
            )

            sum_school2_b = model.NewIntVar(
                0, constants.MAX_TOTAL_STUDENTS, f"sum_s2_wa_{school},{school_2}"
            )
            model.Add(sum_school2_b == students_at_s2_b).OnlyEnforceIf(
                matches[school][school_2]
            )
            model.Add(sum_school2_b == 0).OnlyEnforceIf(matches[school][school_2].Not())
            total_b_students_at_school.append(sum_school2_b)

        # --- Calculate Dissimilarity Index Term for the School ---
        # Scaled total A students at the school.
        scaled_total_a_students_at_school = model.NewIntVar(
            0,
            constants.SCALING[0] * constants.MAX_TOTAL_STUDENTS,
            f"{school}_scaled_total_a",
        )
        model.Add(
            scaled_total_a_students_at_school
            == constants.SCALING[0] * sum(total_a_students_at_school)
        )

        # Scaled total B students at the school.
        scaled_total_b_students_at_school = model.NewIntVar(
            0,
            constants.SCALING[0] * constants.MAX_TOTAL_STUDENTS,
            f"{school}_scaled_total_b",
        )
        model.Add(
            scaled_total_b_students_at_school
            == constants.SCALING[0] * sum(total_b_students_at_school)
        )

        # (A students at school / total A students in district)
        a_ratio_at_school = model.NewIntVar(
            0, constants.SCALING[0], f"{school}_a_ratio"
        )
        model.AddDivisionEquality(
            a_ratio_at_school,
            scaled_total_a_students_at_school,
            total_across_schools_by_category["num_black"]
            + total_across_schools_by_category["num_hispanic"],
        )

        # (B students at school / total B students in district)
        b_ratio_at_school = model.NewIntVar(
            0, constants.SCALING[0], f"{school}_b_ratio"
        )
        model.AddDivisionEquality(
            b_ratio_at_school,
            scaled_total_b_students_at_school,
            total_across_schools_by_category["num_white"]
            + total_across_schools_by_category["num_asian"],
        )

        term = model.NewIntVar(0, constants.SCALING[0], f"{school}_dissimilarity_term")
        model.AddAbsEquality(
            term,
            a_ratio_at_school - b_ratio_at_school,
        )
        dissimilarity_terms.append(term)

    return sum(dissimilarity_terms)


def _sort_sequence(
    model: cp_model.CpModel,
    sequence: list[cp_model.IntVar],
    max_val: int,
    name_prefix: str,
) -> list[cp_model.IntVar]:
    """Given a list of IntVars, returns a new list of IntVars with the same values, but in
    sorted order.

    Args:
        model: The CP-SAT model instance.
        sequence: A list of IntVars.
        max_val: The maximum value of the input IntVars.
        name_prefix: A prefix for the variable names.

    Returns:
        A sorted list of IntVars.
    """
    sorted_vars = [
        model.NewIntVar(0, max_val, f"{name_prefix}_sorted_{i}")
        for i in range(len(sequence))
    ]

    # Enforce sorted order
    for i in range(len(sequence) - 1):
        model.Add(sorted_vars[i] <= sorted_vars[i + 1])

    permutation_indices = [
        model.NewIntVar(0, len(sequence) - 1, f"{name_prefix}_perm_{i}")
        for i in range(len(sequence))
    ]
    model.AddAllDifferent(permutation_indices)

    for i in range(len(sequence)):
        # sorted_vars[i] == sequence[permutation_indices[i]]
        model.AddElement(permutation_indices[i], sequence, sorted_vars[i])

    return sorted_vars


def _median(
    model: cp_model.CpModel,
    sequence: list[cp_model.IntVar],
    max_val: int,
    name_prefix: str,
) -> cp_model.IntVar:
    """Returns the median of the input sequence.

    Args:
        model: The CP-SAT model instance.
        sequence: A list of IntVars.
        max_val: The maximum value of the input IntVars.
        name_prefix: A prefix for the variable names.

    Returns:
        The median of the input sequence.
    """
    sorted_sequence = _sort_sequence(
        model,
        sequence,
        max_val,
        name_prefix,
    )

    if len(sequence) % 2 == 1:
        index = len(sequence) // 2
        return sorted_sequence[index]

    sum_var = model.NewIntVar(0, max_val * 2, f"{name_prefix}_sum")
    model.Add(
        sum_var
        == sorted_sequence[len(sequence) // 2 - 1] + sorted_sequence[len(sequence) // 2]
    )
    median_var = model.NewIntVar(0, max_val, f"{name_prefix}_median")
    model.AddDivisionEquality(
        median_var,
        sum_var,
        2,
    )
    return median_var


def setup_population_capacity(
    model: cp_model.CpModel,
    config: config.Config,
    matches: dict[str, dict[str, cp_model.IntVar]],
    grades_interval_binary: dict[str, list[cp_model.IntVar]],
    students_per_grade_per_school: dict[str, dict[str, list[int]]],
    school_capacities: dict[str, int],
) -> cp_model.IntVar | None:
    """Returns a LinearExpr that represents the population consistency index.

    This index is the average distance from the mean population for each
    school's population.

    Args:
        model: The CP-SAT model instance.
        matches: A nested dictionary of boolean variables, where
            matches[s1][s2] is true if school s1 and s2 are merged.
        grades_interval_binary: Dictionary of binary grade variables.
        students_per_grade_per_school: Student counts by grade,
            school, and race.
        school_capacities: Dictionary of school capacities.

    Returns:
        A LinearExpr that represents the population consistency index.
    """
    school_population_variables = {
        school: sum(
            _get_students_at_school(
                model,
                matches,
                grades_interval_binary,
                school,
                students_per_grade_per_school,
            )
        )
        for school in matches
    }

    percentages: dict[str, cp_model.IntVar] = dict()
    for school in matches:
        percentage = model.NewIntVar(
            0,
            constants.SCALING[0],
            f"{school}_capacity_percentage",
        )

        numerator_expr = constants.SCALING[0] * school_population_variables[school]
        numerator_var = model.NewIntVar(
            0,
            constants.SCALING[0] * constants.MAX_TOTAL_STUDENTS,
            f"{school}_numerator_var",
        )
        model.Add(numerator_var == numerator_expr)
        model.AddDivisionEquality(
            percentage,
            numerator_var,
            school_capacities[school],
        )
        percentages.update({school: percentage})

    average_percentage = model.NewIntVar(0, constants.SCALING[0], "average_percentage")
    sum_percentages_var = model.NewIntVar(
        0,
        len(percentages) * constants.SCALING[0] * constants.MAX_TOTAL_STUDENTS,
        "sum_percentages_var",
    )
    model.Add(sum_percentages_var == sum(percentages.values()))
    model.AddDivisionEquality(
        average_percentage,
        sum_percentages_var,
        len(percentages),
    )

    median = None
    if config.population_consistency_metric == "median":
        median = _median(
            model,
            list(percentages.values()),
            constants.SCALING[0] * constants.MAX_TOTAL_STUDENTS,
            "pop_capacity",
        )

    differences = []
    for school in matches:
        difference = model.NewIntVar(0, constants.SCALING[0], f"{school}_difference")
        model.AddAbsEquality(difference, percentages[school] - average_percentage)
        differences.append(difference)

    average_difference = model.NewIntVar(0, constants.SCALING[0], "average_difference")
    sum_differences_var = model.NewIntVar(
        0,
        len(differences) * constants.SCALING[0],
        "sum_differences_var",
    )
    model.Add(sum_differences_var == sum(differences))
    model.AddDivisionEquality(
        average_difference,
        sum_differences_var,
        len(differences),
    )

    median_difference = None
    if config.population_consistency_metric == "median_difference":
        median_difference = _median(
            model, differences, constants.SCALING[0], "median_diff"
        )

    return {
        "median": median,
        "average_difference": average_difference,
        "median_difference": median_difference,
    }[config.population_consistency_metric]


def set_objective(
    *,
    model: cp_model.CpModel,
    config: config.Config,
    dissimilarity_index: cp_model.IntVar,
    population_consistency_metric: cp_model.IntVar,
    pre_dissimilarity: float,
    pre_population_consistency: float,
) -> None:
    """Sets the multi-objective function for the solver.

    Args:
        model: The CP-SAT model instance.
        config: The configuration for this run.
        dissimilarity_index: The variable representing the dissimilarity index.
        population_consistency_metric: The variable representing the population
            consistency metric.
        pre_dissimilarity: The dissimilarity index before optimization.
        pre_population_consistency: The population consistency metric before
            optimization.
    """
    if config.minimize:
        optimize_function = model.Minimize
    else:
        optimize_function = model.Maximize

    obj = "⇣" if config.minimize else "⇡"

    if config.population_consistency_weight == 0:
        print(f"Objective: {obj} dissimilarity ({config.dissimilarity_flavor})")
        optimize_function(dissimilarity_index)
        return

    if config.dissimilarity_weight == 0:
        print(f"Objective: {obj} population consistency")
        optimize_function(population_consistency_metric)
        return

    # Handle case where a pre-computation is zero to avoid division errors
    if pre_population_consistency == 0 or pre_dissimilarity == 0:
        starting_ratio = Fraction(1, 1)
    else:
        starting_ratio = Fraction(pre_dissimilarity / pre_population_consistency)

    # This ratio balances the weights provided by the user with the initial
    # values of the metrics themselves, preventing one metric from dominating
    # the objective function simply due to its scale.
    ratio = Fraction(
        int(config.dissimilarity_weight * constants.SCALING[0]),
        int(config.population_consistency_weight * constants.SCALING[0]),
    )
    ratio = ratio * starting_ratio
    ratio = ratio.limit_denominator(1000)

    print(
        f"Objective: {obj}"
        f" {ratio.numerator} * dissimilarity ({config.dissimilarity_flavor})"
        f" + {ratio.denominator} * population consistency"
    )
    optimize_function(
        ratio.numerator * dissimilarity_index
        + ratio.denominator * population_consistency_metric
    )


def solve_and_output_results(
    config: config.Config,
    s3_bucket: str = "s3://school-mergers/",
) -> None:
    """Main function to run the school merger optimization.

    This function orchestrates the entire process:
    1.  Loads and processes the data.
    2.  Calculates initial dissimilarity and population consistency metrics.
    3.  Initializes the CP-SAT model and variables.
    4.  Sets the constraints for the model.
    5.  Defines and sets the multi-objective function.
    6.  Runs the solver.
    7.  Outputs the results.

    Args:
        config: The configuration for this run.
        s3_bucket: The S3 bucket to write the results to.
    """
    print(f"Settings: {config}")

    (
        school_capacities,
        permissible_matches,
        students_per_grade_per_school,
        total_across_schools_by_category,
        df_schools_in_play,
    ) = load_and_process_data(config)

    # --- Calculate Initial Metrics ---
    initial_school_clusters = list(df_schools_in_play["NCESSCH"])
    initial_num_per_cat_per_school = defaultdict(Counter)
    for school_id, school_data in students_per_grade_per_school.items():
        for race, grade_counts in school_data.items():
            initial_num_per_cat_per_school[race][school_id] = sum(grade_counts)

    pre_dissim_wnw, pre_dissim_bh_wa = compute_dissimilarity_metrics(
        initial_school_clusters, initial_num_per_cat_per_school
    )
    pre_population_consistencies = compute_population_consistencies(
        df_schools_in_play, initial_num_per_cat_per_school
    )

    if config.dissimilarity_flavor == "bh_wa":
        pre_dissimilarity = pre_dissim_bh_wa
        groups_a = ["black", "hispanic"]
        groups_b = ["white", "asian"]
    else:
        pre_dissimilarity = pre_dissim_wnw
        groups_a = [
            "asian",
            "black",
            "hispanic",
            "native",
            "not_specified",
            "pacific_islander",
            "total",
            "two_or_more",
        ]
        groups_b = ["white"]

    pre_population_consistency = pre_population_consistencies[
        config.population_consistency_metric
    ]

    # Create the cp model
    model = cp_model.CpModel()

    (
        matches,
        grades_interval_binary,
    ) = initialize_variables(model=model, df_schools_in_play=df_schools_in_play)

    set_constraints(
        model=model,
        config=config,
        school_capacities=school_capacities,
        students_per_grade_per_school=students_per_grade_per_school,
        permissible_matches=permissible_matches,
        matches=matches,
        grades_interval_binary=grades_interval_binary,
    )

    dissimilarity_index = calculate_dissimilarity(
        model=model,
        students_per_grade_per_school=students_per_grade_per_school,
        total_across_schools_by_category=total_across_schools_by_category,
        matches=matches,
        grades_interval_binary=grades_interval_binary,
        groups_a=groups_a,
        groups_b=groups_b,
    )

    population_consistency_metric = setup_population_capacity(
        model=model,
        config=config,
        matches=matches,
        grades_interval_binary=grades_interval_binary,
        students_per_grade_per_school=students_per_grade_per_school,
        school_capacities=school_capacities,
    )

    set_objective(
        model=model,
        config=config,
        dissimilarity_index=dissimilarity_index,
        population_consistency_metric=population_consistency_metric,
        pre_dissimilarity=pre_dissimilarity,
        pre_population_consistency=pre_population_consistency,
    )
    print("Solving ...")
    solver = cp_model.CpSolver()

    # Sets a time limit for solver
    solver.parameters.max_time_in_seconds = constants.MAX_SOLVER_TIME

    # Adding parallelism
    solver.parameters.num_search_workers = constants.NUM_SOLVER_THREADS

    # solver.parameters.log_search_progress = True

    status = solver.Solve(model)

    this_result_dirname = (
        f"{config.school_decrease_threshold}_"
        f"{config.dissimilarity_weight},{config.population_consistency_weight}_"
        f"{constants.STATUSES[status]}_{solver.WallTime():.5f}s_"
        f"{solver.NumBranches()}_{solver.NumConflicts()}"
    )
    output_dir = f"data/results/{config.district.state}/{config.district.id}/{this_result_dirname}"
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Status is {constants.STATUSES[status]}.")
        output_solver_solution(
            config=config,
            solver=solver,
            matches=matches,
            grades_interval_binary=grades_interval_binary,
            df_schools_in_play=df_schools_in_play,
            output_dir=output_dir,
            s3_bucket=s3_bucket,
            pre_dissim_wnw=pre_dissim_wnw,
            pre_dissim_bh_wa=pre_dissim_bh_wa,
            pre_population_consistencies=pre_population_consistencies,
        )

    else:
        print("Status: ", status)
        print(model.Validate())
        result = {"status": status}
        df_r = pd.DataFrame(result, index=[0])
        if config.write_to_s3:
            df_r.to_csv(s3_bucket + output_dir, index=False)
        else:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            header.write_json(os.path.join(output_dir, "output.json"), result)


if __name__ == "__main__":
    # solve_and_output_results(
    #     config.Config.custom_config(
    #         district=config.District("MA", "2508700"),
    #         dissimilarity_weight=0,
    #         population_consistency_weight=1,
    #         population_consistency_metric="average_difference",
    #         dissimilarity_flavor="bh_wa",
    #     )
    # )
    solve_and_output_results(config.Config("data/sweep_configs/configs.csv"))
