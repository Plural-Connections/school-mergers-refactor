import mergers_core.utils.header as header
import geopandas as gpd
import us
import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path
from collections import defaultdict

CCD_FILES = {
    "2016-2017": (
        "ccd_sch_029_1617_w_1a_11212017.csv",
        "ccd_sch_052_1617_l_2a_11212017.csv",
    ),
    "2017-2018": (
        "ccd_sch_029_1718_w_1a_083118.csv",
        "ccd_SCH_052_1718_l_1a_083118.csv",
    ),
    "2018-2019": (
        "ccd_sch_029_1819_w_1a_091019.csv",
        "ccd_SCH_052_1819_l_1a_091019.csv",
    ),
    "2019-2020": (
        "ccd_sch_029_1920_w_1a_082120.csv",
        "ccd_SCH_052_1920_l_1a_082120.csv",
    ),
    "2020-2021": (
        "ccd_sch_029_2021_w_1a_080621.csv",
        "ccd_SCH_052_2021_l_1a_080621.csv",
    ),
    "2021-2022": (
        "ccd_sch_029_2122_w_1a_071722.csv",
        "ccd_SCH_052_2122_l_1a_071722.csv",
    ),
}

RACE_KEYS = {
    "White": "num_white",
    "Black or African American": "num_black",
    "Hispanic/Latino": "num_hispanic",
    "American Indian or Alaska Native": "num_native",
    "Asian": "num_asian",
    "Native Hawaiian or Other Pacific Islander": "num_pacific_islander",
    "Two or more races": "num_two_or_more",
    "Not Specified": "num_not_specified",
    "Total": "num_total",
}

GRADE_KEYS = {
    "Grade 13": "13",
    "Grade 4": "4",
    "Grade 6": "6",
    "Grade 8": "8",
    "Kindergarten": "KG",
    "Grade 9": "9",
    "Ungraded": "UG",
    "Grade 12": "12",
    "Grade 1": "1",
    "Grade 5": "5",
    "Grade 10": "10",
    "Pre-Kindergarten": "PK",
    "Grade 11": "11",
    "Grade 7": "7",
    "Not Specified": "NS",
    "Grade 3": "3",
    "Grade 2": "2",
}


def determine_school_capacities(
    input_dir="data/school_data/ccd/{}/{}",
    output_file="~/Downloads/21_22_school_capacities.csv",
):
    years = list(CCD_FILES.keys())
    years.reverse()
    df = pd.DataFrame()

    for year in years:
        print(year)
        df_curr = pd.read_csv(
            input_dir.format(year, CCD_FILES[year][0]),
            encoding="ISO-8859-1",
            dtype=str,
        )
        df_curr_elem = df_curr[df_curr["LEVEL"] == "Elementary"][["NCESSCH"]]
        df_curr_enrollment = pd.read_csv(
            input_dir.format(year, CCD_FILES[year][1]),
            encoding="ISO-8859-1",
            dtype=str,
        )
        df_curr_enrollment = df_curr_enrollment[
            ["NCESSCH", "RACE_ETHNICITY", "STUDENT_COUNT", "TOTAL_INDICATOR"]
        ]
        df_curr_enrollment = (
            df_curr_enrollment[
                df_curr_enrollment["TOTAL_INDICATOR"].isin(
                    ["Derived - Education Unit Total minus Adult Education Count"]
                )
            ][["NCESSCH", "STUDENT_COUNT"]]
            .rename(columns={"STUDENT_COUNT": "total_students_{}".format(year)})
            .reset_index(drop=True)
        )
        df_curr_enrollment = df_curr_enrollment[
            ~df_curr_enrollment["total_students_{}".format(year)].isna()
        ].reset_index(drop=True)
        df_curr_enrollment["total_students_{}".format(year)] = df_curr_enrollment[
            "total_students_{}".format(year)
        ].astype(int)

        df_curr_elem = pd.merge(
            df_curr_elem, df_curr_enrollment, how="inner", on="NCESSCH"
        )
        if df.empty:
            df = df_curr_elem.copy(deep=True)
            continue

        df = pd.merge(df, df_curr_elem, how="left", on="NCESSCH")

    df = df.reset_index(drop=True)

    # Identify max enrollment over the time horizon and set that as the school capacity
    total_keys = []
    for year in years:
        total_keys.append("total_students_{}".format(year))
    student_cap = {"NCESSCH": [], "student_capacity": []}
    for i in range(0, len(df)):
        student_cap["NCESSCH"].append(df["NCESSCH"][i])
        student_cap["student_capacity"].append(np.max(df.loc[i, total_keys]))

    df_cap = pd.DataFrame(student_cap)
    df = pd.merge(df, df_cap, on="NCESSCH", how="inner")
    df.to_csv(output_file, index=False)


def copy_blocks_to_elementary(
    input_dir="/Users/ngillani/OneDrive - Northeastern University/neu/rezoning-schools/data/derived_data/2122/",
    output_dir="data/attendance_boundaries/2122/",
):
    for state in os.listdir(input_dir):
        print(state)
        Path(os.path.join(output_dir, state)).mkdir(parents=True, exist_ok=True)
        shutil.copy(
            os.path.join(input_dir, state, "blocks_to_elementary.csv"),
            os.path.join(output_dir, state, "blocks_to_elementary.csv"),
        )


def output_school_enrollments_by_race_grade(
    input_dir="data/attendance_boundaries/2122/",
    capacities_file="data/school_data/21_22_school_capacities.csv",
    output_file="data/school_data/21_22_school_counts_by_grade_and_race.csv",
):
    # pd.set_option("display.max_columns", None)
    df_caps = pd.read_csv(capacities_file, dtype={"NCESSCH": str})[
        ["NCESSCH", "total_students_2021-2022"]
    ]
    df_enrollment = pd.read_csv(
        os.path.join("data/school_data/ccd/2021-2022/", CCD_FILES["2021-2022"][1]),
        dtype={"NCESSCH": str},
    )
    df_enrollment_cat = df_enrollment[
        (
            df_enrollment["TOTAL_INDICATOR"]
            == "Category Set A - By Race/Ethnicity; Sex; Grade"
        )
        & (df_enrollment["NCESSCH"].isin(df_caps["NCESSCH"]))
    ].reset_index(drop=True)

    df_enrollment_g = (
        df_enrollment_cat.groupby(
            ["NCESSCH", "GRADE", "RACE_ETHNICITY"], as_index=False
        )
        .agg({"STUDENT_COUNT": "sum"})
        .reset_index(drop=True)
    )
    df_enrollment_g["GRADE"] = df_enrollment_g["GRADE"].replace(
        list(GRADE_KEYS.keys()), list(GRADE_KEYS.values())
    )
    df_enrollment_g_pivot = df_enrollment_g.pivot_table(
        index=["NCESSCH", "GRADE"],
        columns=["RACE_ETHNICITY"],
        values="STUDENT_COUNT",
    ).reset_index()
    individual_race_keys = list(RACE_KEYS.keys())
    individual_race_keys.remove("Total")
    individual_race_keys.remove("Two or more races")
    df_enrollment_g_pivot["Total"] = (
        df_enrollment_g_pivot[individual_race_keys].sum(axis=1)
        - df_enrollment_g_pivot["Two or more races"]
    )

    df_enrollment_g_pivot = df_enrollment_g_pivot.rename(columns=RACE_KEYS)
    df_enrollment_g_pivot = df_enrollment_g_pivot.pivot_table(
        index="NCESSCH", columns="GRADE"
    ).reset_index()
    df_enrollment_g_pivot.columns = df_enrollment_g_pivot.columns.to_series().str.join(
        "_"
    )

    df_enrollment_g_pivot = df_enrollment_g_pivot.rename(
        columns={"NCESSCH_": "NCESSCH"}
    )
    df_enrollment_g_pivot = df_enrollment_g_pivot.fillna(0)

    # Set the num per grade/race to always be capped by the num total per grade
    import warnings

    warnings.filterwarnings("ignore")
    grades = set(GRADE_KEYS.values()) - set(["NS"])
    for r in RACE_KEYS.values():
        for g in grades:
            for i in range(0, len(df_enrollment_g_pivot)):
                print(r, g, i / len(df_enrollment_g_pivot))
                df_enrollment_g_pivot[f"{r}_{g}"][i] = min(
                    df_enrollment_g_pivot[f"{r}_{g}"][i],
                    df_enrollment_g_pivot[f"num_total_{g}"][i],
                )

    df_enrollment_g_pivot.to_csv(output_file, index=False)


def output_neighboring_districts(
    district_shapefile="data/school_district_2021_boundaries/shapes/schooldistrict_sy2021_tl21.shp",
    output_file="data/school_district_2021_boundaries/district_neighbors.json",
    output_file_district_centroids="data/school_district_2021_boundaries/district_centroids.json",
):
    print("Loading district shapes ...")
    df_shapes = gpd.read_file(district_shapefile)
    df_shapes["district_centroid_lat"] = df_shapes.centroid.y
    df_shapes["district_centroid_long"] = df_shapes.centroid.x
    district_centroids = {
        df_shapes["GEOID"][i]: [
            df_shapes["district_centroid_lat"][i],
            df_shapes["district_centroid_long"][i],
        ]
        for i in range(0, len(df_shapes))
    }
    header.write_dict(output_file_district_centroids, district_centroids)


"""
    There were some missing districts from the above process, so trying again with a direct download from the dept of education
"""


def output_updated_district_centroids(
    input_file="data/school_district_2021_boundaries/EDGE_GEOCODE_PUBLICLEA_2122.csv",
    curr_centroids_file="data/school_district_2021_boundaries/district_centroids.json",
    output_file="data/school_district_2021_boundaries/updated_district_centroids.json",
):
    df = pd.read_csv(input_file, dtype={"LEAID": str})
    df["LEAID"] = df["LEAID"].str.rjust(7, "0")
    curr_centroids = header.read_json(curr_centroids_file)
    district_centroids = {}
    for i in range(0, len(df)):
        if not df["LEAID"][i] in curr_centroids:
            district_centroids[df["LEAID"][i]] = [df["LAT"][i], df["LON"][i]]
        else:
            district_centroids[df["LEAID"][i]] = curr_centroids[df["LEAID"][i]]
    header.write_dict(output_file, district_centroids)


def output_allowed_mergers(
    input_dir="data/attendance_boundaries/2122/",
    schools_file="data/school_data/21_22_school_counts_by_grade_and_race.csv",
    census_block_shapefiles="/Users/ngillani/OneDrive - Northeastern University/neu/rezoning-schools/data/census_block_shapefiles_2020/tl_2021_{}_tabblock20/tl_2021_{}_tabblock20.shp",
    neighboring_districts_file="data/school_district_2021_boundaries/district_neighbors.json",
    output_dir="data/solver_files/2122/{}/",
    output_file_within="within_district_allowed_mergers.json",
    output_file_between="between_within_district_allowed_mergers.json",
):
    def update_between_dist_dict(
        allowable_between_within_district, allowable_within_district
    ):
        for d in allowable_within_district:
            allowable_between_within_district[d].update(allowable_within_district[d])

    def compute_bordering_schools(
        df_state, allowed_districts, allowable_school_mergers
    ):
        df_curr_d = df_state[df_state["leaid"].isin(allowed_districts)].reset_index(
            drop=True
        )
        school_ids = set(df_curr_d["ncessch"].tolist())
        for i, s in enumerate(school_ids):
            df_curr_d_s = df_curr_d[df_curr_d["ncessch"] == s].reset_index(drop=True)
            bordering_schools = df_curr_d[
                ~df_curr_d["geometry"].disjoint(df_curr_d_s["geometry"].unary_union)
            ]

            allowable_school_mergers[s].update(
                set(bordering_schools["ncessch"].tolist())
            )

    district_neighbors = defaultdict(list)
    curr = header.read_json(neighboring_districts_file)
    df_schools = pd.read_csv(schools_file, dtype={"NCESSCH": str})
    for d in curr:
        district_neighbors[d] = curr[d]

    all_states = os.listdir(input_dir)
    states_already_processed = []
    for state in all_states:
        if os.path.exists(os.path.join(output_dir.format(state), output_file_within)):
            states_already_processed.append(state)
    remaining_states = set(all_states) - set(states_already_processed)
    for state in remaining_states:
        allowable_within_district = defaultdict(set)
        allowable_between_within_district = defaultdict(set)
        state_fips = us.states.lookup(state).fips

        df_blocks = gpd.read_file(
            census_block_shapefiles.format(state_fips, state_fips),
            dtype={"GEOID20": str},
        )[["GEOID20", "geometry"]]
        df_state = pd.read_csv(
            os.path.join(input_dir, state, "blocks_to_elementary.csv"),
            dtype={"ncessch": str, "leaid": str, "GEOID20": str},
        ).drop(columns=["geometry"])
        df_state = df_state[
            df_state["ncessch"].isin(df_schools["NCESSCH"])
        ].reset_index(drop=True)
        df_state = df_blocks.merge(df_state, on="GEOID20", how="inner")
        df_state = df_state.set_geometry("geometry")
        df_state = gpd.GeoDataFrame(df_state)
        df_state = df_state[
            (df_state["openEnroll"] == "N")
            & (df_state["ncessch"].str.startswith(state_fips))
        ].reset_index(drop=True)
        district_ids = set(df_state["leaid"].tolist())
        for i, d in enumerate(district_ids):
            print(state, i / len(district_ids))
            compute_bordering_schools(df_state, [d], allowable_within_district)
            update_between_dist_dict(
                allowable_between_within_district, allowable_within_district
            )
            compute_bordering_schools(
                df_state, district_neighbors[d], allowable_between_within_district
            )

        for s in allowable_within_district:
            allowable_within_district[s] = list(allowable_within_district[s])
        for s in allowable_between_within_district:
            allowable_between_within_district[s] = list(
                allowable_between_within_district[s]
            )
        curr_path = os.path.join(output_dir.format(state))
        Path(curr_path).mkdir(parents=True, exist_ok=True)
        header.write_dict(
            os.path.join(curr_path, output_file_within), allowable_within_district
        )
        header.write_dict(
            os.path.join(curr_path, output_file_between),
            allowable_between_within_district,
        )


def output_districts_all_closed_enrollment_elementary(
    input_dir="data/attendance_boundaries/2122/",
    output_file="data/school_data/entirely_elem_closed_enrollment_districts.csv",
):

    all_states = os.listdir(input_dir)
    all_data = {"district_id": []}
    for state in all_states:
        df_state = pd.read_csv(
            os.path.join(input_dir, state, "blocks_to_elementary.csv"),
            dtype={"ncessch": str, "leaid": str, "GEOID20": str},
        ).drop(columns=["geometry"])
        all_dists = list(set(df_state["leaid"].tolist()))
        for i, d in enumerate(all_dists):
            print(state, i / len(all_dists))
            df_dist = df_state[df_state["leaid"] == d].reset_index(drop=True)
            df_dist_open_enrollment = df_dist[df_dist["openEnroll"] == "Y"].reset_index(
                drop=True
            )
            if len(df_dist_open_enrollment) == 0:
                all_data["district_id"].append(d)

    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False)


def output_allowed_schools(
    schools_file="data/school_data/21_22_school_counts_by_grade_and_race.csv",
    input_dir="data/attendance_boundaries/2122/",
    output_dir="data/solver_files/2122/{}/",
):
    df_schools = pd.read_csv(schools_file, dtype={"NCESSCH": str})
    for state in os.listdir(input_dir):
        curr_state = us.states.lookup(state)
        df_blocks = pd.read_csv(
            os.path.join(input_dir, state, "blocks_to_elementary.csv"),
            dtype={"leaid": str, "ncessch": str},
        )
        df_blocks = df_blocks[df_blocks["openEnroll"] == "N"].reset_index(drop=True)
        nces_ids = list(set(df_blocks["ncessch"].tolist()))
        print(state, curr_state)
        state_fips = curr_state.fips
        df_curr = df_schools[
            (df_schools["NCESSCH"].str.startswith(state_fips))
            & (df_schools["NCESSCH"].isin(nces_ids))
        ].reset_index(drop=True)
        curr_dir = output_dir.format(state)
        Path(curr_dir).mkdir(parents=True, exist_ok=True)
        df_curr.to_csv(os.path.join(curr_dir, "school_enrollments.csv"), index=False)


def output_districts_to_process(
    input_file="data/school_data/21_22_school_counts_by_grade_and_race.csv",
    membership_file="ccd/2021-2022/ccd_sch_029_2122_w_1a_071722.csv",
    output_file_districts="data/school_data/all_districts.csv",
    output_file_schools="data/school_data/all_schools_with_names.csv",
):
    df = pd.read_csv(input_file, dtype={"NCESSCH": str})
    df["district_id"] = df["NCESSCH"].str[:7]
    df_names = pd.read_csv(membership_file, dtype={"NCESSCH": str})[
        ["NCESSCH", "SCH_NAME", "LEA_NAME"]
    ]
    print(len(df))
    df = pd.merge(df, df_names, on="NCESSCH", how="inner")
    print(len(df))
    df.to_csv(output_file_schools, index=False)
    df_g = (
        df.groupby("district_id", as_index=False)
        .agg({"NCESSCH": "count"})
        .rename(columns={"NCESSCH": "num_schools"})
        .reset_index(drop=True)
    )
    all_states = []
    for i in range(0, len(df_g)):
        print(i / len(df_g))
        state = us.states.lookup(df_g["district_id"].str[:2][i])
        try:
            all_states.append(state.abbr)
        except Exception:
            all_states.append("")
    df_g["state"] = all_states
    df_g.to_csv(output_file_districts, index=False)


if __name__ == "__main__":
    # output_neighboring_districts()
    # determine_school_capacities()
    # copy_blocks_to_elementary()
    # output_school_enrollments_by_race_grade()
    # output_allowed_mergers()
    # output_allowed_schools()
    # output_districts_to_process()
    # output_districts_all_closed_enrollment_elementary()
    output_updated_district_centroids()
