import mergers_core.utils.header as header
import mergers_core.models.constants as constants
import geopandas as gpd
import pandas as pd
import numpy as np
import glob
import os
from collections import Counter


def compute_dissimilarity_index(df_dist):
    vals = []
    cat_total = df_dist["num_white"].sum()
    non_cat_total = df_dist["num_total"].sum() - df_dist["num_white"].sum()
    for i in range(0, len(df_dist)):
        vals.append(
            np.abs(
                (df_dist["num_white"][i] / cat_total)
                - ((df_dist["num_total"][i] - df_dist["num_white"][i]) / non_cat_total)
            )
        )

    return 0.5 * np.nansum(vals)


def get_school_demos(df):
    df["district_id"] = df["NCESSCH"].str[:7]
    schools_data = {
        "district_id": [],
        "NCESSCH": [],
        "num_total": [],
        "num_white": [],
        "num_black": [],
        "num_asian": [],
        "num_native": [],
        "num_hispanic": [],
        "num_pacific_islander": [],
    }

    cols_black = ["num_black_{}".format(g) for g in constants.GRADE_TO_IND]
    cols_asian = ["num_asian_{}".format(g) for g in constants.GRADE_TO_IND]
    cols_native = ["num_native_{}".format(g) for g in constants.GRADE_TO_IND]
    cols_hisp = ["num_hispanic_{}".format(g) for g in constants.GRADE_TO_IND]
    cols_pacific_islander = [
        "num_pacific_islander_{}".format(g) for g in constants.GRADE_TO_IND
    ]
    cols_white = ["num_white_{}".format(g) for g in constants.GRADE_TO_IND]
    cols_total = ["num_total_{}".format(g) for g in constants.GRADE_TO_IND]
    for i in range(0, len(df)):
        schools_data["district_id"].append(df["district_id"][i])
        schools_data["NCESSCH"].append(df["NCESSCH"][i])
        schools_data["num_total"].append(df.loc[i, cols_total].sum())
        schools_data["num_white"].append(df.loc[i, cols_white].sum())
        schools_data["num_black"].append(df.loc[i, cols_black].sum())
        schools_data["num_asian"].append(df.loc[i, cols_asian].sum())
        schools_data["num_native"].append(df.loc[i, cols_native].sum())
        schools_data["num_hispanic"].append(df.loc[i, cols_hisp].sum())
        schools_data["num_pacific_islander"].append(
            df.loc[i, cols_pacific_islander].sum()
        )

    df_schools = pd.DataFrame(schools_data)
    return df_schools


def produce_dists_data_file(
    input_file="data/school_data/21_22_school_counts_by_grade_and_race.csv",
    closed_enroll_file="data/school_data/entirely_elem_closed_enrollment_districts.csv",
    output_file="data/school_data/open_v_closed_enrollment_districts.csv",
):
    df = pd.read_csv(input_file, dtype={"NCESSCH": str})

    df_schools = get_school_demos(df)
    df_schools_g = (
        df_schools.groupby(["district_id"], as_index=False)
        .agg(
            {
                "district_id": "first",
                "NCESSCH": "count",
                "num_total": "sum",
                "num_white": "sum",
                "num_black": "sum",
                "num_asian": "sum",
                "num_native": "sum",
                "num_hispanic": "sum",
                "num_pacific_islander": "sum",
            }
        )
        .rename(
            columns={
                "num_total": "dist_num_total",
                "num_white": "dist_num_white",
                "num_black": "dist_num_black",
                "num_asian": "dist_num_asian",
                "num_native": "dist_num_native",
                "num_hispanic": "dist_num_hispanic",
                "num_pacific_islander": "dist_num_pacific_islander",
                "NCESSCH": "num_elem_schools",
            }
        )
    )

    df = pd.merge(df, df_schools_g, how="inner", on="district_id").sort_values(
        by="dist_num_total", ascending=False
    )

    dissim_indices = {
        "district_id": [],
        "white_nonwhite_dissim": [],
    }

    district_ids = set(df["district_id"].tolist())
    for i, dist in enumerate(district_ids):
        print(i / len(district_ids))
        df_curr = df_schools[df_schools["district_id"] == dist].reset_index(drop=True)
        try:
            curr = compute_dissimilarity_index(df_curr)
        except Exception as e:
            curr = float("nan")
        dissim_indices["district_id"].append(dist)
        dissim_indices["white_nonwhite_dissim"].append(curr)

    df_dissim = pd.DataFrame(data=dissim_indices)
    df_dists = pd.merge(
        df_schools_g, df_dissim, how="inner", on="district_id"
    ).reset_index(drop=True)

    df_enroll = pd.read_csv(closed_enroll_file, dtype={"district_id": str})
    df_dists["all_elem_closed_enrollment"] = [
        df_dists["district_id"].iloc[i] in df_enroll["district_id"].tolist()
        for i in range(0, len(df_dists))
    ]

    df_dists.to_csv(output_file, index=False)


def analyze_districts_in_sample(
    all_dists_file="data/school_data/open_v_closed_enrollment_districts.csv",
    results_file="data/school_data/results_top_200_by_population.csv",
):
    df_all = pd.read_csv(all_dists_file, dtype={"district_id": str})
    df_all["prop_white"] = df_all["dist_num_white"] / df_all["dist_num_total"]
    df_all = df_all[df_all["num_elem_schools"] > 1].reset_index(drop=True)
    df_all_closed_enroll = df_all[
        df_all["all_elem_closed_enrollment"] == True
    ].reset_index(drop=True)

    df_non_closed_enrollment = df_all[
        df_all["all_elem_closed_enrollment"] == False
    ].reset_index(drop=True)
    df_res = pd.read_csv(results_file, dtype={"district_id": str})

    df_all["in_sample"] = [
        int(df_all["district_id"][i] in df_res["district_id"].tolist())
        for i in range(0, len(df_all))
    ]

    df_in_sample = df_all[df_all["in_sample"] == 1].reset_index(drop=True)

    print("Num all: ", len(df_all))
    print("Num all and closed enroll: ", len(df_all_closed_enroll))
    print(
        np.median(df_in_sample["white_nonwhite_dissim"]),
        np.median(df_all_closed_enroll["white_nonwhite_dissim"]),
        np.median(df_non_closed_enrollment["white_nonwhite_dissim"]),
    )

    print(
        np.median(df_in_sample["prop_white"]),
        np.median(df_all_closed_enroll["prop_white"]),
        np.median(df_non_closed_enrollment["prop_white"]),
    )
    import statsmodels.formula.api as smf

    mod = smf.ols(
        formula="in_sample ~ dist_num_total + dist_num_white + white_nonwhite_dissim",
        data=df_all,
    )
    res = mod.fit()
    print(res.summary())

    mod = smf.ols(
        formula="in_sample ~ dist_num_total + dist_num_white + white_nonwhite_dissim",
        data=df_all_closed_enroll,
    )
    res = mod.fit()
    print(res.summary())


def analyze_choice_options_in_districts(
    charter_schools_file="data/school_data/21_22_charter_status.csv",
    magnet_schools_file="data/school_data/21_22_magnet_status.csv",
    district_boundaries_file="data/school_district_2021_boundaries/shapes/schooldistrict_sy2021_tl21.shp",
    schools_locations_file="data/school_data/nces_21_22_lat_longs.csv",
    school_demos_file="data/school_data/21_22_school_counts_by_grade_and_race.csv",
    dist_demographics_file="data/school_data/open_v_closed_enrollment_districts.csv",
    top_dists_file="data/school_data/results_top_200_by_population.csv",
    output_file="data/school_data/top_200_school_choice_analysis.csv",
):
    # First, identify lat / long of all charter and magnet elem schools (since we only include non open erollment schools, don't expect magnets?)
    # Next, identify which ones are contained in which of our 98 school districts
    # Next, determine ratio of students at these schools to the in-dist elem options (to determine approx. rates of choice)
    # Next, determine how their racial breakdown compares to those of the district as a whole (in terms of white/non-white)
    # Use this to inform choice scenarios — what if the ratio grew by X%?  Y%?  **need to do this by school, different schools prob have diff opt out rates

    df_geos = pd.read_csv(
        schools_locations_file,
        encoding="utf-8",
        dtype={"nces_id": str},
    )[["nces_id", "lat", "long"]]
    df_public_universe = pd.read_csv(
        charter_schools_file,
        encoding="ISO-8859-1",
        dtype={"NCESSCH": str, "LEAID": str},
    )
    df_public_universe_elem = df_public_universe[
        df_public_universe["LEVEL"] == "Elementary"
    ].reset_index(drop=True)
    df_charters = (
        df_public_universe_elem[df_public_universe_elem["CHARTER_TEXT"] == "Yes"]
        .reset_index(drop=True)[["LEAID", "NCESSCH", "CHARTER_TEXT"]]
        .rename(columns={"LEAID": "leaid_charter"})
    )
    df_magnets = pd.read_csv(
        magnet_schools_file,
        encoding="ISO-8859-1",
        dtype={"NCESSCH": str, "LEAID": str},
    )
    df_magnets = (
        df_magnets[
            (df_magnets["MAGNET_TEXT"] == "Yes")
            & (df_magnets["NCESSCH"].isin(df_public_universe_elem["NCESSCH"]))
        ]
        .reset_index(drop=True)[["LEAID", "NCESSCH", "MAGNET_TEXT"]]
        .rename(columns={"LEAID": "leaid_magnet"})
    )
    df_choice_schools = pd.merge(df_charters, df_magnets, on="NCESSCH", how="outer")

    # Merge lat long
    df_choice_schools = pd.merge(
        df_choice_schools, df_geos, left_on="NCESSCH", right_on="nces_id", how="inner"
    )
    # NExt, let's get identify which district boundaries these choice schools are in
    df_dists = gpd.read_file(district_boundaries_file, dtype={"GEOID": str})
    df_choice_schools = gpd.GeoDataFrame(df_choice_schools)
    df_choice_schools["geometry"] = gpd.points_from_xy(
        df_choice_schools.long, df_choice_schools.lat
    )
    df_choice_schools = df_choice_schools.set_geometry("geometry")
    choice_schools_in_dists = (
        gpd.sjoin(df_choice_schools, df_dists, how="inner", op="intersects")
        .reset_index(drop=True)
        .drop_duplicates(subset=["NCESSCH"])
    )

    # Now, let's compute demos for these schools
    df_school_demos = get_school_demos(
        pd.read_csv(school_demos_file, dtype={"NCESSCH": str})
    )
    choice_schools_by_dist = pd.merge(
        choice_schools_in_dists, df_school_demos, on="NCESSCH", how="inner"
    )

    choice_schools_by_dist = (
        choice_schools_by_dist.groupby(["GEOID"], as_index=False)
        .agg(
            {
                "NCESSCH": "count",
                "num_total": "sum",
                "num_white": "sum",
                "num_native": "sum",
                "num_asian": "sum",
                "num_hispanic": "sum",
                "num_black": "sum",
                "num_pacific_islander": "sum",
            }
        )
        .rename(
            columns={
                "NCESSCH": "num_charter_or_magnet_schools",
                "num_total": "c_m_num_total",
                "num_white": "c_m_num_white",
                "num_native": "c_m_num_native",
                "num_asian": "c_m_num_asian",
                "num_hispanic": "c_m_num_hispanic",
                "num_black": "c_m_num_black",
                "num_pacific_islander": "c_m_num_pacific_islander",
            }
        )
    )

    df_dists_demos = pd.read_csv(dist_demographics_file, dtype={"district_id": str})
    df_merged = pd.merge(
        choice_schools_by_dist,
        df_dists_demos,
        left_on="GEOID",
        right_on="district_id",
        how="inner",
    )
    print(len(df_merged))

    df_merged["ratio_c_or_m_to_dist_enroll"] = (
        df_merged["c_m_num_total"] / df_merged["dist_num_total"]
    )

    df_merged["ratio_c_or_m_to_dist_white"] = (
        df_merged["c_m_num_white"] / df_merged["dist_num_white"]
    )

    df_merged["ratio_c_or_m_to_dist_non_white"] = (
        df_merged["c_m_num_total"] - df_merged["c_m_num_white"]
    ) / (df_merged["dist_num_total"] - df_merged["dist_num_white"])

    df_merged["ratio_c_or_m_to_dist_black"] = (
        df_merged["c_m_num_black"] / df_merged["dist_num_black"]
    )

    df_merged["ratio_c_or_m_to_dist_native"] = (
        df_merged["c_m_num_native"] / df_merged["dist_num_native"]
    )

    df_merged["ratio_c_or_m_to_dist_hispanic"] = (
        df_merged["c_m_num_hispanic"] / df_merged["dist_num_hispanic"]
    )

    df_merged["ratio_c_or_m_to_dist_asian"] = (
        df_merged["c_m_num_asian"] / df_merged["dist_num_asian"]
    )

    df_merged["ratio_c_or_m_to_dist_pacific_islander"] = (
        df_merged["c_m_num_pacific_islander"] / df_merged["dist_num_pacific_islander"]
    )

    df_top_200 = pd.read_csv(top_dists_file, dtype={"district_id": str})
    df_merged = df_merged[
        df_merged["district_id"].isin(df_top_200["district_id"].tolist())
    ]
    print(
        df_merged["ratio_c_or_m_to_dist_enroll"].median(),
        df_merged["ratio_c_or_m_to_dist_enroll"].mean(),
        df_merged["ratio_c_or_m_to_dist_enroll"].max(),
    )
    print(
        df_merged["ratio_c_or_m_to_dist_white"].median(),
        df_merged["ratio_c_or_m_to_dist_white"].mean(),
        df_merged["ratio_c_or_m_to_dist_white"].max(),
    )

    print(
        df_merged["ratio_c_or_m_to_dist_non_white"].median(),
        df_merged["ratio_c_or_m_to_dist_non_white"].mean(),
        df_merged["ratio_c_or_m_to_dist_non_white"].max(),
    )
    df_merged.to_csv(output_file, index=False)


def compute_dissim(schools, school_enrollments):

    # Now, go through and compute dissim values
    dissim_vals = []
    for s in schools:
        dissim_vals.append(
            np.abs(
                (
                    school_enrollments["num_white"][s]
                    / sum(school_enrollments["num_white"].values())
                )
                - (
                    (school_enrollments["num_non_white"][s])
                    / (sum(school_enrollments["num_non_white"].values()))
                )
            )
        )

    return 0.5 * np.sum(dissim_vals)


def estimate_dissim_with_optouts(
    choice_file="data/school_data/top_200_school_choice_analysis.csv",
    post_merger_enrollments_file="data/results/{}/**/{}/**/students_per_group_per_school_post_merger.json",
    mergers_file="data/results/{}/**/{}/**/school_mergers.csv",
    consolidated_results_file="data/results/{}/consolidated_simulation_results_{}_0.2_False.csv",
    batch="min_num_elem_schools_4_constrained",
    output_file="data/school_data/choice_dissim_results_top_200.csv",
):
    df_choice = pd.read_csv(choice_file, dtype={"district_id": str})
    df_results = pd.read_csv(
        consolidated_results_file.format(batch, batch), dtype={"district_id": str}
    )
    choice_dissim_results = {
        "district_id": [],
        "pre_dissim": [],
        "post_dissim": [],
        "post_dissim_choice": [],
    }
    for i in range(0, len(df_choice)):
        # if i == 10:
        #     break
        district_id = df_choice["district_id"][i]
        print(district_id)
        school_enrollments = header.read_json(
            glob.glob(
                os.path.join(post_merger_enrollments_file.format(batch, district_id)),
                recursive=True,
            )[0]
        )
        curr_choice = df_choice[df_choice["district_id"] == district_id].iloc[0]
        num_non_white_per_school = Counter()
        schools = list(school_enrollments["num_total"].keys())
        df_mergers = pd.read_csv(
            glob.glob(
                os.path.join(mergers_file.format(batch, district_id)),
                recursive=True,
            )[0],
            dtype={"school_cluster": str},
        )

        schools_involved_in_merger = []
        for j in range(0, len(df_mergers)):
            cluster = df_mergers["school_cluster"][j].split(", ")
            if len(cluster) > 1:
                schools_involved_in_merger.extend(cluster)

        for s in schools:
            num_non_white_per_school[s] = (
                school_enrollments["num_total"][s] - school_enrollments["num_white"][s]
            )

            # Apply discounting due to choice
            prop_white_to_opt_out = 0
            prop_non_white_to_opt_out = 0
            if s in schools_involved_in_merger:
                prop_white_to_opt_out = np.minimum(
                    curr_choice["ratio_c_or_m_to_dist_white"], 1
                )
                prop_non_white_to_opt_out = np.minimum(
                    curr_choice["ratio_c_or_m_to_dist_non_white"], 1
                )
            school_enrollments["num_white"][s] *= 1 - prop_white_to_opt_out
            num_non_white_per_school[s] *= 1 - prop_non_white_to_opt_out

        discounted_enrollments = {
            "num_white": school_enrollments["num_white"],
            "num_non_white": num_non_white_per_school,
        }
        discounted_dissim = compute_dissim(schools, discounted_enrollments)
        curr_res = df_results[df_results["district_id"] == district_id].iloc[0]
        choice_dissim_results["district_id"].append(district_id)
        choice_dissim_results["pre_dissim"].append(curr_res["pre_dissim"])
        choice_dissim_results["post_dissim"].append(curr_res["post_dissim"])
        choice_dissim_results["post_dissim_choice"].append(discounted_dissim)

    df_choice_results = pd.DataFrame(choice_dissim_results)
    print(df_choice_results.head(10))
    print(
        np.median(df_choice_results["pre_dissim"]),
        np.median(df_choice_results["post_dissim"]),
        np.median(df_choice_results["post_dissim_choice"]),
    )


if __name__ == "__main__":
    # produce_dists_data_file()
    # analyze_districts_in_sample()
    # analyze_choice_options_in_districts()
    estimate_dissim_with_optouts()
