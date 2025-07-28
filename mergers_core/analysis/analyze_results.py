import mergers_core.utils.header as header
import geopandas as gpd
import folium
import us
import pandas as pd
import numpy as np
import glob
import os
import random
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import sys


def quick_analysis(
    results_file="data/results/min_num_elem_schools_4_constrained/consolidated_simulation_results_min_num_elem_schools_4_constrained.csv",
    dists_file="data/school_data/all_districts.csv",
):
    df_r = pd.read_csv(results_file, dtype={"district_id": str})
    df_r = df_r[df_r["num_total_all"] > 0].reset_index(drop=True)
    df_d = pd.read_csv(dists_file, dtype={"district_id": str})
    df = pd.merge(df_r, df_d, on="district_id", how="inner")
    df_200_largest = df.sort_values(by="num_schools", ascending=False).head(200)
    print(df_200_largest["pre_dissim"].median(), df_200_largest["post_dissim"].median())


def identify_moderate_district_large_decrease_in_dissim(
    results_file="data/results/min_num_elem_schools_4_constrained/consolidated_simulation_results_min_num_elem_schools_4_constrained.csv",
    dists_file="data/school_data/all_districts.csv",
):
    df_r = pd.read_csv(results_file, dtype={"district_id": str})
    df_r = df_r[df_r["num_total_all"] > 0].reset_index(drop=True)
    df_d = pd.read_csv(dists_file, dtype={"district_id": str})
    df = pd.merge(df_r, df_d, on="district_id", how="inner")
    df["dissim_change"] = df["pre_dissim"] - df["post_dissim"]
    df = df[(df["num_schools"] <= 15) & (df["num_schools"] >= 10)].reset_index(
        drop=True
    )
    df_largest = df.sort_values(by="dissim_change", ascending=False).iloc[0]
    print(
        df_largest["district_id"],
        df_largest["num_schools"],
        df_largest["pre_dissim"],
        df_largest["post_dissim"],
    )


def viz_assignments(
    results_dir="data/results/min_num_elem_schools_4_constrained/",
    state="PA",
    district_id="4218990",
    district_centroids_file="data/school_district_2021_boundaries/district_centroids.json",
    lat_long_file="data/school_data/nces_21_22_lat_longs.csv",
    orig_boundaries="data/attendance_boundaries/2122/{}/estimated_student_counts_per_block.csv",
    school_names_file="data/all_schools_with_names.csv",
    save_file=True,
):

    df_names = pd.read_csv(school_names_file, dtype={"NCESSCH": str})[
        ["NCESSCH", "SCH_NAME"]
    ]
    df_mergers = pd.read_csv(
        glob.glob(
            os.path.join(results_dir, state, district_id, "**/" + "school_mergers.csv"),
            recursive=True,
        )[0],
        dtype={"school_cluster": str},
    )

    school_cluster_lists = df_mergers["school_cluster"].tolist()
    school_clusters = {}
    cluster_assignments = {"ncessch": [], "cluster_id": []}
    cluster_id = 0
    for c in school_cluster_lists:
        schools = c.split(", ")
        for s in schools:
            cluster_assignments["ncessch"].append(s)
            cluster_assignments["cluster_id"].append(cluster_id)
            school_clusters[s] = schools
        cluster_id += 1

    df_cluster_assgn = pd.DataFrame(data=cluster_assignments)

    df_lat_long = pd.read_csv(lat_long_file, dtype={"nces_id": str})[
        ["nces_id", "lat", "long"]
    ].rename(columns={"lat": "zoned_lat", "long": "zoned_long"})
    district_centroids = header.read_json(district_centroids_file)
    df_asgn_orig = (
        pd.read_csv(orig_boundaries.format(state), dtype={"ncessch": str})
        .drop_duplicates(subset=["block_id"])
        .rename(columns={"block_id": "GEOID20"})
    )
    df_asgn_orig = pd.merge(
        df_asgn_orig, df_names, left_on="ncessch", right_on="NCESSCH", how="left"
    )
    df_asgn_orig["percent_white"] = (
        df_asgn_orig["num_white"] / df_asgn_orig["num_total"]
    )
    df_asgn_orig["percent_white"] = df_asgn_orig["percent_white"]
    df_asgn_orig["district_id"] = df_asgn_orig["ncessch"].str[:7]

    df_asgn_orig = df_asgn_orig[df_asgn_orig["district_id"] == district_id].reset_index(
        drop=True
    )

    df_asgn_orig = pd.merge(
        df_asgn_orig,
        df_lat_long,
        left_on="ncessch",
        right_on="nces_id",
        how="inner",
    )

    blocks_shape_file = "~/OneDrive - Northeastern University/neu/rezoning-schools/data/census_block_shapefiles_2020/tl_2021_{}_tabblock20/tl_2021_{}_tabblock20.shp"
    state_fips = us.states.lookup(state).fips
    state_blocks = gpd.read_file(blocks_shape_file.format(state_fips, state_fips))
    state_blocks["GEOID20"] = state_blocks["GEOID20"].astype(int)

    df_orig = gpd.GeoDataFrame(
        pd.merge(
            df_asgn_orig,
            state_blocks,
            on="GEOID20",
            how="inner",
        )
    )

    df_orig = df_orig.to_crs(epsg=4326)

    all_schools_nces = set(df_orig["ncessch"].tolist())
    print("Num schools: ", len(all_schools_nces))

    school_markers = {}
    for nces in all_schools_nces:
        curr = df_orig[df_orig["ncessch"] == nces].iloc[0]
        school_markers[nces] = [curr["zoned_lat"], curr["zoned_long"], curr["SCH_NAME"]]

    # Generate colors
    colors = {}
    for nces in all_schools_nces:
        random.seed(int(nces))
        colors[nces] = "#" + "%06x" % random.randint(0, 0xFFFFFF)

    # Compute percentage white per school
    per_cat_per_school = defaultdict(Counter)
    for nces in all_schools_nces:
        df_curr = df_orig[df_orig["ncessch"] == nces]
        per_cat_per_school["num_white"][nces] += np.nansum(
            df_curr["num_white"].tolist()
        )
        per_cat_per_school["num_total"][nces] += np.nansum(
            df_curr["num_total"].tolist()
        )

    m_orig = folium.Map(
        location=district_centroids[district_id],
        zoom_start=12,
        tiles="CartoDB positron",
    )

    m_race = folium.Map(
        location=district_centroids[district_id],
        zoom_start=12,
        tiles="CartoDB positron",
    )

    m_merged = folium.Map(
        location=district_centroids[district_id],
        zoom_start=12,
        tiles="CartoDB positron",
    )

    def add_shape_to_map(curr_map, geo_shape, fill_color, fill_opacity, weight):
        if np.isnan(fill_opacity):
            fill_opacity = 0.2
            fill_color = "gray"

        geo_j = folium.GeoJson(
            data=geo_shape,
            style_function=lambda x, fillColor=fill_color, fillOpacity=fill_opacity, weight=weight: {
                "fillOpacity": fillOpacity,
                "fillColor": fillColor,
                "weight": weight,
            },
        )
        geo_j.add_to(curr_map)

    def add_school_markers(curr_map, school_markers):
        for m in school_markers:
            folium.Marker(
                location=[school_markers[m][0], school_markers[m][1]],
                icon=folium.Icon(color="blue", icon_color="white", icon="info-sign"),
                popup=school_markers[m][2],
            ).add_to(curr_map)

    print(len(df_orig))
    df_orig_mega = df_orig.dissolve(by="ncessch", as_index=False)
    print(len(df_orig_mega))
    for i, r in df_orig_mega.iterrows():

        # Adding to orig map
        sim_geo = gpd.GeoSeries(r["geometry"])
        geo_j = sim_geo.to_json()
        add_shape_to_map(m_orig, geo_j, colors[df_orig_mega["ncessch"][i]], 0.5, ".5")

    add_school_markers(m_orig, school_markers)

    for i, r in df_orig_mega.iterrows():
        # Adding to race map
        sim_geo = gpd.GeoSeries(r["geometry"])
        geo_j = sim_geo.to_json()
        add_shape_to_map(
            m_race,
            geo_j,
            "blue",
            1
            - (
                per_cat_per_school["num_white"][df_orig_mega["ncessch"][i]]
                / per_cat_per_school["num_total"][df_orig_mega["ncessch"][i]]
            ),
            ".5",
        )

    add_school_markers(m_race, school_markers)

    df_merged = gpd.GeoDataFrame(pd.merge(df_orig, df_cluster_assgn, on="ncessch"))
    df_merged_mega = df_merged.dissolve(by="cluster_id", as_index=False)
    print("Mega num rows: ", len(df_merged_mega))
    print("Mega fields: ", df_merged_mega.keys())
    for i, r in df_merged_mega.iterrows():
        # Adding to mergers map
        sim_geo = gpd.GeoSeries(r["geometry"])
        geo_j = sim_geo.to_json()
        add_shape_to_map(
            m_merged,
            geo_j,
            colors[school_clusters[df_merged_mega["ncessch"][i]][0]],
            0.5,
            ".5",
        )
    add_school_markers(m_merged, school_markers)

    m_orig.save(os.path.join(results_dir, state, district_id, "original.html"))
    m_race.save(os.path.join(results_dir, state, district_id, "race.html"))
    m_merged.save(os.path.join(results_dir, state, district_id, "mergers.html"))

    # if save_file:
    #     plt.axis("off")
    #     plt.savefig(output_file, bbox_inches="tight")
    #     plt.close()
    # else:
    #     plt.show()


def compare_to_redistricting(
    redist_results="data/misc/all_usa_elem_within_district_sim_possible_changes.csv",
    mergers_results="data/misc/results_top_200_by_population.csv",
    output_file="data/misc/redist_mergers_comparison.csv",
):
    df_redist = pd.read_csv(redist_results, dtype={"district_id": str})
    df_redist = df_redist[df_redist["is_contiguous"] == True].reset_index(drop=True)

    # TODO(ng): this is a hack for figuring out approx percent rezoned since
    # it doesn't account for overlaps in certain group populations ...
    # so ... fix it
    df_redist["approx_percent_rezoned"] = (
        df_redist["white_percent_rezoned"]
        * df_redist["district_perwht"]
        * df_redist["num_elem_students"]
        + df_redist["black_percent_rezoned"]
        * df_redist["district_perblk"]
        * df_redist["num_elem_students"]
        + df_redist["hisp_percent_rezoned"]
        * df_redist["district_perhsp"]
        * df_redist["num_elem_students"]
        + df_redist["asian_percent_rezoned"]
        * df_redist["district_perasn"]
        * df_redist["num_elem_students"]
        + df_redist["native_percent_rezoned"]
        * df_redist["district_pernam"]
        * df_redist["num_elem_students"]
    ) / df_redist["num_elem_students"]
    df_redist = df_redist[
        ["district_id", "white_non_white_seg_change_prop", "approx_percent_rezoned"]
    ]

    df_mergers = pd.read_csv(mergers_results, dtype={"district_id": str})
    df_mergers["dissim_change"] = (
        df_mergers["post_dissim"] - df_mergers["pre_dissim"]
    ) / df_mergers["pre_dissim"]
    df_mergers = df_mergers[["district_id", "dissim_change", "travel_times_change"]]

    df = pd.merge(df_redist, df_mergers, how="inner", on="district_id")
    df = df.dropna().reset_index(drop=True)

    # Get lines of best fit for each policy regime
    import statsmodels.api as sm

    result_redist = sm.OLS(
        df["white_non_white_seg_change_prop"],
        df["approx_percent_rezoned"],
    ).fit()
    redist_slope = list(result_redist.params)[0]

    result_mergers = sm.OLS(
        df["dissim_change"],
        df["travel_times_change"],
    ).fit()
    mergers_slope = list(result_mergers.params)[0]

    df["redist_below_line"] = [
        df["white_non_white_seg_change_prop"][i]
        < redist_slope * df["approx_percent_rezoned"][i]
        for i in range(0, len(df))
    ]

    df["mergers_below_line"] = [
        df["dissim_change"][i] < mergers_slope * df["travel_times_change"][i]
        for i in range(0, len(df))
    ]

    df["same_place_wrt_line"] = df["redist_below_line"] == df["mergers_below_line"]
    print(np.sum(df["same_place_wrt_line"]))

    df.to_csv(output_file)


def plot_dissimilarity_vs_population_consistency(run_paths):
    all_data = []
    for run_path in run_paths:
        analytics_file = os.path.join(run_path, "analytics.csv")
        if os.path.exists(analytics_file):
            df = pd.read_csv(analytics_file)
            all_data.append(df)
        else:
            print(f"Warning: {analytics_file} not found. Skipping.")

    if not all_data:
        print("No data found to plot.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    combined_df["pre_dissim_col"] = combined_df.apply(
        lambda row: row[f"pre_dissim_{row['dissimilarity_flavor']}"], axis=1
    )
    combined_df["post_dissim_col"] = combined_df.apply(
        lambda row: row[f"post_dissim_{row['dissimilarity_flavor']}"], axis=1
    )

    for i in range(len(combined_df)):
        print(
            f"{combined_df['pre_population_consistency'][i]}, {combined_df['pre_dissim_col'][i]} -> {combined_df['post_population_consistency'][i]}, {combined_df['post_dissim_col'][i]}"
        )
        plt.plot(
            [
                combined_df["pre_population_consistency"][i],
                combined_df["pre_dissim_col"][i],
            ],
            [
                combined_df["post_population_consistency"][i],
                combined_df["post_dissim_col"][i],
            ],
            color="black",
            label="Trend",
            marker="o",
        )

    plt.scatter(
        combined_df["post_population_consistency"],
        combined_df["post_dissim_col"],
        color="red",
        label="a",
    )

    plt.xlabel("Population Consistency")
    plt.ylabel("Dissimilarity")
    plt.title("Dissimilarity vs. Population Consistency")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python analyze_results.py <dissimilarity_flavor> <run_path_1> [run_path_2 ...]"
        )
        print("Example: python analyze_results.py /path/to/run1 /path/to/run2")
        sys.exit(1)

    run_paths = sys.argv[1:]
    plot_dissimilarity_vs_population_consistency(run_paths)
