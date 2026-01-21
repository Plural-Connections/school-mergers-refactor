import geopandas as gpd
import folium
import us
import pandas as pd
import numpy as np
import glob
import os
import random
import matplotlib.pyplot as plt
import json
import sys
from models.config import District


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
    district: District,
    save_file=False,
):
    results_dir = f"data/results/{district.state}/{district.id}/"
    df_names = pd.read_csv("data/all_schools_with_names.csv", dtype={"NCESSCH": str})[
        ["NCESSCH", "SCH_NAME"]
    ]
    df_mergers = pd.read_csv(
        glob.glob(
            os.path.join(results_dir, "**/" + "school_mergers.csv"),
            recursive=True,
        )[0],
        dtype={"school_cluster": str},
    )

    df_schools_in_play = pd.read_csv(
        glob.glob(
            os.path.join(results_dir, "**/" + "schools_in_play.csv"),
            recursive=True,
        )[0],
        dtype={"NCESSCH": str},
    )
    school_capacities = df_schools_in_play.set_index("NCESSCH")[
        "student_capacity"
    ].to_dict()

    school_clusters = {}
    for idx, row in df_mergers.iterrows():
        schools_in_cluster = row["school_cluster"].split(", ")
        for s in schools_in_cluster:
            school_clusters[s] = schools_in_cluster

    df_mergers_temp = df_mergers.copy()
    df_mergers_temp["cluster_id"] = df_mergers_temp.index
    df_mergers_temp["schools_list"] = df_mergers_temp["school_cluster"].str.split(", ")

    df_cluster_assgn = df_mergers_temp.explode("schools_list")
    df_cluster_assgn = df_cluster_assgn[["schools_list", "cluster_id"]].rename(
        columns={"schools_list": "ncessch"}
    )

    # TODO: remedy this warning (better log output)
    df_lat_long = pd.read_csv(
        "data/school_data/nces_21_22_lat_longs.csv",
        dtype={"nces_id": str},
        low_memory=False,  # suppress warning on mixed data in columns not used here
    )[["nces_id", "lat", "long"]].rename(
        columns={"lat": "zoned_lat", "long": "zoned_long"}
    )
    with open("data/school_district_2021_boundaries/district_centroids.json") as f:
        district_centroids = json.load(f)
    df_asgn_orig = (
        pd.read_csv(
            f"data/attendance_boundaries/2122/{district.state}"
            f"/estimated_student_counts_per_block.csv",
            dtype={"ncessch": str},
        )
        .drop_duplicates(subset=["block_id"])
        .rename(columns={"block_id": "GEOID20"})
    )
    df_asgn_orig = pd.merge(
        df_asgn_orig, df_names, left_on="ncessch", right_on="NCESSCH", how="left"
    )
    df_asgn_orig["percent_white"] = (
        df_asgn_orig["num_white"] / df_asgn_orig["num_total"]
    )
    df_asgn_orig["district_id"] = df_asgn_orig["ncessch"].str[:7]

    df_asgn_orig = df_asgn_orig[df_asgn_orig["district_id"] == district.id].reset_index(
        drop=True
    )

    df_asgn_orig = pd.merge(
        df_asgn_orig,
        df_lat_long,
        left_on="ncessch",
        right_on="nces_id",
        how="inner",
    )

    state_fips = us.states.lookup(district.state).fips

    # Find the district's bounding box to filter the census blocks shapefile on read.
    all_districts_gdf = gpd.read_file(
        "data/school_district_2021_boundaries/shapes/schooldistrict_sy2021_tl21.shp"
    )
    district_shape = all_districts_gdf[all_districts_gdf["GEOID"] == district.id]
    district_bbox = tuple(district_shape.total_bounds)

    state_blocks = gpd.read_file(
        f"data/census_block_shapefiles_2020/tl_2021_{state_fips}"
        f"_tabblock20/tl_2021_{state_fips}_tabblock20.shp",
        bbox=district_bbox,
    )
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
    school_demographics = df_orig.groupby("ncessch")[["num_white", "num_total"]].sum()
    per_cat_per_school = {
        "num_white": school_demographics["num_white"].to_dict(),
        "num_total": school_demographics["num_total"].to_dict(),
    }

    def make_map():
        return folium.Map(
            location=district_centroids[district.id],
            zoom_start=12,
            tiles="CartoDB positron",
        )

    m_orig = make_map()
    m_merged = make_map()
    m_dissim_pre = make_map()
    m_dissim_post = make_map()
    m_pop_pre = make_map()
    m_pop_post = make_map()
    m_both_pre = make_map()
    m_both_post = make_map()

    def add_school_markers(curr_map, school_markers):
        for m in school_markers:
            folium.Marker(
                location=[school_markers[m][0], school_markers[m][1]],
                icon=folium.Icon(color="blue", icon_color="white", icon="info-sign"),
                popup=school_markers[m][2],
            ).add_to(curr_map)

    add_school_markers(m_orig, school_markers)
    add_school_markers(m_dissim_pre, school_markers)
    add_school_markers(m_dissim_post, school_markers)
    add_school_markers(m_pop_pre, school_markers)
    add_school_markers(m_pop_post, school_markers)
    add_school_markers(m_merged, school_markers)

    df_orig_mega = df_orig.dissolve(by="ncessch", as_index=False)

    # Pre-merger population map calculations
    total_capacity_pre = 0
    school_populations_pre = per_cat_per_school["num_total"]
    schools_with_capacity = {s: c for s, c in school_capacities.items() if c and c > 0}
    if not schools_with_capacity:
        print("No schools with capacity")
        return

    total_population_pre = sum(
        school_populations_pre.get(s, 0) for s in schools_with_capacity
    )
    total_capacity_pre = sum(schools_with_capacity.values())

    if total_capacity_pre <= 0:
        print("No schools with capacity")
        return

    district_utilization_pre = total_population_pre / total_capacity_pre
    school_utilizations_pre = {
        s: school_populations_pre.get(s, 0) / schools_with_capacity[s]
        for s in schools_with_capacity
    }
    school_divergences_pre = {
        s: np.abs(u - district_utilization_pre)
        for s, u in school_utilizations_pre.items()
    }
    max_divergence_pre = (
        max(school_divergences_pre.values()) if school_divergences_pre else 0
    )

    df_orig_mega["divergence"] = (
        df_orig_mega["ncessch"].map(school_divergences_pre).fillna(0)
    )
    if max_divergence_pre > 0:
        df_orig_mega["pop_opacity"] = df_orig_mega["divergence"] / max_divergence_pre
    else:
        df_orig_mega["pop_opacity"] = 0

    def add_shape_to_map(curr_map, geo_shape, fill_color, fill_opacity, weight):
        if np.isnan(fill_opacity):
            fill_opacity = 0.2
            fill_color = "gray"

        geo_j = folium.GeoJson(
            data=geo_shape,
            style_function=lambda x: {
                "fillOpacity": fill_opacity,
                "fillColor": fill_color,
                "weight": weight,
            },
        )
        geo_j.add_to(curr_map)

    def dissim_pre(ncessch_idx):
        return 1 - (
            per_cat_per_school["num_white"][df_orig_mega["ncessch"][ncessch_idx]]
            / per_cat_per_school["num_total"][df_orig_mega["ncessch"][ncessch_idx]]
        )

    for i, r in df_orig_mega.iterrows():
        # Adding to orig map
        sim_geo = gpd.GeoSeries(r["geometry"])
        geo_j = sim_geo.to_json()

        color = colors[df_orig_mega["ncessch"][i]]
        add_shape_to_map(m_orig, geo_j, color, 0.5, ".5")

        add_shape_to_map(m_dissim_pre, geo_j, "blue", dissim_pre(i), ".5")
        add_shape_to_map(m_both_pre, geo_j, "blue", dissim_pre(i), ".5")
        add_shape_to_map(m_pop_pre, geo_j, "red", r["pop_opacity"], ".5")
        add_shape_to_map(m_both_pre, geo_j, "red", r["pop_opacity"], ".5")

    df_merged = gpd.GeoDataFrame(pd.merge(df_orig, df_cluster_assgn, on="ncessch"))

    # Compute percentage white per merged school
    cluster_demographics = df_merged.groupby("cluster_id")[
        ["num_white", "num_total"]
    ].sum()
    per_cat_per_cluster = {
        "num_white": cluster_demographics["num_white"].to_dict(),
        "num_total": cluster_demographics["num_total"].to_dict(),
    }

    # Post-merger population map calculations
    total_capacity_post = 0
    cluster_capacities = {}
    for _, row in df_mergers_temp.iterrows():
        cluster_id = row["cluster_id"]
        schools_in_cluster = row["school_cluster"].split(", ")
        cluster_capacity = 0
        for school in schools_in_cluster:
            cluster_capacity += school_capacities.get(school, 0)
        cluster_capacities[cluster_id] = cluster_capacity

    cluster_populations_post = per_cat_per_cluster["num_total"]
    clusters_with_capacity = {
        c: cap for c, cap in cluster_capacities.items() if cap and cap > 0
    }

    if not clusters_with_capacity:
        print("No clusters with capacity")
        return

    total_population_post = sum(
        cluster_populations_post.get(c, 0) for c in clusters_with_capacity
    )
    total_capacity_post = sum(clusters_with_capacity.values())

    if total_capacity_post <= 0:
        print("No clusters with capacity")
        return

    district_utilization_post = total_population_post / total_capacity_post
    cluster_utilizations_post = {
        c: cluster_populations_post.get(c, 0) / clusters_with_capacity[c]
        for c in clusters_with_capacity
    }
    cluster_divergences_post = {
        c: np.abs(u - district_utilization_post)
        for c, u in cluster_utilizations_post.items()
    }
    if not cluster_divergences_post:
        print("No cluster divergences")
        return
    max_divergence_post = max(cluster_divergences_post.values())

    df_merged_mega = df_merged.dissolve(by="cluster_id", as_index=False)

    df_merged_mega["divergence"] = (
        df_merged_mega["cluster_id"].map(cluster_divergences_post).fillna(0)
    )
    df_merged_mega["pop_opacity"] = df_merged_mega["divergence"] / max_divergence_post

    def dissim_post(cluster_id):
        return 1 - (
            per_cat_per_cluster["num_white"][cluster_id]
            / per_cat_per_cluster["num_total"][cluster_id]
        )

    for i, r in df_merged_mega.iterrows():
        # Adding to post-merger dissimilarity map
        sim_geo = gpd.GeoSeries(r["geometry"])
        geo_j = sim_geo.to_json()

        color = colors[school_clusters[df_merged_mega["ncessch"][i]][0]]
        add_shape_to_map(m_merged, geo_j, color, 0.5, ".5")

        cluster_id = r["cluster_id"]
        add_shape_to_map(m_dissim_post, geo_j, "blue", dissim_post(cluster_id), ".5")
        add_shape_to_map(m_both_post, geo_j, "blue", dissim_post(cluster_id), ".5")
        add_shape_to_map(m_pop_post, geo_j, "red", r["pop_opacity"], ".5")
        add_shape_to_map(m_both_post, geo_j, "red", r["pop_opacity"], ".5")

    m_orig.save(f"{results_dir}/orig.html")
    m_merged.save(f"{results_dir}/merged.html")
    m_dissim_pre.save(f"{results_dir}/dissim_pre.html")
    m_dissim_post.save(f"{results_dir}/dissim_post.html")
    m_pop_pre.save(f"{results_dir}/pop_pre.html")
    m_pop_post.save(f"{results_dir}/pop_post.html")
    m_both_pre.save(f"{results_dir}/both_pre.html")
    m_both_post.save(f"{results_dir}/both_post.html")


def compare_to_redistricting(
    redist_results="data/misc/all_usa_elem_within_district_sim_possible_changes.csv",
    mergers_results="data/misc/results_top_200_by_population.csv",
    output_file="data/misc/redist_mergers_comparison.csv",
):
    df_redist = pd.read_csv(redist_results, dtype={"district_id": str})
    df_redist = df_redist[df_redist["is_contiguous"]].reset_index(drop=True)

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
    viz_assignments(District.from_string(sys.argv[1]))
