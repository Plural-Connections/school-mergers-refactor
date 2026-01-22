import geopandas as gpd
import folium
import us
import pandas as pd
import numpy as np
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


def _calculate_population_metrics(capacities, demographics):
    district_utilization = demographics["num_total"].sum() / capacities.sum()
    utilizations = demographics["num_total"] / capacities
    divergences = np.abs(utilizations - district_utilization)
    return np.sqrt(divergences / divergences.max())


def _calculate_dissimilarity_metrics(demographics):
    white_shares = demographics["num_white"] / demographics["num_white"].sum()
    nonwhite = demographics["num_total"] - demographics["num_white"]
    nonwhite_shares = (nonwhite) / nonwhite.sum()
    dissim_per_school = np.abs(white_shares - nonwhite_shares)
    return np.sqrt(dissim_per_school / dissim_per_school.max())


def viz_assignments(
    # of the format data/results/{district.state}/{district.id}/{run}
    dir: str,
    save_file=False,
):
    _, _, state, district_id, run = dir.split("/")
    district = District(state=state, id=district_id)

    df_names = pd.read_csv("data/all_schools_with_names.csv", dtype={"NCESSCH": str})[
        ["NCESSCH", "SCH_NAME"]
    ]

    df_mergers = pd.read_csv(
        f"{dir}/school_mergers.csv",
        dtype={"school_cluster": str},
    )

    df_schools_in_play = pd.read_csv(
        f"{dir}/schools_in_play.csv",
        dtype={"NCESSCH": str},
    )
    capacities = df_schools_in_play.set_index("NCESSCH")["student_capacity"]

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
    df_asgn_orig["district_id"] = df_asgn_orig["ncessch"].str[:7]
    df_asgn_orig = df_asgn_orig[df_asgn_orig["district_id"] == district.id].reset_index(
        drop=True
    )
    df_demographics = (
        df_asgn_orig[["ncessch", "num_white", "num_total"]].groupby("ncessch").sum()
    )
    df_asgn_orig.drop(columns=["num_white", "num_total"], inplace=True)

    df_asgn_orig = df_asgn_orig.merge(
        df_names,
        left_on="ncessch",
        right_on="NCESSCH",
        how="left",
    ).merge(
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

    pre = (
        gpd.GeoDataFrame(
            pd.merge(
                df_asgn_orig,
                state_blocks,
                on="GEOID20",
                how="inner",
            )
        )
        .to_crs(epsg=4326)
        .dissolve(by="ncessch")
        .merge(
            df_demographics,
            left_on="ncessch",
            right_index=True,
            how="inner",
        )
    )

    def gen_color(nces_id):
        random.seed(int(nces_id))
        return f"#{random.randint(0, 0xFFFFFF):06x}"

    school_markers = pre[["zoned_lat", "zoned_long", "SCH_NAME"]]
    pre_nums = pre[["num_white", "num_total"]]
    post_nums = post[["num_white", "num_total"]]
    pre = pre[["geometry"]]
    pre["color"] = pre.index.map(gen_color)
    pre["weight"] = 0.2
    pre["opacity"] = 1

    post = (
        gpd.GeoDataFrame(pre.merge(df_cluster_assgn, on="ncessch"))
        .dissolve("cluster_id")
        .set_index("ncessch")
    )

    map = folium.Map(
        location=district_centroids[district.id],
        zoom_start=12,
        tiles="CartoDB positron",
    )

    for _, r in school_markers.iterrows():
        folium.Marker(
            location=[r["zoned_lat"], r["zoned_long"]],
            icon=folium.Icon(color="blue", icon_color="white", icon="info-sign"),
            popup=r["SCH_NAME"],
        ).add_to(map)

    def identity_style_function(feature):
        return {
            "fillColor": feature["properties"]["color"],
            "fillOpacity": feature["properties"]["opacity"],
            "weight": feature["properties"]["weight"],
        }

    def add_layer(name):
        nonlocal map, pre, post

    folium.GeoJson(data=pre, style_function=identity_style_function).add_to(map)
    folium.GeoJson(data=post, style_function=identity_style_function).add_to(map)

    pre["color"] = post["color"] = "blue"

    pre["opacity"] = _calculate_population_metrics(capacities, pre_nums)
    folium.GeoJson(data=pre, style_function=identity_style_function).add_to(map)

    post["opacity"] = _calculate_population_metrics(capacities, post_nums)
    folium.GeoJson(data=post, style_function=identity_style_function).add_to(map)

    pre["color"] = post["color"] = "red"

    pre["opacity"] = _calculate_dissimilarity_metrics(pre_nums)
    folium.GeoJson(data=pre, style_function=identity_style_function).add_to(map)

    post["opacity"] = _calculate_dissimilarity_metrics(post_nums)
    folium.GeoJson(data=post, style_function=identity_style_function).add_to(map)

    folium.LayerControl().add_to(map)

    map.save(f"{dir}/map.html")


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
    viz_assignments(sys.argv[1].strip("/"))
