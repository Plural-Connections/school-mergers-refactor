#!/usr/bin/env python3

"""
Generates shape file (geopandas csv dumps) per district for a given state
"""

import mergers_core.utils.header as header
import pandas as pd

import folium
import geopandas as gpd
import os
import us
from pathlib import Path

# Set up the location of all the static data.
# DATA_DIR = "../s3/"
DATA_DIR = ""


if __name__ == "__main__":
    year = "2122"
    states = [s.split("/")[0] for s in os.listdir("data/attendance_boundaries/2122/")]
    for s in states:
        state_fips = us.states.lookup(s).fips

        output_dir = DATA_DIR + ("data/census_block_shapefiles_2020/%s-%s" % (year, s))

        print("Processing {}...".format(s))

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        filename_school_state_data = (
            DATA_DIR + "data/solver_files/2122/{}/school_enrollments.csv"
        )
        filename_blocks_to_schools_file = (
            DATA_DIR
            + "data/attendance_boundaries/2122/{}/estimated_student_counts_per_block.csv"
        )

        try:
            df_school_state_data = pd.read_csv(
                filename_school_state_data.format(s), dtype={"NCESSCH": str}
            )
        except Exception as e:
            continue

        df_school_state_data["leaid"] = df_school_state_data["NCESSCH"].str[:7]

        districts = list(df_school_state_data["leaid"].unique())
        filename_blocks_shape_file = (
            DATA_DIR
            + "/Users/ngillani/OneDrive - Northeastern University/neu/rezoning-schools/data/census_block_shapefiles_2020/tl_2021_{}_tabblock20/tl_2021_{}_tabblock20.shp"
        )

        df_blocks = gpd.read_file(
            filename_blocks_shape_file.format(state_fips, state_fips), dtype={""}
        )

        df_asgn_orig = pd.read_csv(
            filename_blocks_to_schools_file.format(s),
            dtype={"ncessch": str, "block_id": str},
        )
        df_asgn_orig["leaid"] = df_asgn_orig["ncessch"].str[:7]

        for district in districts:
            fname = output_dir + "/%s-%s-%s.geodata.csv" % (year, s, district)
            if os.path.exists(fname):
                print(fname)
                continue
            print(district)
            df_asgn_orig_d = df_asgn_orig[df_asgn_orig["leaid"] == district]
            df_orig = gpd.GeoDataFrame(
                pd.merge(
                    df_asgn_orig_d,
                    df_blocks,
                    left_on="block_id",
                    right_on="GEOID20",
                    how="inner",
                )
            )
            df_orig.crs = "epsg:4326"
            # df_orig["geometry"] = df_orig["geometry"].simplify(0.0002)

            print(fname)
            df_orig.to_csv(fname)
