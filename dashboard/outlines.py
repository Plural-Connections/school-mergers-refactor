from __future__ import annotations
from genericpath import isdir
from typing import Callable, Iterable, Optional
from collections import defaultdict

from pathlib import Path
import glob
import pickle
import warnings
import math
import json

import numpy as np
import scipy.stats
import pandas as pd
import geopandas as gpd
from shapely import wkt
from tqdm import tqdm
from shapely.geometry import MultiPolygon, Polygon

from headers import (
    DEMOGRAPHICS,
    DEMOGRAPHIC_LABELS,
    DEMOS_X_GRADES,
    GRADES,
    DISTRICTS_IN_STATE,
    DISTRICT_ID_TO_STATE,
    STATES,
)
import eat
from eat import District

"""
Compiles the outlines files, so the schools & districts can have nice outlines
in the dashboard map

TODO: simplify the outlines so they load faster?
"""


# https://stackoverflow.com/a/70387141
def _remove_interiors(poly: Polygon) -> Polygon:
    """Close polygon holes by limitation to the exterior ring"""
    if poly.interiors:
        return Polygon(list(poly.exterior.coords))
    else:
        return poly


def _pop_largest(gs: gpd.GeoSeries) -> Polygon | MultiPolygon:
    """Pop the largest polygon off of a GeoSeries"""
    geoms = [g.area for g in gs]
    return gs.pop(geoms.index(max(geoms)))


def close_holes(geom: Polygon | MultiPolygon) -> Polygon | MultiPolygon:
    """Remove holes in a polygon geometry"""
    if isinstance(geom, MultiPolygon):
        ser = gpd.GeoSeries([_remove_interiors(g) for g in geom.geoms])
        big = _pop_largest(ser)
        outers = ser.loc[~ser.within(big)].tolist()
        if outers:
            return MultiPolygon([big] + outers)
        return Polygon(big)
    assert isinstance(geom, Polygon)
    return _remove_interiors(geom)


def make_all_the_outlines(
    *,
    intake: str | Path = Path("data/school_attendance_boundaries"),
    output: str | Path = Path("data/school_attendance_boundaries/outlines"),
) -> None:
    intake = Path(intake)
    output = Path(output)
    if not intake.is_absolute():
        intake = Path.cwd() / intake

    all_states = []
    for folder in intake.iterdir():
        if not folder.is_dir():
            continue
        if not folder.name.isupper() or not len(folder.name) == 2:
            continue
        all_states.append(folder)
    all_states.sort()
    for folder in all_states:
        state = folder.name
        # if state != "NC": continue
        print(f"info: tempering {state}...")
        for csv in folder.iterdir():
            if csv.suffix != ".csv":
                continue
            try:
                district_id = int(csv.stem)
            except ValueError:
                continue
            # load this district's df
            df = gpd.read_file(csv, ignore_geometry=True)
            df["geometry"] = gpd.GeoSeries.from_wkt(df["geometry"])
            gdf = gpd.GeoDataFrame(df)
            # prepare output df
            gdf_out = gpd.GeoDataFrame({"nces_id": [], "geometry": []})
            output_file = output / f"{state}/{district_id:07d}.csv"
            output_file.parent.mkdir(exist_ok=True, parents=True)
            # outline of entire district
            gdf["dummy"] = 1
            gdf_entire_district = gdf.dissolve(by="dummy", as_index=False)
            try:
                entire_district = gdf_entire_district["geometry"].iloc[0]
            except IndexError:
                print(f"warning: {state}/{csv.name} has nothing?")
                continue
            entire_district = close_holes(entire_district)
            gdf_out = pd.concat(
                [
                    gdf_out,
                    gpd.GeoDataFrame(
                        {
                            "nces_id": [f"{district_id:07d}"],
                            "geometry": [entire_district],
                        }
                    ),
                ]
            )
            # outline of each school
            for _, (school_id, geom, _) in gdf.iterrows():
                school_id = int(school_id)
                geom = close_holes(geom)
                gdf_out = pd.concat(
                    [
                        gdf_out,
                        gpd.GeoDataFrame(
                            {"nces_id": [f"{school_id:012d}"], "geometry": [geom]}
                        ),
                    ]
                )
            # simplify the geometries, for faster loading
            # tolerance is in units of lat/long
            # note that 1 degree difference in lat/long = 111 km
            # tolerance of 50 meters seems to work well
            gdf_out["geometry"] = gdf_out["geometry"].simplify(tolerance=50 / 111000)
            # save outline file
            pd.DataFrame(gdf_out).to_csv(output_file, encoding="utf-8", index=False)


if __name__ == "__main__":
    make_all_the_outlines()
