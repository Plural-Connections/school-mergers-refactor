from __future__ import annotations
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

def dissolve_census_blocks(*,
   intake: str | Path=Path("data/census_block_shapefiles_2020"),
   output: str | Path=Path("data/school_attendance_boundaries"),
) -> None:
   """Takes in the census block to school attendance mappings and melts them
   into school attendance boundaries

   Each district gets its own csv
   """
   intake = Path(intake); output = Path(output)
   if not intake.is_absolute(): intake = Path.cwd() / intake
   pattern = str(intake / "2122-*/2122-*-*.geodata.csv")
   for file_str in tqdm(glob.glob(pattern)):
      file = Path(file_str)
      _, state, district_id_str, *_ = file.name.replace(".", "-").split("-")
      district_id = int(district_id_str)
      output_file = output / f"{state}/{district_id_str}.csv"
      if output_file.exists(): continue
      output_file.parent.mkdir(exist_ok=True, parents=True)

      df = gpd.read_file(file, ignore_geometry=True)
      df["geometry"] = gpd.GeoSeries.from_wkt(df["geometry"])
      df = df[["ncessch", "geometry"]]
      gdf = gpd.GeoDataFrame(df)
      gdf_dissolved = gdf.dissolve(by="ncessch", as_index=False)

      pd.DataFrame(gdf_dissolved).to_csv(output_file, encoding="utf-8", index=False)

def manually_calculate_centroids():
   """Calculate district centroids as average of all school centroids

   Go through the CSVs that were generated above, calculate the centroids
   for every district, and save that into calculated_district_centroids.json
   """
   state_centroids = {}
   district_centroids = {}
   schools_centroids = {}
   for folder in tqdm(Path("data/school_attendance_boundaries").iterdir()):
      if not folder.is_dir() or folder.name not in STATES: continue
      state = folder.name
      total_district_lat = 0
      total_district_lng = 0
      district_count = 0
      for file in folder.iterdir():
         if file.suffix != ".csv": continue
         district_id = file.stem
         df = gpd.read_file(file, ignore_geometry=True)
         df["geometry"] = gpd.GeoSeries.from_wkt(df["geometry"])
         df = df[["ncessch", "geometry"]]
         gdf = gpd.GeoDataFrame(df)
         gdf["centroid"] = gdf["geometry"].centroid
         total_lat, total_lng = 0, 0
         count = 0
         for _, (ncessch, geometry, centroid) in gdf.iterrows():
            s_id = int(ncessch)
            schools_centroids[f"{s_id:012d}"] = [centroid.x, centroid.y]
            total_lat += centroid.x
            total_lng += centroid.y
            count += 1
         try:
            district_centroids[district_id] = [total_lat/count, total_lng/count]
         except ZeroDivisionError:
            print("ZeroDivisionError:", str(file))
         else:
            total_district_lat += district_centroids[district_id][0]
            total_district_lng += district_centroids[district_id][1]
            district_count += 1
      try:
         state_centroids[state] = [total_district_lat/district_count, total_district_lng/district_count]
      except ZeroDivisionError:
         print("ZDE:", str(folder))

   with open("data/school_attendance_boundaries/calculated_state_centroids.json", "w") as f:
      json.dump(state_centroids, f, sort_keys=True)
   with open("data/school_attendance_boundaries/calculated_district_centroids.json", "w") as f:
      json.dump(district_centroids, f, sort_keys=True)
   with open("data/school_attendance_boundaries/calculated_school_centroids.json", "w") as f:
      json.dump(schools_centroids, f, sort_keys=True)

def adjacency(district_id: int) -> dict[int, set[int]]:
   """Return the adjacency network of a district's schools

   Mapping from school NCES ID to set of school NCES IDs.

   O(n^2)
   """
   state = DISTRICT_ID_TO_STATE[district_id]
   filepath = (
      Path("data/school_attendance_boundaries")
      / f"{state}"
      / f"{district_id:07d}.csv"
   )
   df = gpd.read_file(filepath, ignore_geometry=True)
   df["geometry"] = wkt.loads(df["geometry"])
   gdf = gpd.GeoDataFrame(df)

   adj = {}
   for i, row in tqdm(gdf.iterrows(), total=len(gdf["ncessch"])):
      school_id = row["ncessch"]
      geometry = row["geometry"]
      for j, other_row in gdf.iterrows():
         other_school_id = other_row["ncessch"]
         if school_id == other_school_id: continue
         if school_id in adj and other_school_id in adj[school_id]: continue
         if school_id not in adj: adj[school_id] = set()
         if other_school_id not in adj: adj[other_school_id] = set()

         other_geometry = other_row["geometry"]
         if geometry.touches(other_geometry):
            adj[school_id].add(other_school_id)
            adj[other_school_id].add(school_id)

   return adj

def adjacency_v2(district_id: int) -> dict[int, set[int]]:
   """Return the adjacency network of a district's schools

   Mapping from school NCES ID to set of school NCES IDs.

   O(n^2)
   """
   state = DISTRICT_ID_TO_STATE[district_id]
   filepath = (
      Path("data/school_attendance_boundaries")
      / f"{state}"
      / f"{district_id:07d}.csv"
   )
   df = gpd.read_file(filepath, ignore_geometry=True)
   df["geometry"] = wkt.loads(df["geometry"])
   gdf = gpd.GeoDataFrame(df)

   adj = {}
   for i, row in tqdm(gdf.iterrows(), total=len(gdf["ncessch"])):
      school_id = row["ncessch"]
      geometry = row["geometry"]
      adj[school_id] = set()
      neighbors = gdf[~gdf.geometry.disjoint(geometry)]
      for j, neighbor in neighbors.iterrows():
         other_school_id = neighbor["ncessch"]
         if school_id == other_school_id: continue
         adj[school_id].add(other_school_id)

   return adj

def save_200_adjacency() -> None:
   """Save pickle file of adjacency information

   266 kB file, last time I ran it
   """
   district_ids = get_200_districts()
   adjs: dict[int, dict[int, set[int]]] = {}
   for district_id in district_ids:
      adjs[district_id] = adjacency_v2(district_id)
      #break
   filepath = Path("data/school_attendance_boundaries/consolidated.pkl")
   with open(filepath, "wb") as f:
      pickle.dump(adjs, f)

def load_200_adjacency() -> dict[int, dict[int, set[int]]]:
   filepath = Path("data/school_attendance_boundaries/consolidated.pkl")
   with open(filepath, "rb") as f: adjs = pickle.load(f)
   return adjs

def save_200_demographics():
   """saves demos data to ``figures/demographics_for_morans_i.csv``
   """
   district_ids = get_200_districts()
   from make_figures import _all_districts
   all_districts = _all_districts()
   data = {
      "district_id": [],
      "school_id": [],
      "proportion_white": [],
      "total": [],
   }
   for district_id in district_ids:
      for school_id, school in all_districts[district_id].schools.items():
         data["district_id"].append(f"{district_id:07d}")
         data["school_id"].append(f"{school_id:013d}")
         num_total = school.population_before.total or 0
         num_white = school.population_before.white or 0
         data["proportion_white"].append(num_white / num_total)
         data["total"].append(num_total)
   df = pd.DataFrame(data)
   df.to_csv("figures/demographics_for_morans_i.csv")

def load_200_demographics(proportion_nonwhite: bool=False) -> dict[int, dict[int, float]]:
   """Proportion White per school (the x values for Moran's I)

   Args:
      proportion_nonwhite: if False, return proportion White; otherwise, return
         proportion non-White
   """
   df = pd.read_csv("figures/demographics_for_morans_i.csv")
   demographics = {}
   for _, row in df.iterrows():
      d_id = row["district_id"].astype(int)
      s_id = row["school_id"].astype(int)
      p = row["proportion_white"].astype(float)
      if d_id not in demographics: demographics[d_id] = {}
      demographics[d_id][s_id] = p if not proportion_nonwhite else (1 - p)
   return demographics

def load_200_population() -> dict[int, dict[int, float]]:
   """ fjkbfkd jkdn fldnlk
   """
   df = pd.read_csv("figures/demographics_for_morans_i.csv")
   population = {}
   for _, row in df.iterrows():
      d_id = row["district_id"].astype(int)
      s_id = row["school_id"].astype(int)
      total = row["total"].astype(int)
      if d_id not in population: population[d_id] = {}
      population[d_id][s_id] = total
   return population

def gearys_c(
   adj: dict[int, set[int]],
   x: dict[int, float],
   t: None | dict[int, float]=None,
) -> float:
   """Calculate Geary's C for the district

   Args:
      adj: Adjacency map listing which schools are adjacent to which other
         schools for some district
      x: Mapping from school to proportion White (or proportion non-White)
      t: blah
   """
   adj = {k: v for k, v in adj.items() if len(v)} # remove singleton islands
   n = len(adj)
   w = np.zeros((n, n))
   for i, school_id in enumerate(adj):
      for j, other_school_id in enumerate(adj):
         if i == j: continue
         w[i, j] = int(other_school_id in adj[school_id])
         if t is not None: w[i, j] *= t[int(school_id)]
   # row standardize
   w /= w.sum(axis=1).reshape(-1, 1)
   w = np.nan_to_num(w)
   var = 0
   wsquares = 0
   if t is None:
      mean_proportion = np.mean([x[int(s_id)] for s_id in adj])
   else:
      mean_proportion = np.sum([x[int(s_id)] * t[int(s_id)] for s_id in adj])
      mean_proportion /= np.sum([t[int(s_id)] for s_id in adj])
   for i, school_id in enumerate(adj):
      proportion = x[int(school_id)]
      var += (proportion - mean_proportion) ** 2
      for j, other_school_id in enumerate(adj):
         other_proportion = x[int(other_school_id)]
         wsquares += w[i, j] * (proportion - other_proportion) ** 2
   c = wsquares / var
   c *= (n - 1) / (2 * np.sum(w))
   return c

def morans_i(
   adj: dict[int, set[int]],
   x: dict[int, float],
   t: None | dict[int, float]=None,
) -> float:
   """Calculate Moran's I for the district, v3

   Weight matric weighted by population size

   Args:
      adj: Adjacency map listing which schools are adjacent to which other
         schools for some district
      x: Mapping from school to proportion White (or proportion non-White)
      t: population
   """
   adj = {k: v for k, v in adj.items() if len(v)} # remove singleton islands
   n = len(adj)
   w = np.zeros((n, n))
   for i, school_id in enumerate(adj):
      for j, other_school_id in enumerate(adj):
         if i == j: continue
         w[i, j] = int(other_school_id in adj[school_id])
         if t is not None: w[i, j] *= t[int(school_id)]
   # row standardize
   w /= w.sum(axis=1).reshape(-1, 1)
   w = np.nan_to_num(w)
   var = 0
   wcovar = 0
   mean_proportion = np.mean([x[int(s_id)] for s_id in adj])
   for i, school_id in enumerate(adj):
      proportion = x[int(school_id)]
      var += (proportion - mean_proportion) ** 2
      for j, other_school_id in enumerate(adj):
         other_proportion = x[int(other_school_id)]
         wcovar += (
            w[i, j] * (proportion - mean_proportion)
            * (other_proportion - mean_proportion)
         )
   i = wcovar / var
   i *= n / np.sum(w)
   return i

def get_200_districts() -> list[int]:
   df = pd.read_csv("figures/top_200_by_population.csv")
   return list(map(int, df["nces_id"]))

def main() -> None:
   """Save all results to ``figures/morans_i_results.csv`` !!
   """
   adjs = load_200_adjacency()
   xs = load_200_demographics()
   ts = load_200_population()
   df = pd.read_csv("figures/top_200_by_population.csv")
   values = []
   for d in tqdm(df["nces_id"]):
      try:
         i = morans_i(adjs[d], xs[d], ts[d])
         values.append(i)
      # except KeyError:
      #    print(d, "key error")
      #    values.append(math.nan)
      except ZeroDivisionError:
         print(d, "zero division error")
         #values.append(math.nan)
         values.append(0)

   df["change_dissim"] = (df["post_dissim"] - df["pre_dissim"]) / df["pre_dissim"]

   #return
   df["morans_i"] = values
   df = df[["nces_id", "pre_dissim", "post_dissim", "change_dissim", "morans_i"]]
   df.to_csv("figures/morans_i_results.csv")

def main_gearys_c() -> None:
   """Save all results to ``figures/gearys_c_results.csv`` !!
   """
   adjs = load_200_adjacency()
   xs = load_200_demographics()
   ts = load_200_population()
   df = pd.read_csv("figures/top_200_by_population.csv")
   values = []
   for d in tqdm(df["nces_id"]):
      try:
         i = gearys_c(adjs[d], xs[d], ts[d])
         values.append(i)
      except KeyError:
         print(d, "key error")
         values.append(math.nan)
      except ZeroDivisionError:
         print(d, "zero division error")
         #values.append(math.nan)
         values.append(0)

   df["change_dissim"] = (df["post_dissim"] - df["pre_dissim"]) / df["pre_dissim"]

   #return
   df["gearys_c"] = values
   df = df[["nces_id", "pre_dissim", "post_dissim", "change_dissim", "gearys_c"]]
   df.to_csv("figures/gearys_c_results.csv")

def test_morans_i_2():
   district_id = 4703030
   adjs = load_200_adjacency(); adj = adjs[district_id]
   return adj

if __name__ == "__main__":
   #dissolve_census_blocks()
   manually_calculate_centroids()
   #save_200_adjacency()
   #save_200_demographics()
   #adj = test_morans_i_2()
   #main()
   #main_gearys_c()
