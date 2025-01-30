from __future__ import annotations
from typing import Any, Callable, Generic, TypeVar
from dataclasses import dataclass

from pathlib import Path
from collections import defaultdict
import warnings
import math
import re
import json
import pickle
import random
import glob
import os

import streamlit as st
import pandas as pd

"""
This module is basically a header file that has high-level constants

It may be a good idea to pickle these later for quicker load times

Todo:
   * Do not show districts in the app list if the districts have dissimilarity
     scores of 0
   * cache stuff (st.cache_resource, st.cache_data)
"""

DEBUG_MODE = bool(os.getenv("PCG_DEBUG"))

def _conflict(value, key1, key2):
   msg = (
      f"conflicting keys found for value {value!r}: "
      f"{key1!r} and {key2!r}"
   )
   return ValueError(msg)

U = TypeVar("U")
V = TypeVar("V")

class bidict(dict, Generic[U, V]):
   """Represents a bidirectional dictionary

   The mapping must be injective (that is, the values must be unique)!
   Otherwise will raise a `ValueError`.
   """
   def __init__(self, *args, **kwargs):
      dict.__init__(self, *args, **kwargs)
      self.inverse = {}
      for key, value in self.items():
         if value in self.inverse:
            raise _conflict(value, self.inverse[value], key)
         self.inverse[value] = key

   def __getitem__(self, key):
      try:
         return dict.__getitem__(self, key)
      except KeyError:
         pass
      if key in self.inverse: return self.inverse[key]
      raise KeyError(key)

   def __setitem__(self, key, value):
      if value in self.inverse:
         if self.inverse[value] == key: return
         raise _conflict(value, self.inverse[value], key)
      if key in self:
         existing_value = self[key]
         del self.inverse[existing_value]
      dict.__setitem__(self, key, value)
      self.inverse[value] = key

   def __delitem__(self, key):
      if key in self:
         dict.__delitem__(self, key)
         return
      if key in self.inverse:
         del self.inverse[key]
         return
      raise KeyError(key)

   def __repr__(self):
      return (
         f"{self.__class__.__name__}("
         f"{dict.__repr__(self)}"
         f")"
      )

@st.cache_data(ttl=3600, show_spinner="Reading CSV files...")
def load_df(csv: Path) -> pd.DataFrame:
   return pd.read_csv(csv)

DATA_ROOT = Path("./data")
POSSIBLE_DIRS = [
   f for f in DATA_ROOT.iterdir()
   if f.is_dir() and re.match(r"^(min_elem|min_num_elem(_schools?)?)_4_([0-9a-zA-Z\-_]*)$", f.name)
]
BASE_URL = "https://mergers.schooldiversity.org"

# list of grades
GRADES: list[str]

# list of supported school enrollment constraint options
THRESHOLDS: dict[str, float]
THRESHOLDS_STR: dict[str, str]

# maps csv column names to the demographic description
# and the values of that map (labels used for plotting)
DEMOGRAPHICS: dict[str, str]
DEMOGRAPHIC_LABELS: list[str]

# maps state (e.g. "NC") or district ID to a mapping (between district IDs
# and district names or between school IDs and school names)
DISTRICTS_IN_STATE: dict[str, bidict[int, str]] # bijective!
SCHOOLS_IN_DISTRICT: dict[int, dict[int, str]] # surjective!

# sometime the district names have conflicts within the same state
# so these variables track which names aren't unique
AMBIGUOUS_DISTRICTS_IN_STATE: dict[str, set[str]]
AMBIGUOUS_SCHOOLS_IN_DISTRICT: dict[int, set[str]]

# maps IDs to the parent state or district it belongs to, from the constrained results
# the "backup" one should have all possible districts regardless
DISTRICT_ID_TO_STATE: dict[int, str]
DISTRICT_ID_TO_STATE_BACKUP: dict[int, str]
SCHOOL_ID_TO_DISTRICT_ID: dict[int, int]

# sorted lists of district names, for the frontend
DISTRICT_NAMES_IN_STATE: dict[str, list[str]]
DISTRICT_NAMES_IN_STATE_SET: dict[str, set[str]]
SCHOOL_NAMES_IN_DISTRICT: dict[int, list[str]]
SCHOOL_NAMES_IN_DISTRICT_SET: dict[int, set[str]]
CLOSED_ENROLLMENT_ONLY_DISTRICTS: set[int]

STATES: dict[str, str] = {} # state abbreviation to state name
STATES_LIST: bidict[str, str] # list of states shown to users

# overview of impacts
NUM_DISTRICTS: int = 0
NUM_SCHOOLS: int = 0
NUM_STUDENTS: int = 0
NUM_ELEMENTARY_SCHOOLS: int = 0

# ---

@st.cache_resource(show_spinner="Loading high-level constants...")
def _load_high_level_constants():
   GRADES = ["PK", "KG"] + [str(g) for g in range(1,13+1)]
   DEMOGRAPHICS = {
      "total": "Total",
      "asian": "Asian",
      "black": "Black",
      "hispanic": "Hispanic",
      "native": "Native American",
      "white": "White",
   }
   DEMOGRAPHIC_LABELS = list(DEMOGRAPHICS.values())
   DEMOS_X_GRADES = [(demo, grade) for demo in DEMOGRAPHICS for grade in GRADES]
   THRESHOLDS = {
      "0%": 0.0,
      "70%": 0.7,
      "80%": 0.8,
      "90%": 0.9,
   }
   THRESHOLDS_STR = {
      "0.0": "0%",
      "0.7": "70%",
      "0.8": "80%",
      "0.9": "90%",
   }

   return (
      GRADES,
      DEMOGRAPHICS,
      DEMOGRAPHIC_LABELS,
      DEMOS_X_GRADES,
      THRESHOLDS,
      THRESHOLDS_STR,
   )

(
   GRADES,
   DEMOGRAPHICS,
   DEMOGRAPHIC_LABELS,
   DEMOS_X_GRADES,
   THRESHOLDS,
   THRESHOLDS_STR,
) = _load_high_level_constants()

@st.cache_resource(show_spinner="Associating districts with states...")
def _associate_districts_with_states():
   print("Associating districts with states...")

   DISTRICT_ID_TO_STATE = {}
   DISTRICT_ID_TO_STATE_BACKUP = {}
   intake = DATA_ROOT / "census_block_shapefiles_2020"
   pattern = str(intake / "2122-*/2122-*-*.geodata.csv")
   for file_str in glob.glob(pattern):
      file = Path(file_str)
      _, state, district_id_str, *_ = file.name.replace(".", "-").split("-")
      district_id = int(district_id_str)
      DISTRICT_ID_TO_STATE_BACKUP[district_id] = state

   NUM_STUDENTS = 0

   df_consolidated_results = load_df((
      DATA_ROOT /
         "min_num_elem_schools_4_constrained" /
         "consolidated_simulation_results_min_num_elem_schools_4_constrained_0.2_False.csv"
   ))
   num_valid_simulations = 0
   num_invalid_pre_dissims = 0
   num_invalid_populations = 0
   for index, row in df_consolidated_results.iterrows():
      if (district_id := row["district_id"]) not in DISTRICT_ID_TO_STATE:
         pre_dissimilarity = float(row["pre_dissim"])
         population = float(row["num_total_all"])
         if pre_dissimilarity == 0.0 or math.isnan(pre_dissimilarity):
            num_invalid_pre_dissims += 1
            continue
         elif population == 0.0 or math.isnan(population):
            num_invalid_populations += 1
            continue
         else:
            NUM_STUDENTS += population
         state: str = row["state"]
         DISTRICT_ID_TO_STATE[district_id] = state
         num_valid_simulations += 1
   del df_consolidated_results
   num_invalid_simulations = num_invalid_populations + num_invalid_populations
   print((
      f"Found {num_valid_simulations} valid simulations and "
      f"{num_invalid_simulations} invalid simulations..."
   ))
   NUM_STUDENTS = round(NUM_STUDENTS)
   print(f"(info: seems to cover --> {round(NUM_STUDENTS, -3):,} <-- students)")

   return DISTRICT_ID_TO_STATE, DISTRICT_ID_TO_STATE_BACKUP, NUM_STUDENTS

# map stuff
def _parse_centroid_json_file(filepath: Path, *, integer=True) -> dict[Any, tuple[int, int]]:
   centroids = {}
   if not filepath.exists(): return centroids
   with open(filepath) as f:
      j = json.loads(f.read())
      for thing, (lat, lng) in j.items():
         if integer: thing = int(thing)
         centroids[thing] = (lat, lng)
   return centroids

@st.cache_resource(show_spinner="Making a note of locations of different districts...")
def _get_district_centroids() -> tuple[
   dict[str, tuple[int, int]],
   dict[int, tuple[int, int]],
   dict[int, tuple[int, int]],
   dict[int, tuple[int, int]],
]:
   district_centroids = {}
   school_centroids = {}
   state_centroids = {}
   sab_filepath = DATA_ROOT / "school_attendance_boundaries"
   state_centroids.update(_parse_centroid_json_file(sab_filepath / "calculated_state_centroids.json", integer=False))
   school_centroids.update(_parse_centroid_json_file(sab_filepath / "calculated_school_centroids.json"))
   district_centroids.update(_parse_centroid_json_file(sab_filepath / "calculated_district_centroids.json"))
   district_centroids.update(_parse_centroid_json_file(sab_filepath / "updated_district_centroids.json"))
   school_locations = {}
   df = pd.read_csv(DATA_ROOT / "school_attendance_boundaries" / "nces_21_22_lat_longs.csv")
   for _, (nces_id, lat, lng) in df[["nces_id", "lat", "long"]].iterrows():
      s_id = int(nces_id)
      school_locations[s_id] = (float(lat), float(lng))
   return state_centroids, district_centroids, school_centroids, school_locations

@st.cache_resource(show_spinner="Making a note of locations of different districts...")
def _load_200_adjacency() -> dict[int, dict[int, set[int]]]:
   filepath = DATA_ROOT / "school_attendance_boundaries" / "consolidated.pkl"
   if not filepath.exists(): return {}
   with open(filepath, "rb") as f: adjs = pickle.load(f)
   return adjs

(
   STATE_TO_CENTROID,
   DISTRICT_ID_TO_CENTROID,
   SCHOOL_ID_TO_CENTROID,
   SCHOOL_ID_TO_LOCATION,
) = _get_district_centroids()
DISTRICT_ADJACENCY_MAPS = _load_200_adjacency()

# etc.
DISTRICT_ID_TO_STATE, DISTRICT_ID_TO_STATE_BACKUP, NUM_STUDENTS = _associate_districts_with_states()
NUM_DISTRICTS = len(DISTRICT_ID_TO_STATE)

@st.cache_resource(show_spinner="Reading district and school names...")
def _read_district_and_school_names():
   print("Reading district and school names...")
   DISTRICTS_IN_STATE = defaultdict(lambda: bidict())
   SCHOOLS_IN_DISTRICT = defaultdict(lambda: {})
   AMBIGUOUS_DISTRICTS_IN_STATE = defaultdict(lambda: set())
   AMBIGUOUS_SCHOOLS_IN_DISTRICT = defaultdict(lambda: set())
   SCHOOL_ID_TO_DISTRICT_ID = {}
   DISTRICT_NAMES_IN_STATE = defaultdict(lambda: [])
   DISTRICT_NAMES_IN_STATE_SET = defaultdict(lambda: set())
   df_all_schools = load_df(DATA_ROOT / "all_schools_with_names.csv")

   # detect ambiguous names......
   _seen_district_ids = set()
   _seen_school_names_in_district = defaultdict(lambda: set())
   _seen_district_names_in_state = defaultdict(lambda: set())
   for index, row in df_all_schools.iterrows():
      district_id: int = int(row["district_id"])
      district_name: str = row["LEA_NAME"]
      school_id: int = row["NCESSCH"]
      school_name: str = row["SCH_NAME"]
      if school_id not in SCHOOL_ID_TO_DISTRICT_ID:
         SCHOOL_ID_TO_DISTRICT_ID[school_id] = district_id
         if school_name in _seen_school_names_in_district[district_id]:
            AMBIGUOUS_SCHOOLS_IN_DISTRICT[district_id].add(school_name)
         else:
            _seen_school_names_in_district[district_id].add(school_name)
      if district_id not in DISTRICT_ID_TO_STATE:
         # we didn't run a valid simulation for this district!
         pass
      elif district_id not in _seen_district_ids:
         _seen_district_ids.add(district_id)
         state = DISTRICT_ID_TO_STATE[district_id]
         if district_name in _seen_district_names_in_state[state]:
            AMBIGUOUS_DISTRICTS_IN_STATE[state].add(district_name)
         else:
            _seen_district_names_in_state[state].add(district_name)

   NUM_ELEMENTARY_SCHOOLS = sum(SCHOOL_ID_TO_DISTRICT_ID[s] in DISTRICT_ID_TO_STATE for s in SCHOOL_ID_TO_DISTRICT_ID)
   print(f"(info: {round(NUM_ELEMENTARY_SCHOOLS, -2):,} elementary schools)")

   print((
      f"Found ambiguous district names in {len(AMBIGUOUS_DISTRICTS_IN_STATE)} "
      f"state(s) and ambiguous school names in {len(AMBIGUOUS_SCHOOLS_IN_DISTRICT)} "
      f"district(s)."
   ))

   SCHOOL_ID_TO_DISTRICT_ID = {}
   SCHOOL_NAMES_IN_DISTRICT = defaultdict(lambda: [])
   SCHOOL_NAMES_IN_DISTRICT_SET = defaultdict(lambda: set())
   num_simulations = 0
   total = 0
   for index, row in df_all_schools.iterrows():
      district_id: int = row["district_id"]
      district_name: str = row["LEA_NAME"]
      school_id: int = row["NCESSCH"]
      school_name: str = row["SCH_NAME"]
      if school_id not in SCHOOL_ID_TO_DISTRICT_ID:
         SCHOOL_ID_TO_DISTRICT_ID[school_id] = district_id
         if school_name in AMBIGUOUS_SCHOOLS_IN_DISTRICT[district_id]:
            school_name = f"{school_name} (NCES ID: {school_id:012d})"
         SCHOOLS_IN_DISTRICT[district_id][school_id] = school_name
         SCHOOL_NAMES_IN_DISTRICT[district_id].append(school_name)
      if district_id not in DISTRICT_ID_TO_STATE:
         # we didn't run a valid simulation for this district!
         pass
      else:
         state = DISTRICT_ID_TO_STATE[district_id]
         if district_name in AMBIGUOUS_DISTRICTS_IN_STATE[state]:
            district_name = f"{district_name} (NCES ID: {district_id:07d})"
         DISTRICTS_IN_STATE[state][district_id] = district_name
         if district_name not in DISTRICT_NAMES_IN_STATE_SET[state]:
            DISTRICT_NAMES_IN_STATE_SET[state].add(district_name)
            DISTRICT_NAMES_IN_STATE[state].append(district_name)

            num_simulations += 1
      total += 1

   del df_all_schools

   print((
      f"Found simulations for {num_simulations/total:.01%} of districts "
      f"({num_simulations}/{total})"
   ))

   for state in DISTRICT_NAMES_IN_STATE:
      DISTRICT_NAMES_IN_STATE[state].sort()

   for d_id in SCHOOL_NAMES_IN_DISTRICT:
      SCHOOL_NAMES_IN_DISTRICT[d_id].sort()
      SCHOOL_NAMES_IN_DISTRICT_SET[d_id] = set(SCHOOL_NAMES_IN_DISTRICT[d_id])

   df_ceo = pd.read_csv("data/entirely_elem_closed_enrollment_districts.csv")
   CLOSED_ENROLLMENT_ONLY_DISTRICTS = set(df_ceo["district_id"])

   return (
      DISTRICTS_IN_STATE,
      SCHOOLS_IN_DISTRICT,
      AMBIGUOUS_DISTRICTS_IN_STATE,
      AMBIGUOUS_SCHOOLS_IN_DISTRICT,
      SCHOOL_ID_TO_DISTRICT_ID,
      DISTRICT_NAMES_IN_STATE,
      DISTRICT_NAMES_IN_STATE_SET,
      SCHOOL_NAMES_IN_DISTRICT,
      SCHOOL_NAMES_IN_DISTRICT_SET,
      NUM_ELEMENTARY_SCHOOLS,
      CLOSED_ENROLLMENT_ONLY_DISTRICTS,
   )

(
   DISTRICTS_IN_STATE,
   SCHOOLS_IN_DISTRICT,
   AMBIGUOUS_DISTRICTS_IN_STATE,
   AMBIGUOUS_SCHOOLS_IN_DISTRICT,
   SCHOOL_ID_TO_DISTRICT_ID,
   DISTRICT_NAMES_IN_STATE,
   DISTRICT_NAMES_IN_STATE_SET,
   SCHOOL_NAMES_IN_DISTRICT,
   SCHOOL_NAMES_IN_DISTRICT_SET,
   NUM_ELEMENTARY_SCHOOLS,
   CLOSED_ENROLLMENT_ONLY_DISTRICTS,
) = _read_district_and_school_names()
NUM_SCHOOLS = len(SCHOOL_ID_TO_DISTRICT_ID)

@st.cache_resource(show_spinner="Reading states list...")
def _read_states_list():
   df_state_codes = load_df(DATA_ROOT / "state_codes.csv")

   STATES = {}
   for index, row in df_state_codes.iterrows():
      STATES[row["abbrev"]] = row["name"]

   del df_state_codes

   if (expected := len(STATES)) > (found := len(DISTRICT_NAMES_IN_STATE)):
      missing_abbreviated = set(STATES) - set(DISTRICT_NAMES_IN_STATE)
      missing_named = tuple(
         STATES[s]
         for s in missing_abbreviated
      )
      msg = (
         f"Expected {expected} states, found {found} "
         f"(missing: {missing_named!r})"
      )
      #print(msg)
      for abbreviation in missing_abbreviated:
         del STATES[abbreviation]

   STATES_LIST = bidict({f"{abbrev} - {name}": abbrev for abbrev, name in STATES.items()})

   return STATES, STATES_LIST

STATES, STATES_LIST = _read_states_list()
