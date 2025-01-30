from __future__ import annotations
from typing import (
   Any, Callable, TypeVar, Generic, Optional,
   Collection, ClassVar, Literal,
)
from dataclasses import dataclass
from enum import Enum

from pathlib import Path
import math
import warnings
import json
from collections import defaultdict
import re
from pprint import pformat
from functools import cached_property

import streamlit as st
import pandas as pd

from headers import (
   POSSIBLE_DIRS,
   load_df,
   DEMOGRAPHICS,
   GRADES,
   DEMOS_X_GRADES,
   SCHOOLS_IN_DISTRICT,
   SCHOOL_ID_TO_DISTRICT_ID,
   DISTRICTS_IN_STATE,
   DISTRICT_ID_TO_STATE,
   DATA_ROOT,
   CLOSED_ENROLLMENT_ONLY_DISTRICTS,
   DISTRICT_ID_TO_CENTROID,
   SCHOOL_ID_TO_CENTROID,
   SCHOOL_ID_TO_LOCATION,
)

"""
This module specializes in eating (interpreting) all the CSV files.
"""

T = TypeVar("T")

DemoType = Literal[
   "color",
   "asian",
   "black",
   "hispanic",
   "native",
   "white",
]
_demo_type: list[DemoType] = [
   "color",
   "asian",
   "black",
   "hispanic",
   "native",
   "white",
]

class StatusCode(Enum):
   UNKNOWN = 0
   INVALID = 1 # model is invalid
   FEASIBLE = 2
   INFEASIBLE = 3
   OPTIMAL = 4

@dataclass
class Simulation:
   """Represents a simulation's context

   Attributes:
      state: state abbreviation (e.g. "NC")
      district_id: district NCES ID
      interdistrict: whether this simulation was done between districts
      bottomless: whether this simulation has a capacity threshold (False) or
         not (True)
      school_descrease_threshold: max proportion (e.g. 0.1) that schools were
         allowed to decrease their population by in this simulation
      folder_name_check: Parity check, make sure the specific folder name of
         this simulation contains this substring (e.g.,
         ``"constrained_bh_wa"``) (default is not to check for a substring)
      time_to_compute: time to compute this simulation, in seconds (extra
         detail from CPSAT)
      branches: extra detail from CPSAT
      conflicts: extra detail from CPSAT
   """
   state: str
   district_id: int
   interdistrict: bool
   school_decrease_threshold: str  # now required
   status: Optional[StatusCode] = None
   time_to_compute: Optional[float] = None
   branches: Optional[int] = None
   conflicts: Optional[int] = None
   folder_name_check: str | None = None
   __hash__ = lambda self: hash(id(self))

   def __post_init__(self):
      # Updates with extra statistics found for this simulation
      try:
         file = self.data_path
      except FileNotFoundError:
         return
      pattern = r"^(?P<interdistrict>.*?)_(?P<school_decrease_threshold>.*?)_(?P<status>.*?)_(?P<time_to_compute>.*?)_(?P<branches>.*?)_(?P<conflicts>.*?)$"

      match = re.match(pattern, file.name)
      if match:
         self.status = StatusCode(int(match.group("status")))
         self.time_to_compute = float(match.group("time_to_compute"))
         self.branches = int(match.group("branches"))
         self.conflicts = int(match.group("conflicts"))

   @cached_property
   def analytics_filepath(self) -> Path:
      """Returns the relevant ``analytics.csv`` filepath of this simulation
      """
      threshold = self.school_decrease_threshold
      if threshold == "1": threshold = "1.0"
      batch_name = rf"[0-9a-zA-Z\-_]*" if self.folder_name_check is None else self.folder_name_check
      pattern = rf"^consolidated_simulation_results_(min_num_elem(_schools?)?|min_elem)_4_({batch_name})_{threshold}_{self.interdistrict}.csv$"
      for dir in POSSIBLE_DIRS:
         for file in dir.iterdir():
            if not file.name.endswith(".csv"): continue
            if re.match(pattern, file.name):
               #print(f"Found {file}")
               return file
      raise FileNotFoundError(f"{pattern!r} <-- looking for {self} in {POSSIBLE_DIRS = !r}")

   @cached_property
   def district_path(self) -> Path:
      """Returns the district folder corresponding to this simulation
      """
      subfolder = self.analytics_filepath.parent
      return (
         subfolder /
         self.state /
         f"{self.district_id:07d}"
      )

   @cached_property
   def data_path(self) -> Path:
      """Returns the data folder that corresponds to this simulation

      Warning: This is overengineered.
      """
      possible_thresholds = [self.school_decrease_threshold] + (["1.0"] if self.school_decrease_threshold == "1" else [])
      tries = []
      district_path = self.district_path
      if not district_path.exists():
         raise FileNotFoundError(f"{str(district_path)!r} for {self!r}")
      while possible_thresholds:
         threshold = possible_thresholds.pop()
         prefix = f"{self.interdistrict}_{threshold}_"
         tries.append(prefix)
         for file in district_path.iterdir():
            if file.name.startswith(prefix) and file.is_dir():
               return file
      msg = (
         f"folder under {str(district_path)!r}? with prefixes: {tuple(tries)}... for {self!r}"
      )
      raise FileNotFoundError(msg)

   @cached_property
   def demos_before_filepath(self) -> Path:
      """Returns the ``schools_in_play.csv`` filepath of this simulation
      """
      return self.data_path / "schools_in_play.csv"

   @cached_property
   def demos_after_filepath(self) -> Path:
      """Returns the ``students_per_group_per_school_post_merger.json``
      filepath of this simulation
      """
      return self.data_path / "students_per_group_per_school_post_merger.json"

   @cached_property
   def demos_x_grades_after_filepath(self) -> Path:
      """Returns the
      ``students_per_grade_per_group_per_school_post_merger.json`` file for
      this simulation, which is needed for `School.grades_population_after`.
      """
      return self.data_path / "students_per_grade_per_group_per_school_post_merger.json"

   @cached_property
   def grades_offered_after_filepath(self) -> Path:
      """Returns the ``grades_served.csv`` filepath of this simulation
      """
      return self.data_path / "grades_served.csv"

   @cached_property
   def clusters_filepath(self) -> Path:
      """Returns the ``school_mergers.csv`` filepath of this simulation
      """
      return self.data_path / "school_mergers.csv"

   @cached_property
   def bottomless(self) -> bool:
      return self.school_decrease_threshold in ("1", "1.0")

def _roundish(a, *_) -> int:
   if a is None or math.isnan(a): return 0
   return math.ceil(a)

@dataclass
class Population:
   """Represents a distribution of demographics

   Note:
      `Population.total` is not automatically calculated and must be computed
      or come from a CSV.
   """
   total: Optional[float] = None
   asian: Optional[float] = None
   black: Optional[float] = None
   hispanic: Optional[float] = None
   native: Optional[float] = None
   not_specified: Optional[float] = None
   pacific_islander: Optional[float] = None
   two_or_more: Optional[float] = None
   white: Optional[float] = None
   __hash__ = lambda self: hash(id(self))

   def __getitem__(self, item: str) -> Optional[float]:
      if item not in DEMOGRAPHICS and item != "color":
         raise KeyError(f"{item!r} not in {tuple(DEMOGRAPHICS)!r}")
      return self.__getattribute__(item)

   def __setitem__(self, item: str, value: Optional[float]):
      if item not in DEMOGRAPHICS:
         raise KeyError(f"{item!r} not in {tuple(DEMOGRAPHICS)!r}")
      self.__setattr__(item, value)

   def _operation(self, other: Any, op: Callable[[float, float], float]) -> Population:
      population = Population()
      for demo in DEMOGRAPHICS:
         self_ = self[demo]
         other_ = other[demo] if isinstance(other, Population) else other
         if self_ is not None and other_ is not None:
            population[demo] = op(self_, other_)
      return population

   def __add__(self, other: Any) -> Population:
      return self._operation(other, lambda a, b: a + b)
   def __sub__(self, other: Any) -> Population:
      return self._operation(other, lambda a, b: a - b)
   def __mul__(self, other: Any) -> Population:
      return self._operation(other, lambda a, b: a * b)
   def __truediv__(self, other: Any) -> Population:
      return self._operation(other, lambda a, b: a / b if b != 0.0 else math.nan)
   def __round__(self) -> Population:
      return self._operation(0, _roundish)

   @staticmethod
   def zero() -> Population:
      return Population(
         total=0,
         asian=0,
         black=0,
         hispanic=0,
         native=0,
         not_specified=0,
         pacific_islander=0,
         two_or_more=0,
         white=0,
      )

   @property
   def color(self) -> Optional[float]:
      """Non-white students
      """
      total = self.total
      white = self.white
      if total is None and white is None: return None
      if white is None: return total
      assert total is not None
      return total - white

   def majority_demographics(self, threshold: float=0.4) -> list[str]:
      """Returns list of up to 3 demographics that make the majority of this
      population
      """
      demos = {demo: self[demo] or 0 for demo in DEMOGRAPHICS}
      if "total" in demos: del demos["total"]
      top_demos = sorted(demos, key=lambda d: demos[d], reverse=True)
      top_n = demos[top_demos[0]]
      threshold_n = round(top_n * threshold)
      top_demos = [d for d in top_demos if demos[d] > threshold_n]
      return top_demos[:3]

@dataclass
class TravelTimes:
   """Represents the travel times aspect of a simulation's results

   Attributes:
      status_quo: Original travel times for each demographic
      switcher_previous: Original travel times for each demographic, for those
         who switched schools after the simulation
      switcher_new: New travel times for each demographic, for those who
         switched schools after the simulation
   """
   status_quo: Population
   switcher_previous: Population
   switcher_new: Population
   __hash__ = lambda self: hash(id(self))

   def _operation(self, other: Any, op: Callable[[float, float], float]) -> TravelTimes:
      return TravelTimes(
         self.status_quo._operation(other, op),
         self.switcher_previous._operation(other, op),
         self.switcher_new._operation(other, op),
      )

   def __add__(self, other: Any) -> TravelTimes:
      return self._operation(other, lambda a, b: a + b)
   def __sub__(self, other: Any) -> TravelTimes:
      return self._operation(other, lambda a, b: a - b)
   def __mul__(self, other: Any) -> TravelTimes:
      return self._operation(other, lambda a, b: a * b)
   def __truediv__(self, other: Any) -> TravelTimes:
      return self._operation(other, lambda a, b: a / b if b != 0.0 else math.nan)
   def __round__(self) -> TravelTimes:
      return self._operation(0, _roundish)

@dataclass
class Analytics:
   """High-level results for a given simulation!

   Attributes:
      simulation: info about this particular simulation
      pre_dissimilarity: dissimilarity score before simulation
      post_dissimilarity: dissimilarity score after simulation
      all_population: demographics of entire simulation
      switched_population: demographics of those in this simulation who switch
         schools after the simulation
      travel_times: travel times results, totaled for each demographic, in
         seconds
      travel_times_per_individual: travel time results, per person for a given
         demographic, in seconds
   """
   simulation: Simulation
   # dissimilarity results
   pre_dissimilarity: float
   post_dissimilarity: float
   # population stats
   all_population: Population
   switched_population: Population
   # travel time stats
   travel_times: TravelTimes
   __hash__ = lambda self: hash(id(self))

   @staticmethod
   def from_simulation(simulation: Simulation) -> Analytics:
      """Eats the analytics file for a particular simulation"""
      df = load_df(simulation.analytics_filepath)
      df = df[df["state"] == simulation.state]
      df = df[df["district_id"] == simulation.district_id]
      #df = df[df["interdistrict"] == simulation.interdistrict]

      assert (num_rows := len(df.index)) == 1, (
         f"found {num_rows} rows for {simulation=!r} (expected 1 row)"
      )

      #this_simulation = Simulation(**simulation.__dict__)
      this_simulation = simulation
      this_simulation.school_decrease_threshold = df["school_decrease_threshold"].astype(str).item()

      return Analytics(
         this_simulation,
         df.iloc[0]["pre_dissim"].item(),
         df.iloc[0]["post_dissim"].item(),
         _digest_population(df, lambda demo: f"num_{demo}_all"),
         _digest_population(df, lambda demo: f"num_{demo}_switched"),
         _digest_travel_times(df),
      )

   @property
   def travel_times_per_individual(self) -> TravelTimes:
      return TravelTimes(
         self.travel_times.status_quo / self.all_population,
         self.travel_times.switcher_previous / self.switched_population,
         self.travel_times.switcher_new / self.switched_population,
      )

@st.cache_data(ttl=3600, show_spinner="Looking up the travel times data...")
def _digest_travel_times(df: pd.DataFrame) -> TravelTimes:
   status_quo = _digest_population(df, lambda demo: f"all_status_quo_time_num_{demo}")
   switcher_previous = _digest_population(df, lambda demo: f"switcher_status_quo_time_num_{demo}")
   switcher_new = _digest_population(df, lambda demo: f"switcher_new_time_num_{demo}")
   return TravelTimes(status_quo, switcher_previous, switcher_new)

def _digest_population(df: pd.DataFrame, column_name_formatter: Callable[[str], str]) -> Population:
   population = Population()
   for demo in DEMOGRAPHICS:
      column_name = column_name_formatter(demo)
      try:
         item = df[column_name].item()
         population[demo] = round(item) if not math.isnan(item) else math.nan
      except KeyError:
         pass
   return population

@dataclass
class School:
   """Represents a school

   Attributes:
      ncessch_id: NCES school ID
      school_name: name of school
      district_id: ID for the district this school is in
      grade_span_before: list of grades offered, before running the algorithm
      grade_span_after: list of grades offereed, after running the algorithm
      grades_population_before: mapping from grade to demographics, before
         algorithm
      grades_population_after: mapping from grade to demographics, after
         algorithm
      cluster_neighbors: neighboring schools in same cluster
      population_before: demographics before algorithm
      population_after: demographics after algorithm
      travel_times_previous: if this school is merged, *status quo* travel
         times, in seconds, *for students who will switch* (deprecated)
      travel_times_new: if this school is merged, new travel times, in seconds,
         for students who switched (deprecated)
      centroid: ``(lat, lng)`` of school attendance boundaries
      location: ``(lat, lng)`` of school itself
      switched_population: demographics of those in this school who switched
         to *new* schools after mergers
      travel_times: travel times results, totaled for each demographic, in
         seconds
      travel_times_per_individual: travel time results, per person for a given
         demographic, in seconds
   """
   ncessch_id: int
   school_name: str
   district_id: int
   #grade_span_after: list[str]
   grades_population_before: dict[str, Population]
   grades_population_after: dict[str, Population]
   population_after: Population
   cluster_neighbors: list[School]
   travel_times_previous: Optional[Population]
   travel_times_new: Optional[Population]
   switched_population: Optional[Population]
   centroid: None | tuple[float, float] = None
   location: None | tuple[float, float] = None
   __hash__ = lambda self: hash(id(self))

   def __post_init__(self) -> None:
      # automatically fetch centroid
      self.centroid = SCHOOL_ID_TO_CENTROID.get(self.nces_id)
      self.location = SCHOOL_ID_TO_LOCATION.get(self.nces_id)
      # compute travel times for this school
      self.travel_times: TravelTimes
      self.travel_times_per_individual: TravelTimes
      self.travel_times = TravelTimes(
         status_quo=None,
         switcher_previous=self.travel_times_previous,
         switcher_new=self.travel_times_new,
      )
      # pyright gets very mad around here, but.. it's fine...
      self.travel_times_per_individual = TravelTimes(
         status_quo=None,
         switcher_previous=(
            (self.travel_times_previous / self.switched_population)
            if (self.travel_times_previous is not None and self.switched_population is not None)
            else None
         ),
         switcher_new=(
            (self.travel_times_new / self.switched_population)
            if (self.travel_times_new is not None and self.switched_population is not None)
            else None
         ),
      )

   @property
   def nces_id(self) -> int:
      """alias for ncessch_id"""
      return self.ncessch_id

   @property
   def grade_span_before(self) -> list[str]:
      # it's implicit
      return [g for g in GRADES if self.grades_population_before[g].total > 0]

   @property
   def grade_span_after(self) -> list[str]:
      return [g for g in GRADES if self.grades_population_after[g].total > 0]

   @property
   def population_before(self) -> Population:
      return sum(self.grades_population_before.values(), start=Population.zero())

   def summarize_grade_span(self, *, when: str="after") -> pd.DataFrame:
      """Returns a Pandas DataFrame for what grades this school offers

      Args:
         when: "before" or "after"
      """
      assert when in ("before", "after")
      grades_offered = self.grade_span_after if when == "after" else self.grade_span_before
      data: dict[str, Any] = {"School": [self.school_name]}
      for grade in GRADES: data[grade] = [bool(grade in grades_offered)]
      return pd.DataFrame(data).reset_index(drop=True)

def _double_check_sets(expected: Collection[Any], seen: Collection[Any], step: str):
   """Just crossing ts and dotting is
   """
   if len(expected) > len(seen):
      msg = f"[{step}] some schools seem to be missing?"
      msg += f" ({len(expected)=} vs. {len(seen)=})"
      missing = set(expected) - set(seen)
      if len(missing) < 10: msg += f" ({missing=!r})"
      warnings.warn(msg)
   elif len(expected) < len(seen):
      msg = f"[{step}] we seem to have extra schools than expected?"
      msg += f" ({len(expected)=} vs. {len(seen)=})"
      extra = set(seen) - set(expected)
      if len(extra) < 10: msg += f" ({extra=!r})"
      warnings.warn(msg)

@st.cache_data(ttl=3600, show_spinner="Looking up schools for this district...")
def _digest_schools(simulation: Simulation) -> dict[int, School]:
   """Reads the CSVs and returns a mapping from school ID to School object

   Note:
      This function assumes that each simulation has its own
      ``grades_served.csv``, ``school_mergers.csv``, ``schools_in_play.csv``,
      and ``students_per_group_per_school_post_merger.json`` set of files.
   """
   district_id = simulation.district_id

   # grade_span_after
   # this file isn't actually necessary to open?
   grade_span_after = {}
   df_grades = load_df(simulation.grades_offered_after_filepath)
   for index, row in df_grades.iterrows():
      ncessch_id = row["NCESSCH"]
      #if SCHOOL_ID_TO_DISTRICT_ID[ncessch_id] != district_id: continue
      if ncessch_id in grade_span_after:
         warnings.warn(f"duplicate school entry for {ncessch_id=} in {str(simulation.grades_offered_after_filepath)!r}?")
      if ncessch_id not in grade_span_after:
         grade_span_after[ncessch_id] = []
      for grade in GRADES:
         if row[grade]:
            grade_span_after[ncessch_id].append(grade)

   # grades_population_before
   grades_population_before = {}
   df_demos_before = load_df(simulation.demos_before_filepath)
   for index, row in df_demos_before.iterrows():
      ncessch_id = int(row["NCESSCH"].item())
      #if SCHOOL_ID_TO_DISTRICT_ID[ncessch_id] != district_id: continue
      if ncessch_id in grades_population_before:
         warnings.warn(f"duplicate school entry for {ncessch_id=} in {str(simulation.demos_before_filepath)!r}?")
      grades_population_before[ncessch_id] = {}
      for grade in GRADES:
         population = Population.zero()
         for demo in DEMOGRAPHICS:
            population[demo] = row[f"num_{demo}_{grade}"].item()
         grades_population_before[ncessch_id][grade] = population

   _double_check_sets(
      grade_span_after,
      grades_population_before,
      "_digest_schools > grades_population_before"
   )

   # grades_population_after
   grades_population_after = {}
   with open(simulation.demos_x_grades_after_filepath) as f:
      demos_x_grades_after = json.load(f)
   for s_id in grades_population_before:
      grades_population_after[s_id] = {}
      for grade in GRADES:
         counts = demos_x_grades_after[f"{s_id:012d}"]
         population = Population.zero()
         for demo in DEMOGRAPHICS:
            population[demo] = counts[f"num_{demo}"].get(grade, 0)
         grades_population_after[s_id][grade] = population

   _double_check_sets(
      grades_population_before,
      grades_population_after,
      "_digest_schools > grades_population_after"
   )

   # population_after
   population_after = {}
   with open(simulation.demos_after_filepath) as f:
      demos_after = json.load(f)
   for demo in DEMOGRAPHICS:
      for ncessch_id_str, count in demos_after[f"num_{demo}"].items():
         ncessch_id = int(ncessch_id_str)
         #if SCHOOL_ID_TO_DISTRICT_ID[ncessch_id] != district_id: continue
         if ncessch_id not in population_after:
            population_after[ncessch_id] = Population.zero()
         population_after[ncessch_id][demo] += count

   _double_check_sets(
      grades_population_before,
      population_after,
      "_digest_schools > population_after"
   )

   # travel times per school per demo
   travel_times_previous: dict[int, Optional[Population]] = {s_id: None for s_id in grade_span_after}
   travel_times_new: dict[int, Optional[Population]] = {s_id: None for s_id in grade_span_after}
   switched_population: dict[int, Optional[Population]] = {s_id: None for s_id in grade_span_after}
   status_quo_file = simulation.data_path / "status_quo_total_driving_times_for_switchers_per_school_per_cat.json"
   new_file = simulation.data_path / "new_total_driving_times_for_switchers_per_school_per_cat.json"
   switchers_file = simulation.data_path / "students_switching_per_group_per_school.json"
   with open(status_quo_file, "r") as f: status_quo_j = json.loads(f.read())
   with open(new_file, "r") as f: new_j = json.loads(f.read())
   with open(switchers_file, "r") as f: switch_j = json.loads(f.read())
   for ncessch_id in grade_span_after:
      ncessch_id_str = f"{ncessch_id:012d}"
      if ncessch_id_str in status_quo_j or ncessch_id_str in new_j:
         error = []
         if ncessch_id_str not in status_quo_j: error.append(f"{status_quo_file.name!r}")
         if ncessch_id_str not in new_j: error.append(f"{new_file.name!r}")
         if ncessch_id_str not in switch_j: error.append(f"{switchers_file.name!r}")
         if error:
            msg = f"{ncessch_id_str!r} not found in " + " nor ".join(error) + f" ({str(status_quo_file.parent)!r})"
            #raise AssertionError(msg)
            print(f"warning: {msg}")
         ncessch_id = int(ncessch_id_str)
         try:
            travel_times_previous[ncessch_id] = Population(
               total=status_quo_j[ncessch_id_str]["switcher_status_quo_time_num_total"],
               asian=status_quo_j[ncessch_id_str]["switcher_status_quo_time_num_asian"],
               black=status_quo_j[ncessch_id_str]["switcher_status_quo_time_num_black"],
               hispanic=status_quo_j[ncessch_id_str]["switcher_status_quo_time_num_hispanic"],
               native=status_quo_j[ncessch_id_str]["switcher_status_quo_time_num_native"],
               white=status_quo_j[ncessch_id_str]["switcher_status_quo_time_num_white"],
            )
            travel_times_new[ncessch_id] = Population(
               total=new_j[ncessch_id_str]["switcher_new_time_num_total"],
               asian=new_j[ncessch_id_str]["switcher_new_time_num_asian"],
               black=new_j[ncessch_id_str]["switcher_new_time_num_black"],
               hispanic=new_j[ncessch_id_str]["switcher_new_time_num_hispanic"],
               native=new_j[ncessch_id_str]["switcher_new_time_num_native"],
               white=new_j[ncessch_id_str]["switcher_new_time_num_white"],
            )
            switched_population[ncessch_id] = Population(
               total=switch_j[ncessch_id_str]["num_total_switched"],
               asian=switch_j[ncessch_id_str]["num_asian_switched"],
               black=switch_j[ncessch_id_str]["num_black_switched"],
               hispanic=switch_j[ncessch_id_str]["num_hispanic_switched"],
               native=switch_j[ncessch_id_str]["num_native_switched"],
               white=switch_j[ncessch_id_str]["num_white_switched"],
            )

         except KeyError:
            # REMOVE LATER
            pass

   # wrapping up
   schools = {}
   for ncessch_id in grade_span_after:
      district_id = SCHOOL_ID_TO_DISTRICT_ID[ncessch_id]
      school_name = SCHOOLS_IN_DISTRICT[district_id][ncessch_id]
      try:
         schools[ncessch_id] = School(
            ncessch_id,
            school_name,
            district_id,
            #grade_span_after[ncessch_id],
            grades_population_before[ncessch_id],
            grades_population_after[ncessch_id],
            population_after[ncessch_id],
            [], # District.from_simulation will fill this in later
            travel_times_previous[ncessch_id],
            travel_times_new[ncessch_id],
            switched_population[ncessch_id],
         )
      except KeyError:
         # THIS WAS ADDED DUE TO THE `status_quo_j` VS `new_j` AssertionError above
         # THIS SHOULD EVENTUALLY BE REMOVED!
         pass

   return schools

@dataclass
class Impact:
   """Represents per-school impacts on segregation, for a district

   Attributes:
      district: The district these results refer to
      district_concentration: relative frequency of demographics (instead of
         absolute counts) (at the district level)
      school_concentrations_pre: relative frequency of demographics, before
         mergers (at the school level)
      school_concentrations_post: relative frequency of demographics, after
         mergers (at the school level)
      overconcentrated_schools_pre: schools for a specific demographic that
         have concentrations of these students *above* the district's
         concentration, before mergers
      overconcentrated_schools_post: schools for a specific demographic that
         have concentrations of these students *above* the district's
         concentration, after mergers
      average_overconcentrated_concentration_before: of these overconcentrated
         schools, the average concentration (at the school level) of
         demographics, before mergers
      average_overconcentrated_concentration_after: of these same
         overconcentrated schools, the average concentration (at the school
         level) of demographics, after mergers
      focal_demos: most highly clustered demographics
      greatest_changing_schools: for each demographic, a list of schools
         **sorted by extent of impact** (defined as greatest change in
         school concentration)
   """
   district: District

   def __post_init__(self):
      district = self.district
      self.district_concentration = district.population / district.population.total
      all_schools = list(district.schools.values())
      self.school_concentrations_pre = {
         school: school.population_before / school.population_before.total
         for school in all_schools
      }
      #print(f"B: {pformat(self.school_concentrations_pre) = !s}")
      self.school_concentrations_post = {
         school: school.population_after / school.population_after.total
         for school in all_schools
      }
      self.overconcentrated_schools_pre: dict[DemoType, list[School]] = {}
      self.overconcentrated_schools_post: dict[DemoType, list[School]] = {}
      self.average_overconcentrated_concentration_before: dict[DemoType, Population] = {}
      self.average_overconcentrated_concentration_after: dict[DemoType, Population] = {}
      demo: DemoType
      for demo in _demo_type:
         # which schools are overconcentrated
         self.overconcentrated_schools_pre[demo] = [
            school
            for school, concentration in self.school_concentrations_pre.items()
            if concentration[demo] > self.district_concentration[demo]
         ]
         self.overconcentrated_schools_post[demo] = [
            school
            for school, concentration in self.school_concentrations_post.items()
            if concentration[demo] > self.district_concentration[demo]
         ]
         # average concentrations of the original overconcentrated schools
         self.average_overconcentrated_concentration_before[demo] = sum([
            school.population_before
            for school in self.overconcentrated_schools_pre[demo]
         ], start=Population.zero())
         self.average_overconcentrated_concentration_before[demo] /= \
            self.average_overconcentrated_concentration_before[demo].total
         self.average_overconcentrated_concentration_after[demo] = sum([
            school.population_after
            for school in self.overconcentrated_schools_pre[demo]
         ], start=Population.zero())
         self.average_overconcentrated_concentration_after[demo] /= \
         self.average_overconcentrated_concentration_after[demo].total
      # ---
      self.greatest_changing_schools: dict[DemoType, list[School]] = {}
      for demo in _demo_type:
         self.greatest_changing_schools[demo] = sorted(
            #all_schools,
            self.overconcentrated_schools_pre[demo],
            key=lambda school: abs(self.school_concentrations_pre[school][demo] - self.school_concentrations_post[school][demo]),
            reverse=True,
         )
      # ---
      self.focal_demos = sorted(
         _demo_type,
         key=lambda demo: (
            len(self.overconcentrated_schools_pre[demo]),
            len(self.greatest_changing_schools[demo]),
            abs(self.district_concentration[demo] - self.average_overconcentrated_concentration_before[demo][demo]),  # tie-breaker
         ),
         reverse=True,
      )

   @property
   def district_id(self) -> int: return self.district.nces_id

   @property
   def focal_demo(self) -> DemoType: return self.focal_demos[0]

@dataclass
class District:
   """Represents a district

   Attributes:
      simulation: info about this particular simulation
      clusters: mapping for all clusters in this district
      clusters_in_simulation: mapping for all clusters in the respective
         simulation for this district, including clusters entirely in
         neighboring districts
      clusters_from_neighboring_districts: all clusters from neighboring
         districts
      schools: all schools in this district, mapping from NCES ID to `School`
         object
      schools_in_simulation: all schools in the respective simulation for this
         district, including neighboring schools
      schools_from_neighboring_districts: all schools from neighboring
         districts, useful for interdistrict simulations
      analytics: results of this simulation for this district
      nces_id: NCES ID of this district
      state: abbreviation of district's state (e.g. "NC")
      population: population distribution of district
      closed_enrollment_only: whether all elementary schools in this district
         are listed as closed-enrollment
      centroid: ``(lat, lng)``
      impact: statistics about the per-school/-cluster impact of mergers in
         this district
   """
   clusters_in_simulation: dict[str, list[School]]
   schools_in_simulation: dict[int, School]
   analytics: Analytics
   centroid: None | tuple[float, float] = None
   __hash__ = lambda self: hash(id(self))

   def __post_init__(self):
      # the following must be explicitly calculated for the interdistrict
      # simulations
      self.schools = {
         s_id: school for s_id, school in self.schools_in_simulation.items()
         if SCHOOL_ID_TO_DISTRICT_ID[school.ncessch_id] == self.nces_id
      }
      self.schools_from_neighboring_districts = {
         s_id: school for s_id, school in self.schools_in_simulation.items()
         if s_id not in self.schools
      }
      self.clusters = {
         name: cluster for name, cluster in self.clusters_in_simulation.items()
         if any(s.district_id == self.nces_id for s in cluster)
      }
      self.clusters_from_neighboring_districts = {
         name: cluster for name, cluster in self.clusters_in_simulation.items()
         if name not in self.clusters
      }
      # automatically fetch centroid
      self.centroid = DISTRICT_ID_TO_CENTROID.get(self.nces_id)
      # handle district population information
      self.population = self.analytics.all_population
      if self.analytics.simulation.interdistrict:
         self.population = sum([s.population_before for s in self.schools.values()], start=Population.zero())
      # ---
      self.impact = Impact(self)
      # COARSE SORT:
      def _sort_cluster_key(cluster: list[School], ascending=False) -> tuple[float, float]:
         """
         Notes:
            * show overconcentrated schools first, particularly those that make
              the most dramatic change in students of color concentration
            * then, for non-overconcentrated schoools: sort clusters by
              schools that end up closest to the district ratios
            * (ultimate tie-breaker is just use alphabetical order)
         """
         def overconcentrated_change(school: School) -> int:
            if school not in self.impact.overconcentrated_schools_pre["color"]: return 0
            return abs(self.impact.school_concentrations_pre[school].color - self.impact.school_concentrations_post[school].color)
         # ---
         def progress_score(school: School) -> float:
            """schools that end up closer to the district ratios than they began
            """
            try:
               return abs(self.impact.school_concentrations_pre[school].color - self.impact.district_concentration.color) - abs(self.impact.school_concentrations_post[school].color - self.impact.district_concentration.color)
            except:
               return 0
         # ---
         a = sum(overconcentrated_change(school) for school in cluster)
         b = sum(progress_score(school) for school in cluster)
         if ascending:
            a *= -1
            b *= -1
         return (a, b)
      sorted_clusters = sorted(
         self.clusters.items(),
         key=lambda item: (*_sort_cluster_key(item[1], ascending=True), item[0]),
         #reverse=True,
      )
      # FINAL SORT: organize clusters by: pairs, then triplets, then singletons
      sorted_clusters = sorted(
         sorted_clusters,
         key=lambda item: (len(item[1]) == 2, len(item[1]) == 3),
         reverse=True,
      )
      self.clusters = {k: v for k, v in sorted_clusters}

#
# end temporary zone, do not touch
#

   @property
   def simulation(self) -> Simulation: return self.analytics.simulation

   @property
   def nces_id(self) -> int:
      return self.simulation.district_id
   @property
   def state(self) -> str:
      return DISTRICT_ID_TO_STATE[self.nces_id]
   @property
   def name(self) -> str:
      return DISTRICTS_IN_STATE[self.state][self.nces_id]
   @property
   def proportion_switched(self) -> float:
      return (
         (self.analytics.switched_population.total or 0)
         / (self.analytics.all_population.total or 1)
      )

   @staticmethod
   def from_simulation(simulation: Simulation) -> District:
      analytics = Analytics.from_simulation(simulation)
      schools_expected = SCHOOLS_IN_DISTRICT[simulation.district_id]

      school_objects = _digest_schools(simulation)
      # _double_check_sets(
      #    schools_expected,
      #    school_objects,
      #    "District.from_simulation > school_objects"
      # )

      clusters = {}
      schools_seen = set()
      df_clusters = load_df(simulation.clusters_filepath)
      for _, line in df_clusters.itertuples():
         line = str(line)
         school_ids = [int(s) for s in line.split(", ")]
         schools_seen |= set(school_ids)
         cluster = [school_objects[s] for s in school_ids]
         for school in cluster:
            school.cluster_neighbors = [
               school_ for school_ in cluster
               if not (school_ is school)
            ]
         cluster_name = "; ".join(s.school_name for s in cluster)
         if len(cluster) == 2:
            cluster_name = f"Pair: ({cluster_name})"
         elif len(cluster) == 3:
            cluster_name = f"Triplet: ({cluster_name})"
         elif len(cluster) == 1:
            cluster_name = f"Unchanged: ({cluster_name})"
         clusters[cluster_name] = cluster

      _double_check_sets(
         school_objects,
         schools_seen,
         "District.from_simulation > clusters"
      )

      return District(
         clusters,
         school_objects,
         analytics,
      )

   @property
   def closed_enrollment_only(self) -> bool:
      return self.nces_id in CLOSED_ENROLLMENT_ONLY_DISTRICTS

# --- names to be used by other modules

analytics = st.cache_data(ttl=3600)(Analytics.from_simulation)
district = District.from_simulation

results = district
context = Simulation

# --- code for testing

def test_analytics():
   import pprint
   pp = pprint.PrettyPrinter(indent=3)
   s = Simulation("NC", 3701500, False)
   a = analytics(s)
   pp.pprint(a)
   return a

def test_district():
   import pprint
   pp = pprint.PrettyPrinter(indent=3)
   s = Simulation("NC", 3701500, False)
   d = district(s)
   #pp.pprint(d)
   return d

if __name__ == "__main__":
   import pprint
   #a = test_analytics()
   d = test_district()
   for s_id in d.schools: break
   for cluster in d.clusters: break
