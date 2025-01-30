from __future__ import annotations
from collections import Counter, defaultdict
from typing import Callable, Iterable, Optional
from enum import Enum

from pathlib import Path
from statistics import median, mean, stdev
from math import isnan, isinf, nan, inf
from colorsys import hls_to_rgb

import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots
import numpy as np
import scipy.stats
import pandas as pd
from tqdm import tqdm
import streamlit as st

# cursed relative imports, do not touch
from headers import (
   DATA_ROOT,
   DEMOGRAPHICS,
   DEMOGRAPHIC_LABELS,
   DEMOS_X_GRADES,
   GRADES,
   DISTRICTS_IN_STATE,
   DISTRICT_ID_TO_STATE,
)
import eat
from eat import District, Simulation
# cursed relative imports, do not touch

def fix_font(fig: Figure, pad: int=10) -> Figure:
   fig.update_layout(
      font={
         "family": "Noto Sans",
         "size": 18,
      },
      margin={"pad": pad}
   )
   return fig

DATA_ROOT = Path(__file__).parent.parent / DATA_ROOT
FIGURE_OUTPUT_PATH = Path(__file__).parent.parent / "figure_output"

class SimulationKind(Enum):
   CONSTRAINED = "constrained_0.2"
   CONSTRAINED_BH_WA = "constrained_bh_wa_0.2"
   BOTTOMLESS = "bottomless_1.0"
   SENSITIVITY_0_1 = "bottom_sensitivity_0.1"
   SENSITIVITY_0_3 = "bottom_sensitivity_0.3"

_thresholds = {
   SimulationKind.CONSTRAINED: "0.2",
   SimulationKind.CONSTRAINED_BH_WA: "0.2",
   SimulationKind.BOTTOMLESS: "1.0",
   SimulationKind.SENSITIVITY_0_1: "0.1",
   SimulationKind.SENSITIVITY_0_3: "0.3",
}

_folder_mapper = {
   SimulationKind.CONSTRAINED: "constrained",
   SimulationKind.CONSTRAINED_BH_WA: "constrained_bh_wa",
   SimulationKind.BOTTOMLESS: "bottomless",
   SimulationKind.SENSITIVITY_0_1: "bottom_sensitivity",
   SimulationKind.SENSITIVITY_0_3: "bottom_sensitivity",
}

_simulation_names = {
   SimulationKind.CONSTRAINED: "Minimum 80%",
   SimulationKind.CONSTRAINED_BH_WA: "Minimum 80% (BH-WA)",
   SimulationKind.BOTTOMLESS: "No minimum",
   SimulationKind.SENSITIVITY_0_1: "Minimum 90%",
   SimulationKind.SENSITIVITY_0_3: "Minimum 70%",
}

def eat_context(state: str, district_id: int, kind: SimulationKind, interdistrict: bool=False) -> Simulation:
   school_decrease_threshold = _thresholds[kind]
   return eat.context(state, district_id, interdistrict, school_decrease_threshold, folder_name_check=_folder_mapper[kind])

def _get_extra_infos(kind: SimulationKind, interdistrict: bool=False) -> pd.DataFrame:
   if interdistrict:
      filepath = DATA_ROOT / "min_num_elem_schools_4_interdistrict" / f"consolidated_simulation_results_min_num_elem_schools_4_interdistrict_{_thresholds[kind]}_{interdistrict}.csv"
   else:
      filepath = DATA_ROOT / f"min_num_elem_schools_4_{_folder_mapper[kind]}" / f"consolidated_simulation_results_min_num_elem_school_4_{_folder_mapper[kind]}_{_thresholds[kind]}_{interdistrict}.csv"
   if kind in (SimulationKind.CONSTRAINED_BH_WA,):
      filepath = DATA_ROOT / filepath.parts[-2].replace("elem_schools_", "elem_") / filepath.parts[-1].replace("elem_school_", "elem_")
   df = pd.read_csv(filepath)
   pop_df = df[[
      "district_id",
      "num_white_all",
      "num_black_all",
      "num_hispanic_all",
      "num_native_all",
      "num_asian_all",
      "num_total_all",
      "num_white_switched",
      "num_black_switched",
      "num_hispanic_switched",
      "num_native_switched",
      "num_asian_switched",
      "num_total_switched",
   ]]
   times_df = df[[
      "district_id",
      "all_status_quo_time_num_white",
      "all_status_quo_time_num_black",
      "all_status_quo_time_num_asian",
      "all_status_quo_time_num_native",
      "all_status_quo_time_num_hispanic",
      "all_status_quo_time_num_total",
      "switcher_status_quo_time_num_white",
      "switcher_status_quo_time_num_black",
      "switcher_status_quo_time_num_asian",
      "switcher_status_quo_time_num_native",
      "switcher_status_quo_time_num_hispanic",
      "switcher_status_quo_time_num_total",
      "switcher_new_time_num_white",
      "switcher_new_time_num_black",
      "switcher_new_time_num_asian",
      "switcher_new_time_num_native",
      "switcher_new_time_num_hispanic",
      "switcher_new_time_num_total",
   ]]
   for demo in DEMOGRAPHICS:
      times_df[f"all_status_quo_time_num_{demo}"] /= 60 * pop_df[f"num_{demo}_all"]
      times_df[f"switcher_status_quo_time_num_{demo}"] /= 60 * pop_df[f"num_{demo}_switched"]
      times_df[f"switcher_new_time_num_{demo}"] /= 60 * pop_df[f"num_{demo}_switched"]
   times_df = times_df.dropna()
   return times_df

@st.cache_data()
def top_200_districts_df(kind: SimulationKind, interdistrict: bool=False, *, extra_info: bool=False) -> pd.DataFrame:
   #df = pd.read_csv(DATA_ROOT / ".." / "old_figs" / "figures" / "top_200_by_population.csv")
   determiner = kind.value + ("_interdistrict" if interdistrict else "")
   df = pd.read_csv(FIGURE_OUTPUT_PATH / f"results_top_200_by_population_{determiner}.csv")
   if extra_info:
      extra_df = _get_extra_infos(kind, interdistrict)
      df = df.merge(extra_df, on="district_id")
   return df

@st.cache_data()
def top_200_districts(kind: SimulationKind, interdistrict: bool=False) -> list[int]:
   df = top_200_districts_df(kind, interdistrict)
   return list(df["district_id"].astype(int))
