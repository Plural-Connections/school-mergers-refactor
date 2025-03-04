# public-school-mergers

Public repo containing code and data release for the [School Mergers paper](https://academic.oup.com/pnasnexus/article/4/3/pgaf050/8046446)

[The 74: School 'Pairings' Can Foster Racial, Socioeconomic Integration](https://www.the74million.org/article/new-research-school-pairings-can-foster-racial-socioeconomic-integration/)

## Pipeline

Check the READMEs of the respective folders for more detailed information.

1. [`mergers_core/`](./mergers_core) — Set up and run CP-SAT simulations
2. [`dashboard/`](./dashboard) — Summarize, plot, & map high-level impacts

The code for the [School Mergers Dashboard](https://mergers.schooldiversity.org/) is under [`dashboard/`](./dashboard).

## Necessary data

This code repo operates on a mass of CSV and shapes files which are available here: [`school_mergers_data.zip`](https://plural-connections.s3.us-east-1.amazonaws.com/school-mergers/school_mergers_data.zip).

<details>
<summary><code>mergers_core/data/</code></summary>

* `mergers_core/data/state_codes.csv` — Abbreviations and FIPS codes of the US states, DC, and PR
* `mergers_core/data/attendance_boundaries/` — School attendance boundaries data
* `mergers_core/data/census_block_shapefiles_2020/` — Census block to school attendance mappings (from [Gillani &al., 2023](https://doi.org/10.3102/0013189X231170858))
* `mergers_core/data/misc/` — Miscellaneous results files
* `mergers_core/data/school_data/` — Miscellaneous school data files
* `mergers_core/data/school_district_2021_boundaries/` — Shape files and centroid/adjacency calculations for analysis and the solver
* `mergers_core/data/solver_files/` — Compiled data used by the solver
* `mergers_core/data/sweep_configs/` — Chunk configurations from/for `mergers_core/simulation_sweeps.py`
* `mergers_core/data/travel_times_files/` — Travel times matrices used by the solver

</details>

<details>
<summary><code>dashboard/data/</code></summary>

* `dashboard/data/all_schools_with_names.csv` — Demographic x grade counts for schools, along with school/district names (symlinked from `mergers_core`)
* `dashboard/data/entirely_elem_closed_enrollment_districts.csv` — Districts not including elementary schools that permit out-of-boundary attendance (e.g., magnet programs) (symlinked from `mergers_core`)
* `dashboard/data/state_codes.csv` (symlinked from `mergers_core`)
* `dashboard/data/min_num_elem_schools_4_bottomless/` — CP-SAT solver results (school closures allowed) (symlinked from `./results`)
* `dashboard/data/min_num_elem_schools_4_constrained/` — Primary CP-SAT solver results of the paper (80% minimum school enrollment). See `data/results/` for the others (symlinked from `./results`)
* `dashboard/data/results/` — All results (including sensitivity analyses) from the paper.
* `dashboard/data/school_attendance_boundaries/` — School attendance zone geometries for the 2021/2022 school year (symlinked from `mergers_core`)

</details>
