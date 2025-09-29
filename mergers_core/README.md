# mergers core

This folder contains the code that sets up and solves the school mergers constraint programming problems.

Contact Nabeel Gillani (`n.gillani` at northeastern dot edu) or Madison Landry
(`landry.ma` at northeastern dot edu) for help.

## Code

When running this code, it's a good idea to have a venv active:
```sh
$ python -m venv .venv  # you only run this once
$ pip install -r requirements.txt  # also only run this once
$ source .venv/bin/activate  # run this on every new shell to reactivate the venv (fish users use activate.fish instead of activate)
```
To deactivate, run `deactivate` anywhere.

Running the code on a Slurm cluster is handled by `dispatch.sh <batch> [<configs_file> ...]`.

## `./models/`

* `constants.py` — Definitions shared by other code
* `merge_cp_sat.py`
   * Set up and solve constraint problems
   * Output results
   * Nominal function: `solve_and_output_results()`
* `model_utils.py`
   * Verify results (e.g., check where constraints were maintained)
   * Estimating impacts on travel time
   * Estimating impacts on segregation via dissimilarity score
   * Generating analytics CSVs after solving
* `simulation_sweeps.py` — Organize simulations into batches (for Slurm)
   * To generate, run `python -c 'import models.config as c; c.generate_all_configs()'`

## `./utils/`

* `compute_travel_times.py` — Computations related to school travel times (generates `data/travel_times_files`)
* `distances_and_times.py` — Computations related to school distances and travel times
* `header.py` — Definitions shared by other code
* `output_block_estimates.py` — Estimate demographics per census block
* `produce_files_for_solver.py`
   * Produce files used by the solver (e.g., `data/solver_files`, centroids/adjacency data)
   * Determine maximum capacities of schools based on history
   * Output school demographic x grade data
   * Compute district adjacency
   * Compile space of possible school mergers for a district simulation
* `split_shapedata_by_district.py` — Interpret shape files into GeoPandas CSV per district

## `./analysis/`

* `analyze_districts.py`
   * Compute district demographic x grade data
   * Compute dissimilarity scores for opt out sensitivity analysis
* `analyze_results.py` — Some functions for auxiliary analysis

## `./data/`

The files under this folder are those necessary for the solver and/or generated from by the `mergers_core` code. Available for download here: [`school_mergers_data.zip`](https://plural-connections.s3.us-east-1.amazonaws.com/school-mergers/school_mergers_data.zip)

* `state_codes.csv` — Abbreviations and FIPS codes of the US states, DC, and PR
* `attendance_boundaries/` — School attendance boundaries data
* `census_block_shapefiles_2020/` — Census block to school attendance mappings (from [Gillani &al., 2023](https://doi.org/10.3102/0013189X231170858))
* `misc/` — Miscellaneous results files
* `school_data/` — Miscellaneous school data files
* `school_district_2021_boundaries/` — Shape files and centroid/adjacency calculations for analysis and the solver
* `solver_files/` — Compiled data used by `merge_cp_sat.py`
* `sweep_configs/` — Chunk configurations from/for `simulation_sweeps.py`
* `travel_times_files/` — Travel times matrices used by `model_utils.py`
