# mergers core

Contact Nabeel Gillani (`n.gillani` at northeastern dot edu) or Madison Landry
(`landry.ma` at northeastern dot edu) for help.

## Code

## `models/`

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

## `utils/`

* `compute_travel_times.py` — Computations related to school travel times
* `distances_and_times.py` — Computations related to school distances and travel times
* `header.py` — Definitions shared by other code
* `output_block_estimates.py` — Estimate demographics per census block
* `produce_files_for_solver.py`
   * Produce files used by `models/merge_cp_sat.py`
   * Determine maximum capacities of schools based on history
   * Output school demographic x grade data
   * Compute district adjacency
   * Compile space of possible school mergers for a district simulation
* `split_shapedata_by_district.py` — Interpret shape files into GeoPandas CSV per district

## `analysis/`

* `analyze_districts.py`
   * Compute district demographic x grade data
   * Compute dissimilarity scores for opt out sensitivity analysis
* `analyze_results.py` — Some functions for auxiliary analysis
