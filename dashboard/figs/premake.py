#!/usr/bin/env python3

from .headers import *
from .headers import _simulation_names, _thresholds, _folder_mapper, pd

import traceback
import shutil

"""
Run this before plotting.  It creates the top 200 results CSVs, which some of
the plotting modules use.
"""

THIS_MANY_DISTRICTS = 200

global ALL_DISTRICTS
ALL_DISTRICTS: dict[tuple[SimulationKind, bool], dict[int, District]] = {}


def _all_districts(
    kind: SimulationKind, interdistrict: bool, *, closed_enrollment_only: bool = True
) -> dict[int, District]:
    """Loads all district results!

    Only returns *valid* simulation results

    Note: Takes a while
    """
    print(f"info: setting oven temperature to {_thresholds[kind]}...")
    these_districts = set(DISTRICT_ID_TO_STATE)
    if closed_enrollment_only:
        df = pd.read_csv(DATA_ROOT / "entirely_elem_closed_enrollment_districts.csv")
        closed_enrollment_districts = set(df["district_id"])
        # print(f"checksum: {len(only_these_districts - old)} versus {old - only_these_districts = }")
        only_these_districts = closed_enrollment_districts & these_districts
    else:
        only_these_districts = these_districts
    print(f"info: ingredients: {len(only_these_districts)} districts...")
    all_districts = {}
    for district_id in tqdm(only_these_districts):
        state = DISTRICT_ID_TO_STATE[district_id]
        context = eat.context(
            state,
            district_id,
            interdistrict,
            _thresholds[kind],
            folder_name_check=_folder_mapper[kind],
        )
        try:
            district = eat.district(context)
            if (
                district.analytics.pre_dissimilarity == 0.0
                or isnan(district.analytics.pre_dissimilarity)
                or district.population.total in (0.0, None)
                or isnan(district.population.total)
            ):
                continue
        except AssertionError as e:
            print(f"critical! please resolve this AssertionError: {e}")
            traceback.print_exception(e.__class__, e, e.__traceback__)
            print(f"Have a nice day!")
            print(f"---")
            continue
        all_districts[district_id] = district
    return all_districts


def _predigest_bh_wa():
    """Change the "``dissim_bh_wa``" columns to "``dissim``" of the bh_wa
    consolidated CSV if not done so yet.
    """
    filename = (
        DATA_ROOT
        / "min_num_elem_4_constrained_bh_wa"
        / "consolidated_simulation_results_min_num_elem_4_constrained_bh_wa_0.2_False.csv"
    )
    df = pd.read_csv(filename)
    if "pre_dissim_bh_wa" not in df:
        return
    print(f"warning: predigesting {str(filename)!r}")
    shutil.copy(filename, filename.parent / "consolidated_original.csv")
    df.drop(labels=["pre_dissim", "post_dissim"], axis="columns", inplace=True)
    df.rename(
        columns={"pre_dissim_bh_wa": "pre_dissim", "post_dissim_bh_wa": "post_dissim"},
        inplace=True,
    )
    df.to_csv(filename, index=False)


def _generate_top_n_districts(
    kind: SimulationKind,
    interdistrict: bool,
    n: int,
    key: Callable[[District], float],
    output: Path,
    closed_enrollment_only: bool = True,
):
    """Generates a CSV of districts and their results sorted by the given key

    Args:
       n: max number of districts to include
       key: way to sort the list of District objects (note: reverse=True)
       output: output CSV filepath
    """
    if kind in (SimulationKind.CONSTRAINED_BH_WA,):
        _predigest_bh_wa()
    global ALL_DISTRICTS
    if (kind, interdistrict) not in ALL_DISTRICTS:
        ALL_DISTRICTS[kind, interdistrict] = _all_districts(
            kind, interdistrict, closed_enrollment_only=closed_enrollment_only
        )
    all_districts_sorted = sorted(
        ALL_DISTRICTS[kind, interdistrict].values(), key=key, reverse=True
    )
    top_n_districts = all_districts_sorted[:n]
    data = {
        "district_id": [f"{d.nces_id:07d}" for d in top_n_districts],
        "population": [d.population.total for d in top_n_districts],
        "pre_dissim": [d.analytics.pre_dissimilarity for d in top_n_districts],
        "post_dissim": [d.analytics.post_dissimilarity for d in top_n_districts],
        "dissim_change": [
            (d.analytics.post_dissimilarity - d.analytics.pre_dissimilarity)
            / d.analytics.pre_dissimilarity
            for d in top_n_districts
        ],
        "pre_times": [
            d.analytics.travel_times_per_individual.switcher_previous.total / 60.0
            for d in top_n_districts
        ],
        "post_times": [
            d.analytics.travel_times_per_individual.switcher_new.total / 60.0
            for d in top_n_districts
        ],
        "times_change": [
            (
                d.analytics.travel_times_per_individual.switcher_new.total
                - d.analytics.travel_times_per_individual.switcher_previous.total
            )
            / 60.0
            for d in top_n_districts
        ],
    }
    df = pd.DataFrame(data)
    output.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(output, encoding="utf-8", index=False)


def generate_top_n_districts_by_population(
    kind: SimulationKind,
    interdistrict: bool = False,
    *,
    n: int = 200,
    closed_enrollment_only: bool = True,
):
    """Generates a CSV of districts and their results sorted by population size"""
    determiner = kind.value + ("_interdistrict" if interdistrict else "")
    print(f"info: Baking a {kind.name} ({interdistrict=}) pie...")
    output = FIGURE_OUTPUT_PATH / f"results_top_{n}_by_population_{determiner}.csv"
    _generate_top_n_districts(
        kind,
        interdistrict,
        n,
        lambda d: d.population.total or 0,
        output,
        closed_enrollment_only,
    )
    print(f"info: ...{kind.name} is finished baking.")


if __name__ == "__main__":
    for kind in (
        # SimulationKind.CONSTRAINED,
        SimulationKind.CONSTRAINED_BH_WA,
        # SimulationKind.BOTTOMLESS,
        # SimulationKind.SENSITIVITY_0_1,
        # SimulationKind.SENSITIVITY_0_3,
    ):
        generate_top_n_districts_by_population(kind)
