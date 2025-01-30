#!/usr/bin/env python3

import figs
from figs import SimulationKind

def make_enrollment_histograms(*, save=True, show=False):
   """Plot the distribution of the relative post-merger changes in school
   enrollment.

   Note:
      Requires the CSV files to have been prebaked.
   """
   kinds = (
      SimulationKind.CONSTRAINED,
      SimulationKind.BOTTOMLESS,
      SimulationKind.SENSITIVITY_0_1,
      SimulationKind.SENSITIVITY_0_3,
   )
   for kind in kinds:
      figs.enrollment.plot_changes(kind, save=save, show=show, cumulative=True)

def make_the_csv_files():
   """Bake the high-level results.
   """
   for kind in (
      SimulationKind.CONSTRAINED,
      #SimulationKind.CONSTRAINED_BH_WA,
      #SimulationKind.BOTTOMLESS,
      #SimulationKind.SENSITIVITY_0_1,
      #SimulationKind.SENSITIVITY_0_3,
   ):
      figs.premake.generate_top_n_districts_by_population(kind, False)
   for kind in (
      SimulationKind.CONSTRAINED,
      #SimulationKind.CONSTRAINED_BH_WA,
      # SimulationKind.BOTTOMLESS,
      # SimulationKind.SENSITIVITY_0_1,
      # SimulationKind.SENSITIVITY_0_3,
   ):
      figs.premake.generate_top_n_districts_by_population(kind, True)

def make_dissim_plots():
   """Make those before -> after histogram plots for dissimilarity.

   Note:
      Requires the CSV files to have been prebaked.
   """
   figs.dissimilarity.plot_overall_histogram(
      SimulationKind.CONSTRAINED,
      interdistrict=False,
      output_filename="dissimilary_primary.pdf",
      title="Impact on integration",
   )
   figs.dissimilarity.plot_overall_histogram(
      SimulationKind.CONSTRAINED,
      interdistrict=True,
      output_filename="dissimilary_interdistrict.pdf",
      title="Interdistrict (n = 154)",
   )
   for kind in (
      #SimulationKind.CONSTRAINED,
      #SimulationKind.CONSTRAINED_BH_WA,
      #SimulationKind.BOTTOMLESS,
      #SimulationKind.SENSITIVITY_0_1,
      #SimulationKind.SENSITIVITY_0_3,
   ):
      figs.dissimilarity.plot_overall_histogram(kind)

def make_travel_times_plots():
   """Make those before -> after histogram plots for travel times.
   """
   # figs.travel_times.plot_district_histograms(
   #    SimulationKind.CONSTRAINED,
   #    interdistrict=False,
   #    output_filename="times_primary.pdf",
   #    title="Impact on travel times",
   # )
   # figs.travel_times.plot_district_histograms(
   #    SimulationKind.CONSTRAINED,
   #    interdistrict=True,
   #    output_filename="times_interdistrict.pdf",
   #    title="Interdistrict (n = 154)",
   # )
   for kind in (
      #SimulationKind.CONSTRAINED,
      SimulationKind.CONSTRAINED_BH_WA,
      #SimulationKind.BOTTOMLESS,
      #SimulationKind.SENSITIVITY_0_1,
      #SimulationKind.SENSITIVITY_0_3,
   ):
      figs.travel_times.plot_district_histograms(kind)

def make_times_demos_box_plots(show: bool=False):
   #figs.travel_times.plot_demographic_travel_times_box(SimulationKind.CONSTRAINED, show=show, title="Impact on travel times", output_filename="times_demos_primary.pdf")
   #figs.travel_times.plot_demographic_travel_times_box(SimulationKind.CONSTRAINED, interdistrict=True, show=show, title="Interdistrict (n = 154)", output_filename="times_demos_interdistrict.pdf")
   for kind in (
      #SimulationKind.CONSTRAINED,
      SimulationKind.CONSTRAINED_BH_WA,
      #SimulationKind.BOTTOMLESS,
      #SimulationKind.SENSITIVITY_0_1,
      #SimulationKind.SENSITIVITY_0_3,
   ):
      figs.travel_times.plot_demographic_travel_times_box(kind, interdistrict=False, show=show)

def make_dissim_scatterplots():
   """For WC and BHWA scatterplot comparison.
   """
   figs.wc_bhwa.make_dissim_scatterplot(save=True, show=True)
   #figs.wc_bhwa.make_dissim_scatterplot_arrows()


if __name__ == "__main__":
   #make_enrollment_histograms(show=True)
   #make_the_csv_files()
   #make_dissim_plots()
   #make_travel_times_plots()
   #make_times_demos_box_plots(show=False)
   make_dissim_scatterplots()
