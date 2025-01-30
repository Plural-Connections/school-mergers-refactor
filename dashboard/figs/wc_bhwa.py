#!/usr/bin/env python3

"""
Plot scatterplot of how dissim scores of optimizing for White–students of Color
matches with dissim scores of optimizing for Black/Hispanic–White/Asian.
"""

from pathlib import Path

from plotly import graph_objects as go
from scipy.stats import pearsonr, spearmanr, linregress

from .headers import *

def make_dissim_scatterplot(save: bool=True, show: bool=False):
   if not any([show, save]): return
   top_200 = set(top_200_districts(SimulationKind.CONSTRAINED, False))
   filename_wc = DATA_ROOT / "min_num_elem_schools_4_constrained" / "consolidated_simulation_results_min_num_elem_schools_4_constrained_0.2_False.csv"
   filename_bhwa = DATA_ROOT / "min_num_elem_4_constrained_bh_wa" / "consolidated_original.csv"

   df_wc = pd.read_csv(filename_wc)
   df_bhwa = pd.read_csv(filename_bhwa)
   df_wc = df_wc[df_wc["district_id"].astype(int).isin(top_200)]
   df_bhwa = df_bhwa[df_bhwa["district_id"].astype(int).isin(top_200)]
   delta_wc = (df_wc["post_dissim"] - df_wc["pre_dissim"]) / df_wc["pre_dissim"]
   delta_bhwa = (df_bhwa["post_dissim_bh_wa"] - df_bhwa["pre_dissim_bh_wa"]) / df_bhwa["pre_dissim_bh_wa"]

   scatter = go.Scatter(
      x=delta_wc, y=delta_bhwa, mode="markers",
      marker_symbol="diamond", marker_line_width=1,
      #marker_size=df_wc["num_total_all"]/5000,
      marker_size=10,
      marker_color="cornflowerblue", marker_line_color="midnightblue",
      name="ΔD",
      customdata=df_wc["num_total_all"],
      text=df_wc["district_id"],
   )

   result = pearsonr(delta_wc, delta_bhwa)
   print(f"info: \\rho = {result.statistic}, p = {result.pvalue}")
   rho = result.statistic
   result = spearmanr(delta_wc, delta_bhwa)
   print(f"info: r = {result.statistic}, p = {result.pvalue}")
   r2 = result.statistic ** 2

   slope, intercept, r_value, p_value, std_err = linregress(delta_wc, delta_bhwa)
   new_x = np.linspace(0, 1.05*min(delta_wc.min(), delta_bhwa.min()), 2)
   new_y = slope * new_x + intercept
   regression_line = go.Scatter(x=new_x, y=new_y, mode="lines",
      line=dict(
         color="gray",
         dash="dot",
      ),
      name=f"ρ = {rho:.3f}",
   )

   fig = go.Figure(data=[regression_line, scatter])
   margin = 100
   fig.update_layout(
      title="Dissimilarity measure sensitivity analysis",
      xaxis_title="ΔD White—non-White",
      yaxis_title="ΔD Black/Hispanic—White/Asian",
      xaxis_tickformat=",.0%",
      yaxis_tickformat=",.0%",
      width=800,
      height=800,
      margin=dict(l=margin, r=margin, t=margin, b=margin),
      title_font_size=24,
      legend_font_size=22,
      xaxis_title_font_size=24,
      yaxis_title_font_size=24,
      xaxis_tickfont_size=22,
      yaxis_tickfont_size=22,
   )
   fig.update_traces(
      hovertemplate="<br>".join([
         "x: %{x}",
         "y: %{y}",
         "ID: %{text}",
         "Population: %{customdata}",
      ]),
   )

   annotations = {
      633840: "Sacramento City Unified (CA)",
      623610: "Manteca Unified (CA)",
      2400420: "Howard County<br>Public Schools (MD)",
   }
   annotation_anchors = {
      633840: "left",
      623610: "left",
      2400420: "right",
   }
   for district_id, name in annotations.items():
      name = name.replace(" (", "<br>(")
      name = name.replace(")", f", {district_id:07d})")
      fig.add_annotation(
         x=delta_wc[df_wc["district_id"] == district_id].item(),
         y=delta_bhwa[df_bhwa["district_id"] == district_id].item(),
         text=name,
         font_size=16,
         showarrow=True,
         arrowhead=0,
         xanchor=annotation_anchors[district_id],
         align=annotation_anchors[district_id],
      )

   if show:
      fig.show()
   if save:
      output = FIGURE_OUTPUT_PATH / "dissim_sensitivity.pdf"
      fig.write_image(output)

def make_dissim_scatterplot_arrows():
   top_200 = set(top_200_districts(SimulationKind.CONSTRAINED, False))
   filename_wc = DATA_ROOT / "min_num_elem_schools_4_constrained" / "consolidated_simulation_results_min_num_elem_schools_4_constrained_0.2_False.csv"
   filename_bhwa = DATA_ROOT / "min_num_elem_4_constrained_bh_wa" / "consolidated_original.csv"

   df_wc = pd.read_csv(filename_wc)
   df_bhwa = pd.read_csv(filename_bhwa)
   df_wc = df_wc[df_wc["district_id"].astype(int).isin(top_200)].reset_index()
   df_bhwa = df_bhwa[df_bhwa["district_id"].astype(int).isin(top_200)].reset_index()

   scatter_source = go.Scatter(
      x=df_wc["pre_dissim"], y=df_bhwa["pre_dissim_bh_wa"],
      mode="markers", name="Status quo", legendgroup="source",
      marker_symbol="circle", marker_line_width=1, marker_size=10,
      #marker_line_color="red",
      marker_color="lightskyblue"
   )
   scatter_target = go.Scatter(
      x=df_wc["post_dissim"], y=df_bhwa["post_dissim_bh_wa"],
      mode="markers", name="Mergers", legendgroup="target",
      marker_symbol="asterisk", marker_line_width=1, marker_size=10,
      marker_line_color="crimson", #marker_color="lightskyblue"
   )

   arrows = []
   for i in range(len(df_wc)):
      arrows.append(
         go.Scatter(
            x=[df_wc["pre_dissim"][i], df_wc["post_dissim"][i]],
            y=[df_bhwa["pre_dissim_bh_wa"][i], df_bhwa["post_dissim_bh_wa"][i]],
            #mode="lines",
            line=dict(width=1),
            marker=dict(
               size=10,
               symbol="arrow-bar-up",
               angleref="previous",
               color="gray",
            ),
            showlegend=False,
         )
      )

   fig = go.Figure(
      data=arrows + [scatter_source, scatter_target],
      layout=dict(title="Optimizing", xaxis_title="X", yaxis_title="Y"),
   )
   fig.show()


if __name__ == "__main__":
   make_dissim_scatterplot(show=True, save=False)
   #make_dissim_scatterplot_arrows()
