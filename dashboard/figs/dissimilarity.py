from .headers import *
from .headers import _simulation_names, _thresholds

"""
Stuff for plotting the change in dissimilarity index.  One of the primary
results figures of our manuscripts.
"""

def _plot_gaussians(df: pd.DataFrame, pre_column: str, post_column: str, *,
   before_color="#6EA6CD",
   after_color="#F67E4B",
   median_line_dash: str="dash",
   median_line_width: int=3,
   arrow_line_width: int=2,
   width: int=round(700*2/3),
   height: int=round(900*2/3),
   median_formatter: Callable[[float], str]=lambda x: f"{x:0.02f}",
   arrow_color="#505050",
   nbins=20,
) -> Figure:
   data = df[[pre_column, post_column]]
   df_melted = data.melt(var_name="when", value_name="value")
   fig = px.histogram(
      df_melted,
      x="value",
      color="when",
      barmode="overlay",
      color_discrete_sequence=[before_color, after_color],
      nbins=nbins,
   )
   fig = fix_font(fig)
   median_pre = float(df[pre_column].median())
   median_post = float(df[post_column].median())
   direction = 0
   if median_post > median_pre: direction = 1
   if median_post < median_pre: direction = -1
   fig.add_vline(
      x=median_pre,
      line_width=median_line_width,
      line_dash=median_line_dash,
      line_color=before_color,
      annotation_text=median_formatter(median_pre),
      annotation_position="top left" if direction >= 0 else "top right",
      annotation_font_color=before_color,
      annotation_font_size=14,
      annotation_borderpad=10,
   )
   fig.add_vline(
      x=median_post,
      line_width=median_line_width,
      line_dash=median_line_dash,
      line_color=after_color,
      annotation_text=median_formatter(median_post),
      annotation_position="top right" if direction >= 0 else "top left",
      annotation_font_color=after_color,
      annotation_font_size=14,
      annotation_borderpad=10,
   )
   if direction:
      full_fig = fig.full_figure_for_development(warn=False)
      xrange = full_fig.layout.xaxis.range
      x_scale = xrange[1] - xrange[0]
      delta = 0.015*direction*x_scale
      y_arrow = full_fig.layout.yaxis.range[1] * 0.2
      text = median_formatter(abs(median_post - median_pre))
      text = ("–" if direction == -1 else "+") + text
      fig.add_annotation(
         x=median_post-delta, y=y_arrow,
         ax=median_pre+delta, ay=y_arrow,
         xref="x", yref="y", axref="x", ayref="y",
         showarrow=True,
         arrowhead=2,
         arrowsize=1,
         arrowwidth=arrow_line_width,
         arrowcolor=arrow_color,
      )
      x_loc = median_post + delta
      fig.add_annotation(
         x=x_loc, y=y_arrow,
         xref="x", yref="y",
         text=text,
         font={"color": arrow_color},
         showarrow=False,
         xanchor="left" if direction == 1 else "right",
      )
   fig.update_layout(autosize=False, width=width, height=height)
   return fig

def plot_overall_histogram(
   kind: SimulationKind,
   interdistrict: bool=False,
   *,
   title: str|None=None,
   output_filename: str|None=None,
   show: bool=False
):
   original_district_ids = top_200_districts_df(SimulationKind.CONSTRAINED, False)["district_id"]
   df = top_200_districts_df(kind, interdistrict)
   df = df[df["district_id"].isin(original_district_ids)]
   n = len(df["pre_dissim"])
   fig = _plot_gaussians(
      df, "pre_dissim", "post_dissim",
      #before_color="#8E7AB5",
      #after_color="#E493B3",
   )
   print(f"info: btw, {n = :,}")
   fig.update_layout(
      title=title or f"{_simulation_names[kind]} ({n = :,})",
      xaxis_title="Dissimilarity score",
      yaxis_title="Frequency",
      legend_title="",
   )
   fig.update_layout(legend=dict(
      yanchor="top",
      y=0.99,
      xanchor="left",
      x=0.7
   ))
   new_names = {"pre_dissim": "Status quo", "post_dissim": "Mergers"}
   fig.for_each_trace(lambda t: t.update(name=new_names[t.name]))
   if show: fig.show()
   determiner = kind.value + ("_interdistrict" if interdistrict else "")
   output = FIGURE_OUTPUT_PATH / (output_filename or f"dissimilarity_{determiner}.pdf")
   fig.write_image(output)
   print(f"info: ...done plotting dissim histogram for {kind.value}.")
