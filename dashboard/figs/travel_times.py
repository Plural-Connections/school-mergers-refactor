from .headers import *
from .headers import _simulation_names
from .dissimilarity import _plot_gaussians

"""
Stuff for plotting the change in travel times across demographics.  One of the
primary results figures of our manuscripts.
"""


@st.cache_data()
def get_school_changes(
    kind: SimulationKind, interdistrict: bool = False
) -> pd.DataFrame:
    data = {"name": [], "pre_times": [], "post_times": []}
    for district_id in top_200_districts(kind, interdistrict):
        state = DISTRICT_ID_TO_STATE[district_id]
        try:
            simulation = eat_context(state, district_id, kind)
            district = eat.district(simulation)
        except AssertionError as e:
            print(
                f"warning: skipping ({state}:{district_id:07d}) due to AssertionError: {e}"
            )
            continue
        except FileNotFoundError as e:
            print(f"warning: missing simulation? {e}")
            continue
        for cluster in district.clusters.values():
            if not len(cluster) >= 2:
                continue
            for school in cluster:
                if school.district_id != district_id:
                    continue  # only focal district
                if (
                    school.travel_times_previous is None
                    or school.travel_times_new is None
                ):
                    continue
                try:
                    before = school.travel_times_previous.total / (
                        60 * school.population_before.total
                    )
                    after = school.travel_times_new.total / (
                        60 * school.population_after.total
                    )
                except ZeroDivisionError:
                    continue
                school_info = (
                    f"{school.school_name} ({school.ncessch_id:013d}) ({state})"
                )
                data["name"].append(school_info)
                data["pre_times"].append(before)
                data["post_times"].append(after)

    return pd.DataFrame(data)


def plot_schools_histograms(
    kind: SimulationKind,
    interdistrict: bool = False,
    *,
    title: str | None = None,
    output_filename: str | None = None,
    show: bool = False,
):
    df = get_school_changes(kind, interdistrict)
    n = len(df["name"])
    fig = _plot_gaussians(
        df,
        "pre_times",
        "post_times",
        nbins=30,
    )
    fig.update_layout(
        title=title or f"{_simulation_names[kind]} ({n = :,})",
        xaxis_title="Travel time (minutes)",
        yaxis_title="Frequency",
        legend_title="",
    )
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.7))
    new_names = {"pre_times": "Status quo", "post_times": "Mergers"}
    fig.for_each_trace(lambda t: t.update(name=new_names[t.name]))
    if show:
        fig.show()
    determiner = kind.value + ("_interdistrict" if interdistrict else "")
    output = FIGURE_OUTPUT_PATH / (output_filename or f"times_{determiner}.pdf")
    fig.write_image(output)
    print(f"info: ...done plotting travel times histogram for {kind.value}.")


def plot_district_histograms(
    kind: SimulationKind,
    interdistrict: bool = False,
    *,
    title: str | None = None,
    output_filename: str | None = None,
    show: bool = False,
):
    original_district_ids = top_200_districts_df(SimulationKind.CONSTRAINED, False)[
        "district_id"
    ]
    df = top_200_districts_df(kind, interdistrict)
    df = df[df["district_id"].isin(original_district_ids)]
    n = len(df["district_id"])
    print(f"info: btw, {n = :,}")
    fig = _plot_gaussians(
        df,
        "pre_times",
        "post_times",
        nbins=20,
    )
    fig.update_layout(
        title=title or f"{_simulation_names[kind]} ({n = :,})",
        xaxis_title="Travel time (minutes)",
        yaxis_title="Frequency",
        legend_title="",
    )
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.7))
    new_names = {"pre_times": "Status quo", "post_times": "Mergers"}
    fig.for_each_trace(lambda t: t.update(name=new_names[t.name]))
    if show:
        fig.show()
    determiner = kind.value + ("_interdistrict" if interdistrict else "")
    output = FIGURE_OUTPUT_PATH / (output_filename or f"times_{determiner}.pdf")
    fig.write_image(output)
    print(f"info: ...done plotting travel times histogram for {kind.value}.")


def plot_demographic_travel_times_box(
    kind: SimulationKind,
    interdistrict: bool = False,
    *,
    before_color="#6EA6CD",
    after_color="#F67E4B",
    width: int = 660,
    height: int = 460,
    show: bool = False,
    title: None | str = None,
    output_filename: None | str = None,
):
    fig = make_subplots(rows=1, cols=len(DEMOGRAPHICS))
    original_district_ids = top_200_districts_df(SimulationKind.CONSTRAINED, False)[
        "district_id"
    ]
    df = top_200_districts_df(kind, interdistrict, extra_info=True)
    df = df[df["district_id"].isin(original_district_ids)]
    n = len(df["district_id"])
    print(f"info: btw, {n = :,}")
    data = {
        "Demographic": np.repeat(list(DEMOGRAPHICS.values()), 2 * n),
        "When": np.tile(["Status quo", "Mergers"], len(DEMOGRAPHICS) * n),
        "Travel time": [
            df.at[i, f"{when}_{demo}"]
            for demo in DEMOGRAPHICS
            for i in range(n)
            for when in ("switcher_status_quo_time_num", "switcher_new_time_num")
        ],
    }
    df_plot = pd.DataFrame(data)

    fig = px.box(
        df_plot,
        x="Demographic",
        y="Travel time",
        color="When",
        color_discrete_sequence=[before_color, after_color],
    )
    fig = fix_font(fig)
    fig.update_layout(autosize=False, width=width, height=height)

    fig.update_layout(
        title=title or f"{_simulation_names[kind]} ({n = :,})",
        xaxis_title="&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Demographic",
        yaxis_title="Travel time (minutes)",
        legend_title="",
        xaxis_tickfont_size=14,
        xaxis_title_standoff=30,
        xaxis_automargin=True,
    )

    if show:
        fig.show()
    determiner = kind.value + ("_interdistrict" if interdistrict else "")
    output = FIGURE_OUTPUT_PATH / (output_filename or f"times_demos_{determiner}.pdf")
    fig.write_image(output)
