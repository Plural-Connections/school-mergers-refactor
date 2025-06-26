from .headers import *
from .headers import _simulation_names

"""
Stuff for plotting the relative changes in school enrollment post-mergers.
Useful for the school enrollment minimum constraints sensitivity analyses.
"""


def _plot_histogram(
    df: pd.DataFrame,
    *,
    color="#8E7AB5",
    median_line_dash: str = "dash",
    median_line_width: int = 3,
    median_formatter: Callable[[float], str] = lambda x: f"{x:0.02f}",
    width: int = 500,
    height: int = 800,
    nbins: int = 30,
    cumulative=False,
    xaxis_title: str | None = None,
) -> Figure:
    fig = make_subplots(
        cols=1,
        rows=3 if cumulative else 2,
        row_heights=[0.146, 0.236, 0.618] if cumulative else [0.382, 0.618],
        shared_xaxes=True,
        x_title=xaxis_title or "",
        vertical_spacing=0.00,
    )
    fig_histogram = px.histogram(
        df,
        x="value",
        barmode="overlay",
        color_discrete_sequence=[color],
        nbins=nbins,
        marginal="box",
        hover_data=df.columns,
    )
    boxplot = fig_histogram.data[1]
    fig.append_trace(fig_histogram.data[0], row=3 if cumulative else 2, col=1)
    if cumulative:
        hist, edges = np.histogram(df["value"], bins=nbins, density=True)
        cumsum = np.cumsum(hist)
        cumsum /= np.max(cumsum)
        fig.append_trace(
            go.Scatter(
                x=edges,
                y=cumsum,
                mode="lines",
                line=dict(color=color, shape="hvh"),
                showlegend=False,
                # xaxis_visible=False,
            ),
            row=2,
            col=1,
        )
    fig.append_trace(boxplot, row=1, col=1)

    fig.update_layout(autosize=False, width=width, height=height)
    fig = fix_font(fig)
    return fig


@st.cache_data()
def get_changes(kind: SimulationKind) -> pd.DataFrame:
    data = {"name": [], "value": []}
    closures = 0
    for district_id in top_200_districts(kind):
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
                before = school.population_before.total
                after = school.population_after.total
                assert (
                    before is not None and not isinf(before) and not isnan(before)
                ), f"{before=!r}"
                assert (
                    after is not None and not isinf(after) and not isnan(after)
                ), f"{after=!r}"
                before = round(before)
                after = round(after)
                relative_change = (after - before) / before
                school_info = (
                    f"{school.school_name} ({school.ncessch_id:013d}) ({state})"
                )
                if after == 0:
                    print(f"info: school closure {before} -> {after} for {school_info}")
                    closures += 1
                # todo: skip if before < 20?
                if relative_change > 2:
                    print(
                        (
                            f"warning: discarding outlier of {relative_change:+.1%} "
                            f"({before} -> {after}) from {school_info}"
                        )
                    )
                    continue
                data["name"].append(school_info)
                data["value"].append(relative_change)

    print(f"info: closures found: {closures}")
    return pd.DataFrame(data)


def plot_changes(
    kind: SimulationKind,
    *,
    show: bool = False,
    save: bool = True,
    cumulative: bool = False,
):
    if not (save or show):
        return
    print(f"info: plotting histogram for {kind.value}...")
    df = get_changes(kind)
    n = len(df["name"])
    x_upper = float(df["value"].max())
    x_lower = float(df["value"].min())
    fig = _plot_histogram(
        df,
        cumulative=cumulative,
        xaxis_title="Relative change in school enrollment",
    )
    title = f"{_simulation_names[kind]} ({n = :,})"
    fig.update_layout(title=title)
    # fig.update_xaxes(range=[-1.05, x_upper + 0.05])
    fig.for_each_xaxis(lambda t: t.update(range=[-1.05, x_upper + 0.05]))

    if show:
        fig.show()
    output = FIGURE_OUTPUT_PATH / f"relative_changes_{kind.value}.pdf"
    if save:
        fig.write_image(output)
    print(f"info: ...done plotting histogram for {kind.value}.")


if __name__ == "__main__":
    kinds = (
        # SimulationKind.CONSTRAINED,
        SimulationKind.BOTTOMLESS,
        # SimulationKind.SENSITIVITY_0_1,
        # SimulationKind.SENSITIVITY_0_3,
    )
    for kind in kinds:
        plot_changes(kind, show=True, cumulative=True)
