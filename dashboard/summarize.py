from __future__ import annotations
from collections import defaultdict
from typing import Any, Callable, Iterable, Mapping, Optional

import functools
import math
from pathlib import Path
from pprint import pformat
import urllib.parse
import random

import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

from headers import (
    DEMOGRAPHICS,
    DEMOGRAPHIC_LABELS,
    DEMOS_X_GRADES,
    DISTRICT_NAMES_IN_STATE_SET,
    GRADES,
    THRESHOLDS,
    SCHOOL_ID_TO_DISTRICT_ID,
    BASE_URL,
    DISTRICTS_IN_STATE,
    STATES_LIST,
)
from eat import (
    Analytics,
    District,
    School,
    Population,
    TravelTimes,
)

"""
This module plots stuff
"""

RED = "#D81B60"
ORANGE = "#FE6100"
YELLOW = "#FFC107"
GREEN = "#009F6B"
BLUE = "#1E88E5"
VIOLET = "#785EF0"

# --- helpers


def rainbow_colors(n: int) -> list[str]:
    """Create a list of ``n`` colors.

    Based on Paul Tol's notes on colour, 2012
    """
    colors = []
    for x in np.linspace(0.0, 1.0, n):
        r = round(
            255
            * (0.472 - 0.567 * x + 4.05 * x**2)
            / (1 + 8.72 * x - 19.17 * x**2 + 14.1 * x**3)
        )
        g = round(
            255
            * (
                0.108932
                - 1.22635 * x
                + 27.284 * x**2
                - 98.577 * x**3
                + 163.3 * x**4
                - 131.395 * x**5
                + 40.634 * x**6
            )
        )
        b = round(
            255 / (1.97 + 3.54 * x - 68.5 * x**2 + 243 * x**3 - 297 * x**4 + 125 * x**5)
        )
        r = min(255, max(0, r))
        g = min(255, max(0, g))
        b = min(255, max(0, b))
        colors.append(f"#{r:02x}{g:02x}{b:02x}")
    return colors


def interweave(*lists: Iterable[Any]) -> list[Any]:
    """Intrweave 2 or more lists"""
    return [value for tuple_ in zip(*lists) for value in tuple_]


def grade_range(*grade_spans: list[str]) -> list[str]:
    """Returns the full range of grades, given an input list of grafes that may
    have holes in it

    Example:
       >>> grade_range(["PK", "K", "1", "4", "5", "6"])
       ['PK', 'K', '1', '2', '3', '4', '5', '6']
    """
    a = len(GRADES) - 1
    b = 0
    for grade_span in grade_spans:
        a = min(a, GRADES.index(grade_span[0]))
        b = max(b, GRADES.index(grade_span[-1]))
    return GRADES[a : b + 1]


_describe_grade = {
    "PK": "pre-kindergarten",
    "KG": "kindergarten",
    "1": "1st grade",
    "2": "2nd grade",
    "3": "3rd grade",
    "4": "4th grade",
    "5": "5th grade",
    "6": "6th grade",
    "7": "7th grade",
    "8": "8th grade",
    "9": "9th grade",
    "10": "10th grade",
    "11": "11th grade",
    "12": "12th grade",
    "13": "13th grade",
}
_describe_grade_shorthand = {
    "PK": "preK",
    "KG": "K",
}


def _describe_grade_span(grade_span: list[str], *, short: bool = False) -> str:
    if len(grade_span) == 0:
        return "no grades"
    if len(grade_span) == 1:
        grade = grade_span[0]
        return (
            _describe_grade_shorthand.get(grade, grade)
            if short
            else _describe_grade.get(grade, grade)
        )
    lower = grade_span[0]
    upper = grade_span[-1]
    lookup = _describe_grade_shorthand if short else _describe_grade
    connect_and = ", " if short else " and "
    connect_range = "–" if short else " through "
    connector = "–" if short else " through "
    if len(grade_span) == 2:
        connector = ", " if short else " and "
    return f"{lookup.get(lower, lower)}{connector}" f"{lookup.get(upper, upper)}"


def md(where, text: str, **kwargs):
    """Pre-process markdown"""
    text = text.replace(
        "<|", """<mark style="background-color: rgb(253, 222, 150);">"""
    )
    text = text.replace("|>", "</mark>")
    where.markdown(text, unsafe_allow_html=True, **kwargs)


def _html_format(
    text: str,
    *,
    bold: bool = False,
    italic: bool = False,
    underline: bool = False,
    code: bool = False,
    text_align: Optional[str] = None,
    font_size: Optional[Any] = None,
    color: Optional[Any] = None,
) -> str:
    """Formats font as HTML"""
    style = {}
    if font_size:
        style["font-size"] = str(font_size)
    if color:
        style["color"] = str(color)
    style_str = " ".join(f"{k}: {v};" for k, v in style.items())
    if code:
        text = f"<code>{text}</code>"
    if bold:
        text = f"<b>{text}</b>"
    if italic:
        text = f"<i>{text}</i>"
    if underline:
        text = f"<u>{text}</u>"
    html = f"""<font style="{style_str}">""" f"""{text}""" f"""</font>"""
    if text_align is not None:
        html = (
            f"""<div style="text-align: {text_align};">\n"""
            f"""{html}\n"""
            f"""</div"""
        )
    return html


# --- actual plotting/summarizing functions


def analytics(analytics: Analytics) -> str:
    """Returns an HTML-formatted sentence describing the change in dissimilarity"""
    delta = (
        analytics.post_dissimilarity - analytics.pre_dissimilarity
    ) / analytics.pre_dissimilarity
    if delta < 0:
        decrease = _html_format("decrease", color=BLUE, bold=False)
        return (
            f"""Measure of dissimilarity could {decrease} by """
            f"""<b>{-delta:.1%}</b>"""
        )

    datum = f"{delta:.1%}"
    if datum == "0.0%":
        return f"""Measure of dissimilarity might not significantly change"""

    increase = _html_format("increase", color=RED, bold=False)
    # I doubt this line will ever be reached, but just in case
    return f"""Measure of dissimilarity could {increase} by <b>{datum}</b>"""


_travel_time_significance_threshold = 0.4


def switched_travel_times(travel_times: TravelTimes, demo: str = "total") -> str:
    """Returns an HTML-formatted sentence describing the change in travel times
    for a specific demographic ("total", "asian", etc.)
    """
    before = travel_times.switcher_previous[demo] or 0
    after = travel_times.switcher_new[demo] or 0
    if math.isnan(before):
        before = 0
    if math.isnan(after):
        after = 0
    delta = (after - before) / 60  # absolute change, in minutes
    delta_int = round(abs(delta))
    datum = f"{delta_int} minute" + ("s" if delta_int != 1 else "")

    demographic = (
        "the students who switched"
        if demo == "total"
        else f"the {DEMOGRAPHICS[demo]} students who switched"
    )
    if delta >= _travel_time_significance_threshold:
        increase = _html_format("increase", color=RED, bold=False)
        return (
            f"""Travel times could {increase} by about <b>{datum}</b> """
            f"""for {demographic}"""
        )

    if abs(delta) < _travel_time_significance_threshold:
        verb = _html_format("could not significantly change", color=BLUE, bold=True)
        return f"Travel times {verb} for {demographic}"

    decrease = _html_format("decrease", color=BLUE, bold=False)
    return (
        f"""Travel times could {decrease} by about <b>{datum}</b> """
        f"""for {demographic}"""
    )


def before_after_arrow(where, before: str, after: str, *, font_size: str = "xxx-large"):
    c_before, c_arrow, c_after = where.columns(3)
    c_before.markdown(
        _html_format(before, font_size=font_size, text_align="right", code=False),
        unsafe_allow_html=True,
    )
    c_arrow.markdown(
        _html_format("⟶", font_size=font_size, text_align="center"),
        unsafe_allow_html=True,
    )
    c_after.markdown(
        _html_format(after, font_size=font_size, text_align="left", code=False),
        unsafe_allow_html=True,
    )


def grade_span(where, cluster: list[School], *, when: str = "after"):
    """Takes a list of schools and returns a combined DataFrame showing what
    grades they offer

    Args:
       where: st or st.sidebar
       when: "before" or "after"
    """
    assert len(cluster) >= 1
    assert when in ("before", "after")
    columns = ["School"]
    columns += grade_range(*[school.grade_span_before for school in cluster])
    dfs = [school.summarize_grade_span(when=when) for school in cluster]
    df = pd.concat(dfs)[columns].reset_index(drop=True)
    where.dataframe(df)


def _description_grade_spans_before(cluster: list[School]) -> str:
    #  msg = (
    #      "School mergers work by combining the attendance boundaries of two or more adjacent schools "
    #      "and changing the grade levels that each school involved in the merger subsequently serves. "
    #  )
    msg = ""
    # case 0: singleton
    if len(cluster) == 1:
        msg += (
            f"In the case of results for this district, {cluster[0].school_name} "
            f"was not merged with an adjacent school and would continue to offer "
            f"{_describe_grade_span(cluster[0].grade_span_before)}."
        )
        return msg
    the_schools = "Both schools" if len(cluster) == 2 else f"All {len(cluster)} schools"
    # case 1: all schools offer the same grade spans
    span_0 = cluster[0].grade_span_before
    if all(s.grade_span_before == span_0 for s in cluster):
        #   msg += (
        #       f"For example, {the_schools} offer {_describe_grade_span(span_0)} "
        #       f"before merging."
        #   )
        msg += f"{the_schools} offer {_describe_grade_span(span_0)} " f"before merging."
    # case 2: at least one school offers a different grade span
    else:
        strs = [
            f"{s.school_name} offered {_describe_grade_span(s.grade_span_before)}"
            for s in cluster
        ]
        strs_ = ", ".join(strs[:-1]) + ", and " + strs[-1]
        #   msg += f"For example, before merging, {strs_}."
        msg += f"Before merging, {strs_}."

    return msg


def _description_grade_spans_after(cluster: list[School]) -> str:
    strs = [
        f"{s.school_name} could offer <|{_describe_grade_span(s.grade_span_after)}|>"
        for s in cluster
    ]
    strs_ = ", ".join(strs[:-1]) + ", and " + strs[-1]
    msg = f"After merging, {strs_}."
    return msg


def cluster_grade_spans(where, cluster: list[School]):
    if len(cluster) == 1:
        md(where, _description_grade_spans_before(cluster))
        # grade_span(where, cluster, when="before")
        # with where.expander("Demographics across grades"):
        #    with where.container(height=600):
        #       demos_grades(where, cluster, when="before")
    else:
        md(where, _description_grade_spans_before(cluster))
        #   grade_span(where, cluster, when="before")
        #   with where.expander("Demographics across grades (status quo)"):
        #       with where.container(height=600):
        #           demos_grades(where, cluster, when="before")

        md(where, _description_grade_spans_after(cluster))
        #   grade_span(where, cluster)
        #   with where.expander("Demographics across grades (potential mergers)"):
        #       with where.container(height=600):
        #           demos_grades(where, cluster, when="after")

    #   md(
    #       where,
    #       (
    #           "Merging the schools in this way could allow for students from different backgrounds to meet and possibly befriend each other, strengthen a sense of community across the schools, and allow a chance for every student to access the different resources available across the schools, such as teachers, spaces, advanced programs, and PTAs."
    #       ),
    #   )


def district_clusters(where, clusters: dict[str, list[School]]):
    """Count triplets, pairs, and singletons"""
    counts = {1: 0, 2: 0, 3: 0}
    for cluster in clusters.values():
        n = len(cluster)
        counts[n] += 1

    if sum(counts.values()) == counts[1]:
        md(where, f"We could not find any feasible mergers for this configuration.")
        return

    triplets = "form no triplets"
    pairs = "form no pairs"
    singletons = "leave no school unclustered"
    if counts[3]:
        triplets = f"triple {3*counts[3]} schools into <|{counts[3]} triplet" + (
            "s|>" if counts[3] != 1 else "|>"
        )
    if counts[2]:
        pairs = f"pair {2*counts[2]} schools into <|{counts[2]} pair" + (
            "s|>" if counts[2] != 1 else "|>"
        )
    if counts[1]:
        singletons = (
            f"leave <|{counts[1]}|> school"
            + ("s" if counts[1] != 1 else "")
            + " unclustered"
        )

    md(
        where,
        (
            f"This configuration proposes to {pairs}, {triplets}, and {singletons}. Explore these on the map below."
        ),
    )


def demos_grades(
    where, cluster: list[School], *, when: str = "after", use_rainbow_colors=True
):
    """Plot extra detailed demographics distribution across all grades for a list
    of schools

    Args:
          where: st or st.sidebar
          when: "before" or "after"
    """
    assert when in ("before", "after")
    for school in cluster:
        md(where, f"#### {school.school_name}")
        distribution = school.__getattribute__(f"grades_population_{when}")
        grades = grade_range(school.__getattribute__(f"grade_span_{when}"))
        demo_data = {
            "Grade": [g for g in grades for d in DEMOGRAPHICS],
            "Demographic": DEMOGRAPHIC_LABELS * len(grades),
            "Count": [distribution[g][d] for g in grades for d in DEMOGRAPHICS],
        }
        # (todo: get ride of null grades here)
        df_demos_plot = pd.DataFrame(demo_data)
        # where.dataframe(df_demos_plot)

        colors = "Grade:N"
        if use_rainbow_colors:
            colors = alt.Color(
                "Grade:N",
                scale=alt.Scale(domain=list(grades), range=rainbow_colors(len(grades))),
            )

        chart = (
            alt.Chart(df_demos_plot)
            .mark_bar()
            .encode(
                x=alt.X("Count:Q", title="Count"),
                y=alt.Y("Grade:N", title="", sort=grades),
                color=colors,
                row=alt.Row(
                    "Demographic:N",
                    title="Demographic",
                    spacing=15,
                    sort=DEMOGRAPHIC_LABELS,
                    header=alt.Header(
                        labelOrient="top",
                        labelAlign="left",
                        labelAnchor="start",
                        labelPadding=5,
                    ),
                ),
            )
        )
        where.altair_chart(chart)


def students_who_switched(district: District, cluster: list[School]) -> str:
    """Blurb summarizing how school populations transferred between schools"""
    if len(cluster) == 1:
        return (
            f"Since {cluster[0].school_name} was not merged with an adjacent "
            f"school in this configuration, there is no change in the school's "
            f"demographics. Any improvements in integration for this district "
            f"come from other schools that were merged."
        )
    pre_dissim = district.analytics.pre_dissimilarity
    post_dissim = district.analytics.post_dissimilarity
    delta = (post_dissim - pre_dissim) / pre_dissim
    msg = ""
    n_total = sum([s.population_before.total or 0 for s in cluster])
    # not sure about this calculation... double check:
    n_switched = round(
        sum(abs((s.population_after - s.population_before).total or 0) for s in cluster)
    )
    if delta <= 0 and n_switched > 0:
        p_switched = n_switched / n_total
        msg = f"If the district were to merge the schools in this cluster, we estimate that approximately <|{n_switched:,}|> students (<|{p_switched:.1%}|> out of all elementary students in the cluster) would have to switch schools. "

    return msg


def demographics_v0(
    where, cluster: list[School], *, when: str = "after", use_rainbow_colors=True
):
    """Plot demographics distribution for a list of Schools

    Args:
       where: st or st.sidebar
       when: "before" or "after"
    """
    assert when in ("before", "after")
    # integrate across grades
    counts = {}
    for school in cluster:
        counts[school.ncessch_id] = dict.fromkeys(DEMOGRAPHICS, 0)
        for demo, grade in DEMOS_X_GRADES:
            delta = school.__getattribute__(f"grades_population_{when}")[grade][demo]
            counts[school.ncessch_id][demo] += delta
        for demo in DEMOGRAPHICS:
            counts[school.ncessch_id][demo] = round(counts[school.ncessch_id][demo])

    demo_data = {
        "Demographic": DEMOGRAPHIC_LABELS * len(cluster),
        "School": [s.school_name for s in cluster] * len(DEMOGRAPHIC_LABELS),
        "Count": [counts[id][d] for id in counts for d in DEMOGRAPHICS],
    }
    df_demos_plot = pd.DataFrame(demo_data)

    colors = "Demographic:N"
    if use_rainbow_colors:
        colors = alt.Color(
            "Demographic:N",
            scale=alt.Scale(
                domain=DEMOGRAPHIC_LABELS, range=rainbow_colors(len(DEMOGRAPHICS))
            ),
        )

    chart = (
        alt.Chart(df_demos_plot)
        .mark_bar()
        .encode(
            x=alt.X("Demographic:N", title="Demographic", sort=None),
            y=alt.Y("Count:Q", title="Count"),
            color=colors,
            column="School:N",
        )
    )
    where.altair_chart(chart)


def demographics_v1(where, cluster: list[School]):
    """Plot demographics distribution before *and* after for a list of Schools"""
    for school in cluster:
        s_id = school.ncessch_id
        counts_before = {}
        counts_after = {}
        for demo in DEMOGRAPHICS:
            counts_before[demo] = round(school.population_before[demo])
            counts_after[demo] = round(school.population_after[demo])

        md(where, f"#### {school.school_name}")
        demo_data = {
            "Demographic": interweave(DEMOGRAPHIC_LABELS, DEMOGRAPHIC_LABELS),
            "Time": ["Before merging", "After merging"] * len(DEMOGRAPHIC_LABELS),
            "Count": interweave(counts_before.values(), counts_after.values()),
        }
        df_demos_plot = pd.DataFrame(demo_data)

        chart = (
            alt.Chart(df_demos_plot)
            .mark_bar()
            .encode(
                x=alt.X("Demographic:N", title="Demographic", sort=None),
                y=alt.Y("Count:Q", title="Count"),
                color="Time:N",
                column=alt.Column(
                    "Time:N", title="Time", sort=["Before merging", "After merging"]
                ),
            )
        )
        where.altair_chart(chart)


def demographics_v2(where, cluster: list[School]):
    """Plot demographics distribution before *and* after for a list of Schools
    (version 2)
    """
    for school in cluster:
        s_id = school.ncessch_id
        counts_before = {}
        counts_after = {}
        for demo in DEMOGRAPHICS:
            counts_before[demo] = round(school.population_before[demo]) or None
            counts_after[demo] = round(school.population_after[demo]) or None

        demo_data = {
            "Demographic": interweave(DEMOGRAPHIC_LABELS, DEMOGRAPHIC_LABELS),
            "Time": ["Before merging", "After merging"] * len(DEMOGRAPHIC_LABELS),
            "Count": interweave(counts_before.values(), counts_after.values()),
        }
        df_demos_plot = pd.DataFrame(demo_data)

        chart = (
            alt.Chart(df_demos_plot)
            .mark_bar()
            .encode(
                y=alt.Y("Count:Q", title="Count"),
                x=alt.X("Time:N", title="", sort=None, axis=alt.Axis(labels=False)),
                color=alt.Color("Time:N", sort=None),
                column=alt.Column(
                    "Demographic:N",
                    title=school.school_name,
                    spacing=15,
                    sort=DEMOGRAPHIC_LABELS,
                    header=alt.Header(
                        labelOrient="bottom",
                        labelPadding=20,
                    ),
                ),
            )
            .properties(width=70, height=200)
        )
        where.altair_chart(chart)


def _determine_majority_demographics(
    population: Population, threshold: float = 0.4
) -> list[str]:
    """Returns list of demographics that make the majority of the population

    Up to three
    """
    demos = {demo: population[demo] or 0 for demo in DEMOGRAPHICS}
    if "total" in demos:
        del demos["total"]
    top_demos = sorted(demos, key=lambda d: demos[d], reverse=True)
    top_n = demos[top_demos[0]]
    threshold_n = round(top_n * threshold)
    top_demos = [d for d in top_demos if demos[d] > threshold_n]
    return top_demos[:3]


def _description_district_demographics(district: District) -> str:
    """Writes a paragraph summarizing the demographics of a district population"""
    demos = _determine_majority_demographics(district.population)
    demos = [f"{DEMOGRAPHICS[d]}" for d in demos]
    n = len(demos)
    if n >= 3:
        demos_str = ", ".join(demos[:-1]) + ", and " + demos[-1]
    elif n == 2:
        demos_str = " and ".join(demos)
    elif n == 1:
        demos_str = demos[0]
    else:
        demos_str = "[something went wrong here]"

    n_schools = len(district.schools)
    n_schools_ = f"<|{n_schools}|> closed-enrollment elementary school" + (
        "s" if n_schools != 1 else ""
    )

    msg = (
        f"{district.name} has <|{district.population.total:,}|> students "
        f"across {n_schools_}, with the majority being {demos_str} students. "
    )

    return msg


def district_demographics_v0(where, district: District, *, use_rainbow_colors=True):
    """One summarizing distribution plot for a district"""
    data = {
        "Demographic": DEMOGRAPHIC_LABELS,
        "Count": [district.population[demo] for demo in DEMOGRAPHICS],
    }
    df_plot = pd.DataFrame(data)

    colors = alt.Color("Demographic:N", sort=None)
    if use_rainbow_colors:
        colors = alt.Color(
            "Demographic:N",
            scale=alt.Scale(
                domain=DEMOGRAPHIC_LABELS, range=rainbow_colors(len(DEMOGRAPHIC_LABELS))
            ),
        )

    chart = (
        alt.Chart(df_plot)
        .mark_bar()
        .encode(
            x=alt.X("Demographic:N", sort=None),
            y=alt.Y("Count"),
            color=colors,
        )
        .properties(width=500)
    )
    chart_title = f"Demographics of {district.name}"
    chart = chart.properties(title=chart_title)
    chart = chart.configure_title(anchor="middle")
    where.altair_chart(chart)
    md(where, _description_district_demographics(district))


def _describe_minutes(minutes: float, *, handwavy=False) -> str:
    if math.isnan(minutes):
        return "[NaN]"
    if minutes < 2.25:
        return "a couple minutes"
    if minutes < 3.25:
        return "a few minutes"
    if not handwavy:
        return f"<|{round(minutes)}|> minutes"
    m = math.floor(minutes)
    s = math.ceil(60 * (minutes - (m * 60)))
    if s < 15:
        return f"just over <|{m}|> minutes"
    if s > 45:
        return f"just under <|{m+1}|> minutes"
    return f"around <|{m+1}|> minutes"


def _format_duration(seconds: int, signed=True) -> str:
    """i.e. 10m 30s"""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    sign = "+" if seconds >= 0 else "–"
    sign = sign if signed else ""
    if h:
        return f"{sign}{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{sign}{m}m {s:02d}s"
    return f"{sign}{s}s"


def _description_travel_times(district: District) -> str:
    """Blurb of the travel times results for a given District"""
    tt = district.analytics.travel_times_per_individual
    sq = (tt.status_quo.total or 0) / 60
    sq_ = _describe_minutes(sq, handwavy=True)
    msg = f"Before mergers, the average student travel " f"time is {sq_}.\n\n"

    if district.analytics.switched_population.total in (None, math.nan, 0.0):
        msg += (
            "No feasible mergers were found for this configuration, "
            "so no students are estimated to switch schools."
        )
        return msg

    sw_pre = (tt.switcher_previous.total or 0) / 60
    sw_new = (tt.switcher_new.total or 0) / 60
    sw_pre_ = _describe_minutes(sw_pre)
    sw_new_ = _describe_minutes(sw_new)
    delta_s = (
        _format_duration(math.ceil(60 * (sw_new - sw_pre)))
        if not (math.isnan(sw_new) or math.isnan(sw_pre))
        else "+???s"
    )

    switched_count = round(district.analytics.switched_population.total)
    msg += (
        f"In this configuration, <|{district.proportion_switched:.1%}|> of "
        f"elementary students (approximately <|{switched_count:,}|>) would switch schools. "
        f"For these students, their average travel time starts "
    )
    if sw_pre < sq - 1:
        msg += f"less than the average "
    msg += (
        f"at {sw_pre_} before mergers and ends around {sw_new_} after "
        f"(<|{delta_s}|>)"
    )
    if sw_new < sq - 1:
        msg += f", which is still under the existing average"
    msg += "."

    return msg


def travel_times_v0(where, district: District, *, per_individual=True):
    """Visualize the change in travel times for a district (version 0)

    Does not display bars when switchers total is nan
    """
    analytics = district.analytics
    if per_individual:
        travel_times = analytics.travel_times_per_individual / 60
        md(where, _description_travel_times(district))
        md(
            where,
            "These travel times are calculated as car driving times. Elementary school students travel to and from school in a variety of ways—by bus, walking, or by car, for instance—so these travel times should interpretted according to local context.",
        )
    else:
        travel_times = analytics.travel_times / 60

    if district.proportion_switched == 0.0:
        return

    categories = {
        "Status quo": [travel_times.status_quo[d] for d in DEMOGRAPHICS],
        "Switchers (before)": [travel_times.switcher_previous[d] for d in DEMOGRAPHICS],
        "Switchers (after)": [travel_times.switcher_new[d] for d in DEMOGRAPHICS],
    }
    demographic_labels = list(DEMOGRAPHIC_LABELS)
    if per_individual and "Total" in demographic_labels:
        i = demographic_labels.index("Total")
        demographic_labels[i] = "All"
    data = {
        "Demographic": interweave(*[demographic_labels] * 3),
        "Population": list(categories.keys()) * len(demographic_labels),
        "Travel time": interweave(*categories.values()),
    }
    df_plot = pd.DataFrame(data)

    chart = (
        alt.Chart(df_plot)
        .mark_bar()
        .encode(
            x=alt.X("Population:N", title="", sort=None, axis=alt.Axis(labels=False)),
            y=alt.Y("Travel time:Q", title="Travel time (minutes)"),
            color=alt.Color("Population:N", sort=None),
            column=alt.Column(
                "Demographic:N",
                spacing=20,
                sort=demographic_labels,
                header=alt.Header(
                    labelOrient="bottom",
                    title="Demographic",
                    titleOrient="bottom",
                ),
            ),
        )
    )
    chart_title = (
        "Travel time per individual across demographics"
        if per_individual
        else "Summed travel times across demographics"
    )
    chart = chart.properties(title=chart_title)
    chart = chart.configure_title(anchor="middle")
    chart = chart.properties(height=200)
    where.altair_chart(chart)


def travel_times_v1(
    where, analytics: Analytics, *, per_individual=False, use_rainbow_colors=True
):
    """Visualize the change in travel times for a district (version 1)"""
    if per_individual:
        travel_times = round(analytics.travel_times_per_individual / 60)
    else:
        travel_times = round(analytics.travel_times / 60)

    demographics = list(DEMOGRAPHICS)
    if "total" in demographics:
        demographics.remove("total")
    demographic_labels = [DEMOGRAPHICS[d] for d in demographics]

    categories = {
        "Status quo": [travel_times.status_quo[d] for d in demographics],
        "Switchers (before)": [travel_times.switcher_previous[d] for d in demographics],
        "Switchers (after)": [travel_times.switcher_new[d] for d in demographics],
    }
    data = {
        "Demographic": interweave(*[demographic_labels] * 3),
        "Population": list(categories.keys()) * len(demographic_labels),
        "Travel time": interweave(*categories.values()),
    }
    df_plot = pd.DataFrame(data)

    colors = alt.Color("Demographic:N", sort=None)
    if use_rainbow_colors:
        colors = alt.Color(
            "Demographic:N",
            scale=alt.Scale(
                domain=demographic_labels, range=rainbow_colors(len(demographics))
            ),
        )

    chart = (
        alt.Chart(df_plot)
        .mark_bar()
        .encode(
            x=alt.X("sum(Travel time)", title="Travel time (minutes)"),
            y=alt.Y("Population:N", title="", sort=None, axis=alt.Axis(labels=False)),
            color=colors,
        )
        .properties(width=600)
    )
    chart_title = (
        "Travel times (individual)" if per_individual else "Travel times (total)"
    )
    chart = chart.properties(title=chart_title)
    chart = chart.configure_title(anchor="middle")
    where.altair_chart(chart)


def district(where, district: District):
    num_students = round(
        sum(s.population_before.total or 0 for s in district.schools.values())
    )
    data = {
        "District name": district.name,
        "NCES District ID": f"{district.nces_id:07d}",
        "Number of schools": len(district.schools),
        "Number of students": num_students,
    }
    where.table(pd.DataFrame(data.items(), columns=("Detail", "Value")))


def school_travel_times(where, school: School):
    pre = school.travel_times_per_individual.switcher_previous
    post = school.travel_times_per_individual.switcher_new
    # print(f"{pre = }, {post = }")
    if pre is None or post is None:
        # md(where, f"missing pre/post for {school.school_name} ({school.ncessch_id})")
        return  # no data to show
    data = {
        "Demographic": np.repeat(list(DEMOGRAPHICS.values()), 2),
        "When": np.tile(["Before merging", "After merging"], len(DEMOGRAPHICS)),
        "Travel time": [
            when[demo] / 60 if when else None
            for demo in DEMOGRAPHICS
            for when in (pre, post)
        ],
    }
    df_plot = pd.DataFrame(data)

    chart = (
        alt.Chart(df_plot)
        .mark_bar()
        .encode(
            x=alt.X("When:N", title="", sort=None, axis=alt.Axis(labels=False)),
            y=alt.Y("Travel time:Q", title="Travel time (minutes)"),
            color=alt.Color("When:N", sort=None),
            column=alt.Column(
                "Demographic:N",
                title=f"{school.school_name}",
                # spacing=15,
                sort=DEMOGRAPHIC_LABELS,
                header=alt.Header(
                    labelOrient="bottom",
                    labelPadding=20,
                ),
            ),
        )
        .properties(width=70, height=200)
    )
    where.altair_chart(chart)


def leaflet(where, district: District) -> str:
    """The first paragraph(s) that readers will encounter for this district

    Returns:
       str: focal demographic (e.g., "color", "hispanic")
    """
    if district.simulation.interdistrict:
        num_extra_schools = len(district.schools_in_simulation) - len(district.schools)
        extra_districts = set(
            SCHOOL_ID_TO_DISTRICT_ID[s_id]
            for s_id in district.schools_in_simulation
            if s_id not in district.schools
        )
        extra_district_names = []
        for nces_id in extra_districts:
            try:
                extra_district_names.append(DISTRICTS_IN_STATE[district.state][nces_id])
            except:
                pass
        extra_schools_label = "schools" if num_extra_schools != 1 else "school"
        extra_districts_label = "districts" if len(extra_districts) != 1 else "district"
        such_as = ""
        if len(extra_districts) == 1:
            such_as = f", {extra_district_names[0]}" if extra_district_names else ""
        elif len(extra_district_names) == 2:
            such_as = f""", including {" and ".join(extra_district_names)}"""
        elif len(extra_district_names) > 2:
            such_as = f""", such as {" and ".join(extra_district_names[:2])}"""
        md(
            where,
            f"""Allowing school mergers between districts introduces {district.name} to <|{num_extra_schools}|> additional {extra_schools_label} from <|{len(extra_districts)}|> surrounding {extra_districts_label}{such_as}.""",
        )

    merged_clusters = [c for c in district.clusters.values() if len(c) > 1]
    if len(merged_clusters) == 0:
        # district.simulation.status
        summary = f"""Our simulations could not converge on any school merger clusters. School mergers may be an infeasible integration strategy with respect to how our simulations were arranged for this district."""
        md(where, summary)
        return "color"

    # -
    impact = district.impact
    # -

    majority_demos = district.population.majority_demographics()
    focal_demo = impact.focal_demo
    if focal_demo != "color":
        for focal_demo in impact.focal_demos:
            if focal_demo in majority_demos:
                break
    if focal_demo == "white":
        focal_demo = "color"
    focal_demo_name = (
        "students of color"
        if focal_demo == "color"
        else f"{DEMOGRAPHICS[focal_demo]} students"
    )

    p_switched = (
        district.analytics.switched_population.total
        / district.analytics.all_population.total
    )
    travel_times_delta = (
        district.analytics.travel_times_per_individual.switcher_new
        - district.analytics.travel_times_per_individual.switcher_previous
    ).total / 60

    background = f"""According to our data, <|{impact.district_concentration.color:.1%}|> of elementary school students in {district.name} are classified as students of color, and they are concentrated in <|{len(impact.overconcentrated_schools_pre["color"])}|> out of the <|{len(district.schools)}|> elementary schools, where <|{impact.average_overconcentrated_concentration_before["color"].color:.1%}|> are students of color on average."""

    if focal_demo != "color":
        background += f"""\n\nIn particular, {focal_demo_name} make up a majority of the district population and are concentrated in <|{len(impact.overconcentrated_schools_pre[focal_demo])}|> of the {len(district.schools)} schools at <|{impact.average_overconcentrated_concentration_before[focal_demo][focal_demo]:.1%}|> {focal_demo_name}, compared to <|{impact.district_concentration[focal_demo]:.1%}|> across the whole district."""

    n = len(impact.overconcentrated_schools_pre["color"])
    gcs = impact.greatest_changing_schools["color"][0]
    descriptor_color = "greater" if n == 2 else "greatest"

    opportunity = f"""Implementing the school mergers proposed below could assign students more equitably across the district's schools, changing the average percentage of students of color at these concentrated schools from <|{impact.average_overconcentrated_concentration_before["color"].color:.1%}|> to <|{impact.average_overconcentrated_concentration_after["color"].color:.1%}|>, with <|{gcs.school_name}|> experiencing the {descriptor_color} change from <|{impact.school_concentrations_pre[gcs].color:.1%}|> to <|{impact.school_concentrations_post[gcs].color:.1%}|>."""

    determiner = (
        "districts" if district.clusters_from_neighboring_districts else "district"
    )
    tradeoffs = f"""These changes would require about <|{p_switched:.1%}|> of students across the {determiner} to switch schools and experience an average change of <|{travel_times_delta:+.1f}|> minutes in their travel time to school each way."""

    summary = "\n\n".join([background, opportunity, tradeoffs])
    md(where, summary)

    return focal_demo


def _make_link(
    base_url: str,
    configuration: dict[str, Any],
    district: District | None = None,
    **kwargs,
) -> str:
    state = STATES_LIST[configuration["selected_state"]]

    _query_params = {
        "interdistrict": ("i", int),
        "threshold": ("t", lambda threshold_label: THRESHOLDS[threshold_label]),
        "selected_district": (
            "nces",
            lambda district_name: f"{DISTRICTS_IN_STATE[state][district_name]:07d}",
        ),
    }
    query = {}
    for key_name, (new_key_name, func) in _query_params.items():
        if key_name not in configuration:
            continue
        query[new_key_name] = func(configuration[key_name])
    for key_name, value in kwargs.items():
        query[key_name] = value

    if district is not None:
        school = district.clusters[configuration["selected_cluster"]][0]
        query["nces"] = f"{school.nces_id:012d}"

    query_string = urllib.parse.urlencode(query)
    return f"{base_url}?{query_string}"


def hyperlink(where, district: District | None = None) -> None:
    """Renders permalink to this simulation for sharing"""
    configuration: dict[str, Any] = where.session_state
    link = _make_link(BASE_URL, configuration, district)
    msg = f"Want to share these results with others? Use this link to return to this configuration: <{link}>"
    md(where, msg)
    where.code(link)


def survey_link(where, sid, district: District | None = None) -> str:
    configuration: dict[str, Any] = where.session_state
    link = _make_link(
        "https://neu.co1.qualtrics.com/jfe/form/SV_1C9zPpTutj551dQ",
        configuration,
        district,
        sid=sid,
    )
    return link
