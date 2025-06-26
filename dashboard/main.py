from __future__ import annotations
from typing import Any, Callable

import functools
import math
from pathlib import Path
import sys
import traceback
import os, random

menu_items = {
    "Get help": "mailto:hello@pluralconnections.org",
    "Report a Bug": "mailto:hello@pluralconnections.org",
    # "About": "",
}

import streamlit as st

st.set_page_config(
    page_title="School Mergers",
    page_icon=None,  # "icon.png",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items=None,
)

# Generate session ID
if not "sid" in st.session_state:
    st.session_state["sid"] = hex(hash(random.random()))[2:]

from headers import (
    DEBUG_MODE,
    DEMOGRAPHICS,
    DISTRICT_ID_TO_STATE,
    THRESHOLDS,
    THRESHOLDS_STR,
    STATES_LIST,
    DISTRICT_NAMES_IN_STATE,
    DISTRICTS_IN_STATE,
    SCHOOL_ID_TO_DISTRICT_ID,
    SCHOOL_NAMES_IN_DISTRICT,
    NUM_DISTRICTS,
    NUM_ELEMENTARY_SCHOOLS,
    NUM_STUDENTS,
)
import eat
import summarize
import logger
from summarize import md
import maps

# needs to be imported last
from streamlit_theme import st_theme

theme = st_theme()
is_dark_mode = theme and theme.get("base") == "dark"

#
# The Dashboard Streamlit app module itself
#

# https://cheat-sheet.streamlit.app/
# helpers

line_break = lambda x: x.write("<br>", unsafe_allow_html=True)
horizontal_bar = lambda x: x.write("<hr>", unsafe_allow_html=True)

#
# ------------------
# Sidebar extra info
# ------------------
#

with st.sidebar:
    md(
        st,
        """
# Learn more about this project

This is a research project by the [Plural Connections Group](http://pluralconnections.org/) at Northeastern University, with generous support from the Overdeck Family Foundation.

To generate these hypothetical school pairings and triplings, we use data from the 2021/2022 US Department of Education Common Core of Data; 2020 US Census; and other datasets (including estimated school attendance boundaries) released by (Gillani et al., 2023 — reference below). To explore different school pairings and triplings, we use a technique called constraint programming: an algorithm that tries out different ways of pairing or tripling schools in order to minimize segregation between White students and students of color, subject to a number of different constraints (like how many students a school can serve, and other factors).

**Note**: This dashboard has not been updated with more recent data, so please interpret the depicted pairings and triplings as suggestive and exploratory, as new schools may have been built or boundaries may have changed since we ran the simulations shown here.

If you have any questions, please feel free to reach out to <hello@pluralconnections.org> at any time—we would love to hear from you!

N. Gillani, D. Beeferman, C. Vega-Pourheydarian, C. Overney, P. Van Hentenryck, D. Roy (2023). [Redrawing attendance boundaries to promote racial and ethnic diversity in elementary schools](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4387899). Educational Researcher.
   """,
    )

#
# ---------------------
# Main page starts here
# ---------------------
#

md(
    st,
    f"""
## School pairings and triplings

_Elementary school mergers_—i.e., pairing or tripling adjacent elementary schools so that each
serves a smaller number of grades—can foster more racially/ethnically
integrated schools.

For example, in Escondido Union, California, according to our data, Rock Springs Elementary and Bernardo Elementary offer kindergarten through 5th grade.  If these schools merge as a pair, where Rock Springs offers <|kindergarten through 2nd grade|> and Bernardo offers <|3rd grade through 5th grade|>, then the proportion of students of color could change from <|88%|> and <|41%|>, respectively, to <|61%|> and <|66%|>.""",
)

_, center, _ = st.columns([0.35, 1, 0.3])
with center:
    if is_dark_mode:
        st.image("mergers-escondido-inv.gif")
    else:
        st.image("mergers-escondido.gif")

md(
    st,
    f"""
This dashboard explores how school mergers might help advance elementary school integration across {NUM_DISTRICTS:,}
US school districts. The results may have implications for nearly 12 million students across 32 thousand "closed-enrollment" schools (schools whose enrollments are entirely determined by their attendance boundaries, according to our data).

Select a school district below to explore the possibilities.
   """,
)

st.html(
    "<style> .big-font { font-size:30px !important ; padding: 5px; background-color: lightgreen; text-align: center; } </style>"
)

#
# Handle query params and default settings here
#

_default_school = ("CA", 612880, None)


def _handle_nces(nces: str) -> tuple[str, int, int | None]:
    try:
        nces_ = int(nces)
    except ValueError:
        return _default_school
    if nces_ in DISTRICT_ID_TO_STATE:
        return (DISTRICT_ID_TO_STATE[nces_], nces_, None)
    if nces_ in SCHOOL_ID_TO_DISTRICT_ID:
        district_id = SCHOOL_ID_TO_DISTRICT_ID[nces_]
        return (DISTRICT_ID_TO_STATE[district_id], district_id, nces_)
    return _default_school


q = st.query_params
if "i" in q:
    st.session_state["interdistrict"] = q["i"].lower() in ("1", "true", "y")
if q.get("t") in THRESHOLDS_STR:
    st.session_state["threshold"] = THRESHOLDS_STR[q["t"]]
nces = q.get("nces", str(_default_school[-1] or _default_school[1]))
state, district_id, school_id = _handle_nces(nces)

if "selected_state" not in st.session_state:
    st.session_state["selected_state"] = STATES_LIST[state]
    if "selected_district" not in st.session_state:
        st.session_state["selected_district"] = DISTRICTS_IN_STATE[state][district_id]

logger.init()
logger.log(st.session_state["sid"], "START", (), query_params=q.__dict__)

# reset url
for param in ("t", "i", "nces"):
    if param in q:
        del q[param]

#
# End handle query params
#

# _, center, _ = st.columns([0.4, 1, 0.4])
left, right = st.columns([0.618, 1])
with left:
    st.selectbox("State", STATES_LIST.keys(), key="selected_state")
    state = STATES_LIST[st.session_state.selected_state]

with right:
    st.selectbox(
        "Select district", DISTRICT_NAMES_IN_STATE[state], key="selected_district"
    )

# horizontal_bar(st)
_, center, _ = st.columns([0.4, 0.618, 0.4])
with center:
    with st.expander("Extra configuration settings"):
        if "threshold" not in st.session_state:
            st.session_state["threshold"] = "80%"
        st.selectbox(
            "Minimum school enrollment",
            THRESHOLDS.keys(),
            key="threshold",
            help="When schools are merged, their resulting enrollments may shift. Configure this setting to specify what a school's minimum enrollment should be post-mergers (the default is 80% of the school's current enrollment).",
        )
        st.checkbox(
            "Allow mergers between schools in adjacent districts",
            False,
            key="interdistrict",
        )

district_name: str = st.session_state.selected_district
district_id = DISTRICTS_IN_STATE[state][district_name]
school_names = SCHOOL_NAMES_IN_DISTRICT[district_id]
interdistrict: bool = st.session_state.interdistrict
threshold: float = THRESHOLDS[st.session_state.threshold]
bottomless: bool = threshold == 0
school_decrease_threshold = "1" if bottomless else f"{1-threshold:0.1f}"
folder_name_check = "interdistrict" if interdistrict else None

md(
    st,
    f"""
## {district_name}
""",
)

if district_id == 3701500:
    st.info(
        (
            "As a part of a US Department of Education-funded Fostering Diverse Schools (FDS) project, Winston-Salem / Forsyth County Schools is exploring how residential boundary changes might foster more socioeconomically diverse and integrated schools.  That project is not considering school mergers as a policy pathway.  If you wish to learn more about the project, please visit the project website: <https://www.wsfcs.k12.nc.us/o/wsfcs/page/fostering-diverse-schools>."
        )
    )
    st.stop()

context = eat.context(
    state,
    district_id,
    interdistrict,
    school_decrease_threshold,
    folder_name_check=folder_name_check,
)

results = None
survey_link = ""
try:
    results = eat.district(context)
except Exception as e:
    st.info(
        (
            f"{district_name} is not available for this configuration.\n\n"
            f"Please try another configuration, or reach out to <hello@pluralconnections.org> with any questions."
        )
    )
    if DEBUG_MODE:
        exception_info = "".join(
            traceback.format_exception(e.__class__, e, e.__traceback__)
        )
        st.warning(f"```\n{exception_info}\n```")
else:
    if not results.closed_enrollment_only:
        st.info(
            (
                f"We only compute segregation results for closed-enrollment schools---i.e., schools whose attendance is entirely determined by their attendance boundaries. "
                f"Our data indicates that {district_name} might include some elementary schools that "
                f"are not closed enrollment, so please interpret these results with caution, and/or reach out to "
                f"<hello@pluralconnections.org> with any specific questions."
                # f"would exercise within-district choice in the event of a merger, "
                # f"obviating the integration efforts."
            )
        )

    # high-level summary
    focal_demo = summarize.leaflet(st, results)

    #  horizontal_bar(st)

    # horizontal_bar(st)

    ## Map!

    md(st, "### Possible pairings & triplings in this district")

    summarize.district_clusters(st, results.clusters)

    md(
        st,
        'In the map below, <span style="color:#6c24ff; font-weight:bold">darker purple</span> represents higher concentrations of the specified demographic.',
    )

    named_clusters = results.clusters.keys()
    if "selected_cluster" not in st.session_state and school_id is not None:
        for cluster_name, schools in results.clusters.items():
            if any(school_id == s.ncessch_id for s in schools):
                st.session_state["selected_cluster"] = cluster_name
                break

    st.selectbox(
        "Select schools to outline their boundaries and analyze expected impacts of mergers",
        named_clusters,
        key="selected_cluster",
    )
    cluster = results.clusters[st.session_state.selected_cluster]
    logger.log(
        st.session_state["sid"],
        "DASHBOARD_VIEW",
        (
            state,
            district_id,
            district_name,
            interdistrict,
            threshold,
            st.session_state.selected_cluster,
        ),
        query_params=q.__dict__,
    )

    # st.image("map-placeholder.jpg")
    try:
        maps.draw(st, results, focal_demo=focal_demo, cluster=cluster)
    #   if focal_demo != "color":
    #       md(
    #           st,
    #           f"{DEMOGRAPHICS[focal_demo]} students are particularly over-concentrated in the district's schools, yet comprise a majority of the population. Although our simulations were optimized for all students of color, it may be helpful to visualize how specifically {DEMOGRAPHICS[focal_demo]} students could be affected, so we have included this information for viewing in the map.",
    #       )
    except Exception as e:
        st.info(
            (
                f"A map is not available for this district.\n\n"
                f"Please reach out <hello@pluralconnections.org> for more information."
            )
        )
        if DEBUG_MODE:
            exception_info = "".join(
                traceback.format_exception(e.__class__, e, e.__traceback__)
            )
            st.warning(f"```\n{exception_info}\n```")

    survey_link = summarize.survey_link(st, st.session_state["sid"], results)
    st.markdown(
        '<p class="big-font"><a href="%s" target="_blank">Tell us what you think</a> about these clusters!</p>'
        % (survey_link),
        unsafe_allow_html=True,
    )

    school_names = [school.school_name for school in cluster]
    if len(school_names) <= 2:
        schools_title = " and ".join(school_names)
    else:
        schools_title = ", ".join(school_names[:-1]) + ", and " + school_names[-1]

    ## Schools
    md(st, f"#### {schools_title}")

    ### Grades offered
    #  md(st, f"""#### Grade span{"s" if len(cluster) != 1 else ""}""")
    summarize.cluster_grade_spans(st, cluster)

    ### Demographics
    md(st, "##### School demographics")

    # summary_students_who_switched = summarize.students_who_switched(results, cluster)
    # md(st, summary_students_who_switched)

    summarize.demographics_v2(st, cluster)

    if len(cluster) > 1:
        md(st, f"""##### Estimated travel (driving) times""")
        for school in cluster:
            summarize.school_travel_times(st, school)

if results:
    with st.expander("More about this district"):
        md(st, f"### More background on {district_name}")
        summarize.district_demographics_v0(st, results)
        summarize.travel_times_v0(st, results, per_individual=True)

horizontal_bar(st)

if results is not None:
    #  md(
    #      st,
    #      f"Have feedback for us? [Let us know]({survey_link}); we appreciate your perspective.",
    #  )
    st.markdown(
        '<p class="big-font"><a href="%s" target="_blank">Let us know your feedback</a>; we appreciate your input!</p>'
        % (survey_link,),
        unsafe_allow_html=True,
    )

summarize.hyperlink(st, results)

# if context.status == eat.StatusCode.FEASIBLE:
#     st.info("Note: This simulation is feasible but did not run to optimality.")

# st.json(st.session_state)
