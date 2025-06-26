from pathlib import Path
import random
import colorsys
from enum import Enum
import itertools
from typing import Collection

import streamlit as st
import streamlit.components.v1 as components
import folium
import folium.plugins
import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon

from headers import (
    DATA_ROOT,
    DEMOGRAPHICS,
    DISTRICT_ID_TO_STATE_BACKUP,
    GRADES,
    DISTRICT_ID_TO_STATE,
    DISTRICT_ID_TO_CENTROID,
    STATE_TO_CENTROID,
    DISTRICT_ADJACENCY_MAPS,
    SCHOOL_ID_TO_DISTRICT_ID,
    DISTRICTS_IN_STATE,
)
from eat import District, Population, School
from outlines import close_holes
from summarize import _describe_grade_span

MAP_HEIGHT_PIXELS: int = 500


@st.cache_data(ttl=3600, show_spinner="Remembering where this district is located...")
def load_gpd(csv: Path | str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(csv, ignore_geometry=True)
    gdf["geometry"] = gpd.GeoSeries.from_wkt(gdf["geometry"])
    gdf = gdf.set_geometry("geometry")
    gdf.crs = "epsg:4326"  # ?
    return gdf


@st.cache_data(ttl=3600, show_spinner="Remembering where this district is located...")
def _read_geodata(
    state: str,
    focal_district_id: int,
    neighboring_district_ids: set[int],
    clusters: None | dict[int, str] = None,
) -> pd.DataFrame:
    gdf = gpd.GeoDataFrame()
    # load the focal district FIRST, then the neighboring districts, so that
    # when clusters are shown in the map, the colors from the *focal* district
    # are used, rather than the colors from the neighboring districts
    district_ids = itertools.chain((focal_district_id,), neighboring_district_ids)
    for i, district_id in enumerate(district_ids):
        filepath = (
            DATA_ROOT
            / "school_attendance_boundaries"
            / state
            / f"{district_id:07d}.csv"
        )
        gdf_partial = load_gpd(filepath)  # ncesschid, geometry
        if i == 0:
            gdf = gdf_partial
        else:
            gdf = pd.concat([gdf, gdf_partial], ignore_index=True)
    if clusters is None:
        return gdf
    # dissolve according to the clusters
    cluster_names = []
    for _, (ncessch_id, *_) in gdf.iterrows():
        s_id = int(ncessch_id)
        try:
            cluster_name = clusters[s_id]
        except KeyError:
            print(
                f"warning: could not find {s_id} in {focal_district_id} interdistrict sim"
            )
            cluster_name = "not found"
            # raise
        cluster_names.append(cluster_name)
    gdf["cluster"] = cluster_names
    gdf = gdf.dissolve(by="cluster", as_index=False)
    return gdf


# _palette = ["#4477AA", "#EE6677", "#228833", "#CCBB44"]
# _palette = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"]
_palette = ["#EE7733", "#0077BB", "#33BBEE", "#EE3377", "#CC3311", "#009988"]
assert len(_palette) >= 4, "four color map theorem"


@st.cache_data(ttl=3600, show_spinner="Coloring this map...")
def school_colors_v0(schools: tuple[int, ...]) -> dict[int, str]:
    """Broken for districts with discontiguous school zones

    (because this function employs the four color map theorem)
    """
    if len(schools) <= len(_palette):
        return dict(zip(schools, _palette))
    adj = {}
    seen = set()
    for school in schools:
        district_id = SCHOOL_ID_TO_DISTRICT_ID[school]
        if district_id in seen:
            continue
        seen.add(district_id)
        try:
            partial_adj = DISTRICT_ADJACENCY_MAPS[district_id]
        except KeyError:
            print("TODO - adj >200")
            partial_adj = {}
        adj.update(partial_adj)  # interdistrict.....
    possible_colors = {s: _palette.copy() for s in schools}
    random.seed(23)
    for palette in possible_colors.values():
        random.shuffle(palette)
    sorted_schools = sorted(
        schools, key=lambda s: len(adj.get(f"{s:012d}", ())), reverse=True
    )
    colors = {}
    for school in sorted_schools:
        try:
            color = possible_colors[school][0]
        except IndexError:  # ??
            color = "#BBBBBB"
        colors[school] = color
        for neighbor in adj.get(f"{school:012d}", ()):
            if neighbor == school:
                continue
            if color in possible_colors[int(neighbor)]:
                possible_colors[int(neighbor)].remove(color)
    return colors


@st.cache_data(ttl=3600, show_spinner="Coloring this map...")
def school_colors_v1(
    schools: tuple[int, ...],
    schools_from_neighboring_districts: tuple[int, ...],
) -> dict[int, str]:
    colors = {}
    for school in schools:
        random.seed(school)
        colors[school] = "#" + "%06x" % random.randint(0, 0xFFFFFF)
    for school in schools_from_neighboring_districts:
        random.seed(school)
        rgb = "#" + "%06x" % random.randint(0, 0xFFFFFF)
        r, g, b = _color_hex_to_rgb(rgb)
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        r, g, b = colorsys.hsv_to_rgb(h, s / 100, v)
        colors[school] = _color_rgb_to_hex(r, g, b)
    return colors


# _demographic_color = "#0000FF"
_demographic_color = "#6c24ff"


def _color_hex_to_rgb(hexcode: str) -> tuple[float, float, float]:
    hexcode = hexcode.lstrip("#")
    r = int(hexcode[0:2], 16) / 255
    g = int(hexcode[2:4], 16) / 255
    b = int(hexcode[4:6], 16) / 255
    return r, g, b


def _color_rgb_to_hex(r: float, g: float, b: float) -> str:
    r_ = round(r * 255)
    g_ = round(g * 255)
    b_ = round(b * 255)
    return f"#{r_:02x}{g_:02x}{b_:02x}"


def school_colors_demographics_v0(
    district: District,
    clusters: None | dict[int, str] = None,
    *,
    demo: str = "color",
    after: bool = False,
) -> dict[int, str]:
    """..."""
    cluster_population = {}
    if clusters is not None:
        for s_id, cluster_name in clusters.items():
            if cluster_name not in cluster_population:
                cluster_population[cluster_name] = Population.zero()
            pop = (
                district.schools_in_simulation[s_id].population_after
                if after
                else district.schools_in_simulation[s_id].population_before
            )
            cluster_population[cluster_name] += pop
    demographic_color = _demographic_color.lstrip("#")
    r, g, b = _color_hex_to_rgb(_demographic_color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    colors = {}
    for s_id, school in district.schools_in_simulation.items():
        population = school.population_after if after else school.population_before
        if clusters is not None:
            try:
                population = cluster_population[clusters[s_id]]
            except KeyError:
                print(f"info: ghost school: {s_id}")
                continue
        if population.total in (0, None):
            colors[s_id] = "#BBBBBB"
            continue
        demo_proportion = population[demo] / population.total
        new_s = s * demo_proportion
        r, g, b = colorsys.hsv_to_rgb(h, new_s, v)
        colors[s_id] = _color_rgb_to_hex(r, g, b)

    return colors


def make_map_base_layer(district: District):
    """Create the base folium map without any of our layers on it"""
    # center the map on the district
    try:
        location = DISTRICT_ID_TO_CENTROID[district.nces_id]
    except KeyError:
        location = STATE_TO_CENTROID[district.state]
    map = folium.Map(location=location, tiles=None, zoom_start=9, prefer_canvas=True)

    folium.TileLayer(
        show=True,
        overlay=False,
        control=False,
        tiles="CartoDB positron",
        name="Base map",
    ).add_to(map)

    return map


def render_map(existing=None, rendered=None, *, height=MAP_HEIGHT_PIXELS):
    if rendered is None:
        figure = folium.Figure()
        if existing is not None:
            figure.add_child(existing)
        rendered = figure.render()
    components.html(rendered, height=height + 10)
    return rendered


class MapLayer(Enum):
    CATCHMENT_AREAS_BEFORE = "<font color='#6EA6CD' style='font-weight: lighter;'><b>Status quo</b>: School catchment zones</font>"
    CATCHMENT_AREAS_AFTER = "<font color='#F67E4B' style='font-weight: lighter;'><b>Post-mergers</b>: School catchment zones</font>"
    DEMOGRAPHICS_BEFORE = "<font color='#6EA6CD' style='font-weight: lighter;'>Status quo: {demographic}</font>"
    DEMOGRAPHICS_AFTER = "<font color='#F67E4B' style='font-weight: lighter;'>Post-mergers: {demographic}</font>"


def _demographics_tooltip(
    population_before: Population, population_after: None | Population = None
) -> str:
    """HTML was a mistake"""
    if population_before == population_after:
        return _demographics_tooltip(population_before)
    _style = lambda alignment: f"padding: 0.1ch 0.6ch; text-align: {alignment};"
    _right = _style("right")
    _center = _style("center")
    _left = _style("left")
    tooltips = [f"<table>"]
    if population_after is None:
        tooltips.append(
            f"""<tr style='border-bottom: 1px solid black;'><th>&nbsp;</th><th style='{_center}'>Count</th></tr>"""
        )
        for demo, demo_name in DEMOGRAPHICS.items():
            pb = round(population_before[demo])
            tooltip = f"<tr><td style='{_right}'>{demo_name}</td><td style='{_center}'><b>{pb:,}</b>"
            if demo != "total" and (pbt := population_before.total) not in (0, None):
                tooltip += f" ({pb/pbt:.1%})"
            tooltip += "</td>"
            tooltips.append(tooltip)
    else:
        tooltips.append(
            f"<tr style='border-bottom: 1px solid black;'><th>&nbsp;</th><th style='{_center}'>Before</th><th style='{_center}'>After</th></tr>"
        )
        for demo, demo_name in DEMOGRAPHICS.items():
            pb = round(population_before[demo])
            pa = round(population_after[demo])
            if demo != "total":
                tooltip = (
                    f"<tr><td style='{_right}'>{demo_name}</td><td style='{_center}'>"
                )
                if (pbt := population_before.total) not in (0, None):
                    tooltip += f"<b>{pb:,}</b> ({pb/pbt:.1%})</td>"
                else:
                    tooltip += f"<b>{pb:,}</b></td>"
                tooltip += f"<td style='{_center}'>"
                if (pat := population_after.total) not in (0, None):
                    tooltip += f"<b>{pa:,}</b> ({pa/pat:.1%})</td>"
                else:
                    tooltip += f"<b>{pa:,}</b></td>"
                tooltip += "</tr>"
            else:
                tooltip = f"<tr><td style='{_right}'>{demo_name}</td><td style='{_center}'><b>{pb:,}</td><td style='{_center}'><b>{pa:,}</b></td></tr>"
            tooltips.append(tooltip)
    tooltips.append("</table>")
    tooltips_str = "\n".join(tooltips)
    # print(tooltips_str, "\n\n")
    return tooltips_str


# @st.cache_data(ttl=3600, show_spinner="Remembering info about this district...")
def make_a_layer(
    district: District,
    layer_kind: MapLayer,
    outline_these_schools: Collection[int] | None = None,
    *,
    demo: str = "color",
    is_default=False,
) -> folium.GeoJson:
    # take note of all involved districts, which is particularly important for
    # interdistrict simulations
    neighboring_district_ids = set()
    for school in district.schools_from_neighboring_districts.values():
        neighboring_district_ids.add(school.district_id)
    # load geodata
    clusters = None
    cluster_them_up = layer_kind in (MapLayer.CATCHMENT_AREAS_AFTER,)
    if cluster_them_up:
        clusters = {}
        for cluster_name, schools in district.clusters_in_simulation.items():
            for school in schools:
                clusters[school.ncessch_id] = cluster_name
    gdf = _read_geodata(
        district.state, district.nces_id, neighboring_district_ids, clusters
    ).copy()
    # add the tooltips
    tooltips = []
    if cluster_them_up:  # cluster-based tooltips
        for cluster_name in gdf["cluster"].astype(str):
            if cluster_name not in district.clusters_in_simulation:
                tooltips.append("not found")
                continue
            cluster = district.clusters_in_simulation[cluster_name]
            school_names = []
            population = Population.zero()
            for school in cluster:
                population += school.population_before
                school_name = f"<b>{school.school_name}</b><br>&emsp;&emsp;(NCES ID: {school.ncessch_id})"
                if school.district_id != district.nces_id:
                    district_name = DISTRICTS_IN_STATE[district.state].get(
                        school.district_id, f"NCES ID: {school.district_id:07d}"
                    )
                    school_name += (
                        f"<br>&emsp;&emsp;(from neighboring district, {district_name})"
                    )
                school_names.append(school_name)
            school_names_str = "<br>".join(school_names)
            demographics_info = _demographics_tooltip(population)
            tooltip = f"{school_names_str}<br><br>{demographics_info}"
            tooltips.append(tooltip)
    else:  # school-based tooltips
        for s_id in gdf["ncessch"].astype(int):
            school = district.schools_in_simulation[s_id]
            school_name = f"<b>{school.school_name}</b><br>&emsp;&emsp;(NCES ID: {school.ncessch_id})"
            if school.district_id != district.nces_id:
                district_name = DISTRICTS_IN_STATE[district.state].get(
                    school.district_id, f"NCES ID: {school.district_id:07d}"
                )
                school_name += (
                    f"<br>&emsp;&emsp;(from neighboring district, {district_name})"
                )
            demographics_info = _demographics_tooltip(
                school.population_before, school.population_after
            )
            tooltip = f"{school_name}<br><br>{demographics_info}"
            tooltips.append(tooltip)
    gdf["tooltip"] = tooltips
    gdf["weight"] = 0.1
    if outline_these_schools:
        gdf["weight"] = [
            (3.0 if int(nces_id) in outline_these_schools else 0.1)
            for nces_id in gdf["ncessch"]
        ]
    # https://python-visualization.github.io/folium/latest/user_guide/geojson/geojson_popup_and_tooltip.html
    tooltip = folium.features.GeoJsonTooltip(
        fields=["tooltip"],
        labels=False,
        localize=True,
        sticky=False,
        max_width=400,  # does not seem to work?
        style=("background-color: #f0efefBB;" "color: #000000;"),
    )
    # decide how to color the layer
    match layer_kind:
        case MapLayer.CATCHMENT_AREAS_BEFORE:
            colors = school_colors_v1(
                tuple(s_id for s_id in district.schools),
                tuple(s_id for s_id in district.schools_from_neighboring_districts),
            )
        case MapLayer.CATCHMENT_AREAS_AFTER:
            colors = school_colors_v1(
                tuple(s_id for s_id in district.schools),
                tuple(s_id for s_id in district.schools_from_neighboring_districts),
            )
        case MapLayer.DEMOGRAPHICS_BEFORE:
            colors = school_colors_demographics_v0(district, demo=demo)
        case MapLayer.DEMOGRAPHICS_AFTER:
            colors = school_colors_demographics_v0(
                district, clusters, demo=demo, after=True
            )
    style_function = lambda feature: {
        "fillColor": colors.get(int(feature["properties"]["ncessch"]), "#BBBBBB"),
        "fillOpacity": 0.75,
        "color": "#574a80",  # colors.get(int(feature["properties"]["ncessch"]), "#BBBBBB"),
        "dashArray": "10,10",
        "weight": float(feature["properties"]["weight"]),
    }
    # formally create the layer
    demographic = "Students of color"
    if demo in DEMOGRAPHICS:
        demographic = DEMOGRAPHICS[demo] + " students"
    layer_name = layer_kind.value.format(demographic=demographic)
    layer = folium.GeoJson(
        gdf,
        overlay=False,
        style_function=style_function,
        name=layer_name,
        tooltip=tooltip,
        highlight_function=lambda feature: {
            "fillColor": colors.get(int(feature["properties"]["ncessch"]), "#BBBBBB"),
            "fillOpacity": 0.7,
            # "weight": float(feature["properties"]["weight"]),
        },
        show=is_default,
        zoom_on_click=True,
    )
    # okay done
    return layer


def make_outline_layer(
    nces_ids: Collection[int], color: str | None = None
) -> folium.FeatureGroup | folium.GeoJson:
    # prep layer group object
    layer = folium.FeatureGroup(
        name="+".join([str(i) for i in nces_ids]),
        overlay=True,
        control=False,
        show=True,
    )
    # get the outline data
    gdf = gpd.GeoDataFrame({"nces_id": [], "geometry": []})
    for nces_id in nces_ids:
        if len(str(nces_id)) > 7:  # it's a school ID
            nces_id_str = f"{nces_id:012d}"
            district_id_str = nces_id_str[:7]
        else:  # it's a district ID
            nces_id_str = f"{nces_id:07d}"
            district_id_str = nces_id_str
        state = DISTRICT_ID_TO_STATE_BACKUP[int(district_id_str)]
        path = (
            Path("data/school_attendance_boundaries/outlines")
            / f"{state}/{district_id_str}.csv"
        )
        gdf_partial = load_gpd(path)
        gdf_partial = gdf_partial[gdf_partial["nces_id"].astype(str) == nces_id_str]
        gdf = pd.concat([gdf, gdf_partial], ignore_index=True)

    return folium.GeoJson(
        gdf,
        overlay=False,
        style_function=lambda feature: {
            "color": color or "#888888",
            "weight": 3,
            "interactive": False,
            "fillOpacity": 0,
            "opacity": 0.6,
        },
        show=True,
        interactive=False,
        control=False,
    )


def make_school_markers_layer(
    district: District,
    mark_these_schools: Collection[int] | None = None,
    *,
    except_=False,
) -> folium.FeatureGroup:
    layer = folium.FeatureGroup(
        name="<font style='font-weight: lighter;'>Show all school markers</font>",
        show=not except_,
        control=except_,
    )
    colors = school_colors_v1(
        tuple(s_id for s_id in district.schools),
        tuple(s_id for s_id in district.schools_from_neighboring_districts),
    )
    _style = (
        lambda alignment: f"font-size: small; padding: 0.2ch 0.6ch; text-align: {alignment};"
    )
    _right = _style("right")
    _center = _style("center")
    _left = _style("left")
    for s_id, school in district.schools_in_simulation.items():
        if mark_these_schools is not None:
            if (s_id not in mark_these_schools) ^ except_:
                continue
        if school.location is None:
            continue
        if school.grade_span_before != school.grade_span_after:
            grade_table = f"""<table style="margin: auto;"><tr><th colspan=2 style='border-bottom: 1px solid black; {_center}'>Grades offered</th></tr>"""
            grade_span_before = _describe_grade_span(
                school.grade_span_before, short=True
            )
            grade_span_after = _describe_grade_span(school.grade_span_after, short=True)
            grade_table += f"""<tr><td style="{_right}">Before</td><td style="{_center}">{grade_span_before}</td></tr>"""
            grade_table += f"""<tr><td style="{_right}">After</td><td style="{_center}">{grade_span_after}</td></tr>"""
            grade_table += "</table>"
            height = "135"
        else:
            grade_span = _describe_grade_span(school.grade_span_before).capitalize()
            grade_span = grade_span.replace("through", "through<br>")
            grade_table = grade_span
            height = "90"
        html = (
            f"""<div style="font-family: 'Helvetica Neue', Arial, Helvetica, sans-serif; text-align: center;">"""
            f"""<font style="font-size: smaller;"><b>{school.school_name}</b><br><br></font>"""
            f"""<font style="font-size: smaller;">{grade_table}</font>"""
            f"""</div>"""
        )
        iframe = folium.IFrame(html=html, height=height, width="200")
        popup = folium.Popup(iframe)
        marker = folium.Marker(
            school.location,
            icon=folium.Icon(color="lightgreen", icon_color=colors[s_id]),
            popup=popup,
            # tooltip=html,
        )
        marker.add_to(layer)
    return layer


def draw(
    where,
    district: District,
    *,
    focal_demo="color",
    cluster: None | Collection[School] = None,
):
    """Render map of a district

    Args:
       where: streamlit object to draw map to (usually ``st``)
       district: district to center on
    """
    with where.container():
        spinner_placeholder = components.html("", height=100)
        message_placeholder = st.empty()
        map_placeholder = st.empty()
        try:
            with spinner_placeholder:
                with st.spinner(f"Drawing the map..."):
                    # Render base map first
                    base_map = make_map_base_layer(district)
                    with map_placeholder:
                        render_map(base_map)

                    # Folium search bar, so users add their house, etc., as a waypoint
                    folium.plugins.Geocoder().add_to(base_map)

                    # Add the layers
                    these = [s.nces_id for s in cluster] if cluster else ()
                    before_layer = make_a_layer(
                        district, MapLayer.CATCHMENT_AREAS_BEFORE, these
                    )
                    before_layer.add_to(base_map)
                    demographics_layer = make_a_layer(
                        district, MapLayer.DEMOGRAPHICS_BEFORE, these
                    )
                    demographics_layer.add_to(base_map)
                    if focal_demo != "color":
                        extra_layer = make_a_layer(
                            district,
                            MapLayer.DEMOGRAPHICS_BEFORE,
                            these,
                            demo=focal_demo,
                        )
                        extra_layer.add_to(base_map)

                    clusters_layer = make_a_layer(
                        district, MapLayer.CATCHMENT_AREAS_AFTER, these
                    )
                    clusters_layer.add_to(base_map)
                    clustered_demographics_layer = make_a_layer(
                        district, MapLayer.DEMOGRAPHICS_AFTER, these, is_default=True
                    )
                    clustered_demographics_layer.add_to(base_map)
                    if focal_demo != "color":
                        clustered_extra_layer = make_a_layer(
                            district,
                            MapLayer.DEMOGRAPHICS_AFTER,
                            these,
                            demo=focal_demo,
                        )
                        clustered_extra_layer.add_to(base_map)

                    # district outlines
                    district_ids = {district.nces_id} | {
                        s.district_id
                        for s in district.schools_from_neighboring_districts.values()
                    }
                    outlines = make_outline_layer(district_ids, color="#325ea8")
                    outlines.add_to(base_map)
                    # school outlines
                    # if cluster:
                    #    school_ids = set(s.nces_id for s in cluster)
                    #    outlines = make_outline_layer(school_ids, color="#a83268")
                    #    outlines.add_to(base_map)

                    school_markers = make_school_markers_layer(district, these or None)
                    school_markers.add_to(base_map)
                    school_markers = make_school_markers_layer(
                        district, these or None, except_=True
                    )
                    school_markers.add_to(base_map)

                    # Render the map
                    base_map.fit_bounds(base_map.get_bounds(), padding=(5, 5))
                    # https://python-visualization.github.io/folium/latest/reference.html#folium.map.LayerControl
                    # https://leafletjs.com/reference.html#control-layers
                    folium.LayerControl(
                        collapsed=False,
                        hideSingleBase=True,
                        sortLayers=False,
                    ).add_to(base_map)
                    with map_placeholder:
                        render_map(base_map, None)
                    # with message_placeholder:
                    #    st.markdown("Use this map to explore the proposed school mergers. You can switch between different views using the layer control at the upper right.")
                    spinner_placeholder = components.html("", height=100)
        except:
            with map_placeholder:
                st.empty()
            raise
