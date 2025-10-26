import haversine as hs
import openrouteservice
import numpy as np

# coords = ((-71.0854323,42.3664655),(-71.1267582,42.3417178))
client = openrouteservice.Client(base_url="http://localhost:8080/ors/")


def get_distance_for_coord_pair(coords, unit=hs.Unit.MILES):
    return hs.haversine(
        coords[0],
        coords[1],
        unit=unit,
    )


def get_travel_time_for_coord_pair(coords):
    try:
        routes = client.request(
            url="/v2/matrix/driving-car",
            post_json={"locations": coords, "sources": [0], "destinations": [1]},
        )
        return routes["durations"][0][0]
    except Exception:
        return np.nan


def compute_distances_to_schools(df, school_lat="lat", school_long="long"):
    distances = []
    for i in range(0, len(df)):
        coords = (
            (
                float(df["block_centroid_lat"][i]),
                float(df["block_centroid_long"][i]),
            ),
            (float(df["lat"][i]), float(df["long"][i])),
        )
        distances.append(get_distance_for_coord_pair(coords))

    distances = np.array(distances, dtype=float).tolist()

    return distances


def compute_travel_times_to_schools(df):
    travel_times = []
    for i in range(0, len(df)):
        coords = (
            (float(df["block_centroid_long"][i]), float(df["block_centroid_lat"][i])),
            (float(df["long"][i]), float(df["lat"][i])),
        )
        travel_times.append(get_travel_time_for_coord_pair(coords))

    travel_times = np.array(travel_times, dtype=float).tolist()

    return travel_times


def compute_travel_info_to_schools(df):
    distances = []
    travel_times = []
    directions = []
    for i in range(0, len(df)):
        coords = (
            (float(df["block_centroid_long"][i]), float(df["block_centroid_lat"][i])),
            (float(df["long"][i]), float(df["lat"][i])),
        )

        try:
            curr = client.directions(coords)
        except Exception:
            distances.append(np.nan)
            travel_times.append(np.nan)
            directions.append("")
            continue

        # Store distance
        try:
            distances.append(curr["routes"][0]["summary"]["distance"])
        except Exception:
            distances.append(np.nan)

        try:
            travel_times.append(curr["routes"][0]["summary"]["duration"])
        except Exception:
            travel_times.append(np.nan)

        try:
            directions.append(curr["routes"][0]["segments"][0]["steps"])
        except Exception:
            directions.append("")

    return distances, travel_times, directions
