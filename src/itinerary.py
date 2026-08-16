# pyrefly: ignore [missing-import]
"""
Itinerary Generation Module for Travel Destination Recommendation System.

Selects geographically feasible attractions from a Top-N candidate pool,
groups them into travel days using K-Means, enforces pace-aware daily capacity,
and orders within-day stops using a Greedy Nearest-Neighbor Haversine heuristic.
"""

import math
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, Union, cast, overload
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("itinerary")

MAX_STOPS_PER_DAY = 3

EXCLUSION_REASON_LABELS: Dict[str, str] = {
    "missing_coordinates": "Missing map coordinates",
    "geographically_incompatible": "Too far from your top destination region",
    "capacity_overflow": "Exceeded daily capacity for this trip",
    "not_selected": "Lower priority among geographically compatible options",
}

EXCLUDED_OUTPUT_COLS = [
    "attraction_uid",
    "attraction_name",
    "city",
    "rank",
    "recommendation_score",
    "exclusion_reason",
    "exclusion_reason_label",
]


def load_coordinates(coordinates_path: str) -> pd.DataFrame:
    """
    Loads coordinates dataset from a CSV file.

    Args:
        coordinates_path (str): Path to coordinates.csv.

    Returns:
        pd.DataFrame: Loaded coordinates data.
    """
    path = Path(coordinates_path)
    if not path.is_file():
        raise FileNotFoundError(f"Coordinates file not found at: {coordinates_path}")
    return pd.read_csv(path)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates geodetic distance in kilometers between two coordinates.

    Args:
        lat1 (float): Latitude of first point.
        lon1 (float): Longitude of first point.
        lat2 (float): Latitude of second point.
        lon2 (float): Longitude of second point.

    Returns:
        float: Distance in kilometers.
    """
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return c * 6371.0


def _recommendation_score(row: pd.Series) -> float:
    if "similarity_score" in row.index and pd.notna(row["similarity_score"]):
        return float(row["similarity_score"])
    if "predicted_rating" in row.index and pd.notna(row["predicted_rating"]):
        return float(row["predicted_rating"])
    return 0.0


def _build_excluded_df(df: pd.DataFrame, reason_code: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=EXCLUDED_OUTPUT_COLS)

    excluded = df.copy()
    excluded["recommendation_score"] = excluded.apply(_recommendation_score, axis=1)
    excluded["exclusion_reason"] = reason_code
    excluded["exclusion_reason_label"] = EXCLUSION_REASON_LABELS[reason_code]

    for col in EXCLUDED_OUTPUT_COLS:
        if col not in excluded.columns:
            excluded[col] = np.nan

    return excluded[EXCLUDED_OUTPUT_COLS].reset_index(drop=True)


def _join_coordinates(
    recommendations_df: pd.DataFrame, coordinates_df: pd.DataFrame
) -> pd.DataFrame:
    """Left-join recommendations with coordinates and province; keeps unmatched rows."""
    if recommendations_df.empty:
        return recommendations_df.copy()

    recs = recommendations_df.copy()
    if "province" in recs.columns:
        recs = recs.drop(columns=["province"])

    coords = coordinates_df.copy()
    coords["attraction_uid"] = [
        f"{str(name).strip()} ({str(city).strip()}, {str(province).strip()})"
        for name, city, province in zip(
            coords["attraction_name"], coords["city"], coords["province"]
        )
    ]

    return pd.merge(
        recs,
        coords[["attraction_uid", "latitude", "longitude", "province"]],
        on="attraction_uid",
        how="left",
    )


def attach_coordinates(
    recommendations_df: pd.DataFrame, coordinates_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Joins the recommendation dataframe with coordinates on attraction_uid.
    Drops rows with missing coordinates for downstream clustering compatibility.
    """
    merged = _join_coordinates(recommendations_df, coordinates_df)

    unmatched = merged[merged["latitude"].isna() | merged["longitude"].isna()]
    if not unmatched.empty:
        unmatched_names = unmatched["attraction_name"].tolist()
        logger.warning(
            f"Failed to match coordinates for the following attractions: {unmatched_names}"
        )
        merged = merged[merged["latitude"].notna() & merged["longitude"].notna()].copy()

    return merged


def select_geographic_candidates(
    candidates_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Selects geographically compatible attractions from the candidate pool.

    Uses the Rank 1 attraction as the geographic anchor. Same-city candidates
    are strongly preferred, nearby cities may be included using relative
    distance thresholds derived from the candidate set, and clearly distant
    cities are excluded regardless of recommendation rank.
    """
    if candidates_df.empty:
        return candidates_df.copy(), pd.DataFrame(columns=EXCLUDED_OUTPUT_COLS)

    df = candidates_df.copy().reset_index(drop=True)
    anchor_idx = cast(Any, df["rank"].idxmin())
    anchor = df.loc[anchor_idx]
    if isinstance(anchor, pd.DataFrame):
        anchor = anchor.iloc[0]

    anchor_city = str(anchor["city"]).strip()
    anchor_province = str(anchor.get("province", "")).strip()
    anchor_lat = float(anchor["latitude"])
    anchor_lon = float(anchor["longitude"])

    df["dist_to_anchor"] = df.apply(
        lambda row: haversine_distance(
            anchor_lat, anchor_lon, float(row["latitude"]), float(row["longitude"])
        ),
        axis=1,
    )

    all_dists = cast(np.ndarray, df["dist_to_anchor"].to_numpy())
    p50 = float(np.percentile(all_dists, 50)) if len(all_dists) else 0.0
    p75 = float(np.percentile(all_dists, 75)) if len(all_dists) else 0.0

    same_prov_mask = df["province"].astype(str).str.strip() == anchor_province
    prov_dists = cast(np.ndarray, df.loc[same_prov_mask, "dist_to_anchor"].to_numpy())
    prov_p75 = float(np.percentile(prov_dists, 75)) if len(prov_dists) > 1 else p75

    same_city_mask = df["city"].astype(str).str.strip() == anchor_city
    t1_dists = cast(np.ndarray, df.loc[same_city_mask, "dist_to_anchor"].to_numpy())
    t1_radius = float(np.max(t1_dists)) if len(t1_dists) > 0 else 0.0
    t1_radius = max(t1_radius, 1e-6)

    nearest_same_prov_city_dist = p50
    other_city_centroid_dists: List[float] = []

    same_prov_df = df.loc[same_prov_mask].copy()
    same_prov_df["city_clean"] = same_prov_df["city"].astype(str).str.strip()

    for city, group in same_prov_df.groupby("city_clean"):
        if city == anchor_city:
            continue
        clat = group["latitude"].mean()
        clon = group["longitude"].mean()
        other_city_centroid_dists.append(
            haversine_distance(anchor_lat, anchor_lon, clat, clon)
        )
    if other_city_centroid_dists:
        nearest_same_prov_city_dist = min(other_city_centroid_dists)

    selected_indices: List[Any] = []
    excluded_indices: List[Any] = []

    t1_rows: List[Tuple[Any, float]] = []
    t2_rows: List[Tuple[Any, float]] = []

    for idx, row in df.iterrows():
        city = str(row["city"]).strip()
        province = str(row.get("province", "")).strip()
        dist = float(row["dist_to_anchor"])
        rank = float(row["rank"])

        if city == anchor_city:
            t1_rows.append((idx, rank))
            continue

        is_distant = (
            (province != anchor_province and dist > p75)
            or (dist > 2.0 * t1_radius and dist > p50)
        )
        if is_distant:
            excluded_indices.append(idx)
            continue

        is_nearby = (
            (province == anchor_province and dist <= prov_p75)
            or dist <= p50
            or dist <= nearest_same_prov_city_dist
        )
        if is_nearby:
            t2_rows.append((idx, rank))
        else:
            excluded_indices.append(idx)

    t1_rows.sort(key=lambda item: item[1])
    t2_rows.sort(key=lambda item: item[1])
    selected_indices = [idx for idx, _ in t1_rows] + [idx for idx, _ in t2_rows]

    selected_df = df.loc[selected_indices].drop(columns=["dist_to_anchor"]).copy()
    excluded_df = _build_excluded_df(
        df.loc[excluded_indices].drop(columns=["dist_to_anchor"]),
        "geographically_incompatible",
    )
    return selected_df, excluded_df


def assign_days(candidates_df: pd.DataFrame, num_days: int) -> pd.DataFrame:
    """
    Groups attractions into geographically coherent travel days using K-Means.

    K-Means runs only on the geographically selected subset. Empty days are
    permitted when the requested trip length exceeds the number of selected stops.
    """
    if candidates_df.empty:
        df_out = candidates_df.copy()
        df_out["day"] = []
        df_out.attrs["kmeans_cluster_sizes"] = [0] * num_days
        return df_out

    if num_days < 1:
        raise ValueError("Number of travel days must be at least 1.")

    n = len(candidates_df)
    df_out = candidates_df.copy()

    if num_days == 1 or n == 1:
        df_out["day"] = 1
        df_out.attrs["kmeans_cluster_sizes"] = [n] + [0] * (num_days - 1)
        return df_out

    effective_clusters = min(num_days, n)
    coords = np.column_stack((df_out["latitude"].to_numpy(), df_out["longitude"].to_numpy()))
    kmeans = KMeans(
        n_clusters=effective_clusters,
        random_state=42,
        n_init=10,
    )
    labels = kmeans.fit_predict(coords)

    cluster_priority = (
        pd.DataFrame({"label": labels, "rank": df_out["rank"].to_numpy()})
        .groupby("label")["rank"]
        .min()
        .sort_values()
    )
    label_to_day = {
        int(label): day for day, label in enumerate(cluster_priority.index, start=1)
    }
    df_out["day"] = [label_to_day[int(label)] for label in labels]

    df_out.attrs["kmeans_cluster_sizes"] = [
        int((df_out["day"] == day).sum()) for day in range(1, num_days + 1)
    ]
    return df_out


def _day_centroid(day_df: pd.DataFrame) -> Tuple[float, float]:
    return day_df["latitude"].mean(), day_df["longitude"].mean()


def _swap_cost(row: pd.Series, target_day_df: pd.DataFrame) -> float:
    if target_day_df.empty:
        return 0.0
    tlat, tlon = _day_centroid(target_day_df)
    return haversine_distance(float(row["latitude"]), float(row["longitude"]), tlat, tlon)


def _balance_pace(
    df_assigned: pd.DataFrame, travel_pace: str, num_days: int
) -> pd.DataFrame:
    """Adjusts day assignments toward pace targets using local swaps."""
    if df_assigned.empty:
        return df_assigned

    pace = travel_pace.strip().title()
    df = df_assigned.copy()
    max_iterations = len(df) * 2

    for _ in range(max_iterations):
        counts = df.groupby("day").size().reindex(range(1, num_days + 1), fill_value=0)
        changed = False

        if pace == "Relaxed":
            heavy_days = [day for day, count in counts.items() if count >= 3]
            light_days = [day for day, count in counts.items() if count <= 1]
            for heavy_day in heavy_days:
                for light_day in light_days:
                    heavy_df = df[df["day"] == heavy_day].sort_values("rank", ascending=False)
                    light_df = df[df["day"] == light_day]
                    if heavy_df.empty or len(light_df) >= 2:
                        continue
                    candidate_idx = cast(Any, heavy_df.index[0])
                    candidate = df.loc[candidate_idx]
                    if _swap_cost(candidate, light_df) <= _swap_cost(candidate, heavy_df.iloc[:-1]):
                        df.at[candidate_idx, "day"] = cast(int, light_day)
                        changed = True
                        break
                if changed:
                    break

        elif pace == "Balanced":
            # Balance from heavy days (>= 3 stops) to light days (<= 1 stops) to prefer around 2 stops/day
            heavy_days = [day for day, count in counts.items() if count >= 3]
            light_days = [day for day, count in counts.items() if count <= 1]
            for heavy_day in heavy_days:
                for light_day in light_days:
                    heavy_df = df[df["day"] == heavy_day].sort_values("rank", ascending=False)
                    light_df = df[df["day"] == light_day]
                    if heavy_df.empty or len(light_df) >= 3:
                        continue
                    candidate_idx = cast(Any, heavy_df.index[0])
                    candidate = df.loc[candidate_idx]
                    if _swap_cost(candidate, light_df) <= _swap_cost(candidate, heavy_df.iloc[:-1]):
                        df.at[candidate_idx, "day"] = cast(int, light_day)
                        changed = True
                        break
                if changed:
                    break

        elif pace == "Packed":
            light_days = [day for day, count in counts.items() if count == 1]
            donor_days = [day for day, count in counts.items() if 1 < count < MAX_STOPS_PER_DAY]
            for light_day in light_days:
                for donor_day in donor_days:
                    light_df = df[df["day"] == light_day]
                    donor_df = df[df["day"] == donor_day].sort_values("rank", ascending=False)
                    if donor_df.empty:
                        continue
                    candidate_idx = cast(Any, donor_df.index[0])
                    candidate = df.loc[candidate_idx]
                    if _swap_cost(candidate, light_df) <= _swap_cost(
                        candidate, donor_df.iloc[:-1]
                    ):
                        df.at[candidate_idx, "day"] = cast(int, light_day)
                        changed = True
                        break
                if changed:
                    break

        if not changed:
            break

    return df


def enforce_day_capacity(
    df_assigned: pd.DataFrame,
    travel_pace: str,
    num_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Enforces the hard 1-3 stops per day limit and applies pace-aware balancing.
    """
    if df_assigned.empty:
        return df_assigned.copy(), pd.DataFrame(columns=EXCLUDED_OUTPUT_COLS)

    excluded_parts: List[pd.DataFrame] = []
    df = df_assigned.copy()

    pace = travel_pace.strip().title()
    limit = 2 if pace == "Relaxed" else 3

    overflow_indices: List[Any] = []
    for day in sorted(df["day"].unique()):
        day_df = df[df["day"] == day].sort_values("rank")
        if len(day_df) > limit:
            overflow = day_df.iloc[limit:]
            overflow_indices.extend(overflow.index.tolist())
            excluded_parts.append(_build_excluded_df(overflow, "capacity_overflow"))

    if overflow_indices:
        df = df.drop(index=overflow_indices)

    max_total = num_days * limit
    if len(df) > max_total:
        excess = df.sort_values("rank", ascending=False).head(len(df) - max_total)
        df = df.drop(index=excess.index)
        excluded_parts.append(_build_excluded_df(excess, "capacity_overflow"))

    df = _balance_pace(df, travel_pace, num_days)

    excluded_df = (
        pd.concat(excluded_parts, ignore_index=True)
        if excluded_parts
        else pd.DataFrame(columns=EXCLUDED_OUTPUT_COLS)
    )
    return df, excluded_df


def _correct_day_coherence(
    df_assigned: pd.DataFrame,
    travel_pace: str,
    num_days: int,
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """
    Detects obvious geographic outliers within each assigned day.
    Tries to move them to a more coherent day under capacity limits.
    If no coherent day fits, excludes them as geographically incompatible.
    """
    if df_assigned.empty or len(df_assigned) <= 1:
        return df_assigned.copy(), []

    df = df_assigned.copy()
    pace = travel_pace.strip().title()
    limit = 2 if pace == "Relaxed" else 3

    # 1. Compute a data-derived threshold based on pairwise distances of all selected attractions
    pairwise_dists = []
    coords = df[["latitude", "longitude"]].to_numpy()
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d = haversine_distance(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
            pairwise_dists.append(d)
    
    if len(pairwise_dists) > 0:
        # Use 75th percentile of all pairwise distances as a relative benchmark
        threshold = max(float(np.percentile(pairwise_dists, 75)), 15.0)
    else:
        threshold = 50.0

    excluded_parts: List[pd.DataFrame] = []
    
    # We will do up to 3 passes to allow re-assignments to settle
    for _ in range(3):
        counts = df.groupby("day").size().reindex(range(1, num_days + 1), fill_value=0)
        moved_or_excluded = False

        for day in sorted(df["day"].unique()):
            day_mask = df["day"] == day
            day_df = df[day_mask]
            if len(day_df) <= 1:
                continue

            # Identify if there is an outlier in this day
            outlier_idx = None
            max_avg_dist = -1.0

            for idx, row in day_df.iterrows():
                other_day_df = day_df.drop(index=idx)
                dists = [
                    haversine_distance(
                        float(row["latitude"]),
                        float(row["longitude"]),
                        float(other_row["latitude"]),
                        float(other_row["longitude"]),
                    )
                    for _, other_row in other_day_df.iterrows()
                ]
                avg_dist = float(np.mean(dists)) if dists else 0.0
                # An outlier has average distance to day mates > 1.5 * threshold
                if avg_dist > 1.5 * threshold and avg_dist > max_avg_dist:
                    max_avg_dist = avg_dist
                    outlier_idx = idx

            if outlier_idx is not None:
                # We found an outlier! Try to find another existing day
                outlier_row = df.loc[outlier_idx]
                best_new_day = None
                best_new_day_dist = float("inf")

                for target_day in range(1, num_days + 1):
                    if target_day == day:
                        continue
                    if counts[target_day] >= limit:
                        continue  # target day is at capacity

                    target_day_df = df[df["day"] == target_day]
                    if target_day_df.empty:
                        # An empty day is geographically reasonable to move to
                        if 0.0 < best_new_day_dist:
                            best_new_day = target_day
                            best_new_day_dist = 0.0
                        continue

                    # Calculate average distance to attractions in target day
                    t_dists = [
                        haversine_distance(
                            float(outlier_row["latitude"]),
                            float(outlier_row["longitude"]),
                            float(t_row["latitude"]),
                            float(t_row["longitude"]),
                        )
                        for _, t_row in target_day_df.iterrows()
                    ]
                    t_avg_dist = float(np.mean(t_dists)) if t_dists else 0.0
                    # The target day must be geographically reasonable (avg distance <= threshold)
                    if t_avg_dist <= threshold and t_avg_dist < best_new_day_dist:
                        best_new_day = target_day
                        best_new_day_dist = t_avg_dist

                if best_new_day is not None:
                    # Move the outlier to best_new_day
                    df.at[outlier_idx, "day"] = cast(int, best_new_day)
                    moved_or_excluded = True
                else:
                    # No suitable day found: exclude it as geographically incompatible
                    outlier_df = df.loc[[outlier_idx]]
                    df = df.drop(index=outlier_idx)
                    excluded_parts.append(
                        _build_excluded_df(outlier_df, "geographically_incompatible")
                    )
                    moved_or_excluded = True
                
                # Break to recompute counts and masks
                break

        if not moved_or_excluded:
            break

    return df, excluded_parts


def order_day(day_df: pd.DataFrame) -> pd.DataFrame:
    """
    Orders the attractions in a single day using Greedy Nearest-Neighbor heuristic.
    Starts with the highest-ranked (smallest rank value) attraction in the day.
    """
    if day_df.empty:
        return day_df.copy()

    unvisited = day_df.copy().reset_index(drop=True)
    start_idx = cast(Any, unvisited["rank"].idxmin())
    current_node_raw = unvisited.loc[start_idx]
    if isinstance(current_node_raw, pd.DataFrame):
        current_node_raw = current_node_raw.iloc[0]
    current_node = current_node_raw.to_dict()
    unvisited = unvisited.drop(start_idx)

    current_node["stop_order"] = 1
    current_node["distance_from_prev_km"] = 0.0
    path = [current_node]
    stop_idx = 2

    while not unvisited.empty:
        curr_lat = current_node["latitude"]
        curr_lon = current_node["longitude"]

        distances = []
        for idx, row in unvisited.iterrows():
            dist = haversine_distance(
                curr_lat, curr_lon, row["latitude"], row["longitude"]
            )
            distances.append((dist, row["rank"], idx))

        distances.sort(key=lambda item: (item[0], item[1]))
        min_dist, _, best_idx = distances[0]

        best_idx_cast = cast(Any, best_idx)
        current_node = unvisited.loc[best_idx_cast].to_dict()
        unvisited = unvisited.drop(best_idx_cast)
        current_node["stop_order"] = stop_idx
        current_node["distance_from_prev_km"] = min_dist
        path.append(current_node)
        stop_idx += 1

    return pd.DataFrame(path)


def print_itinerary_debug(
    itinerary_df: pd.DataFrame,
    top_n_candidates: int,
    num_days: int,
    cluster_sizes: list[int],
    selected_count: int,
    excluded_count: int,
) -> None:
    """Prints itinerary diagnostics for thesis validation and debugging."""
    print(f"Top-N candidates: {top_n_candidates}")
    print(f"Geographically selected: {selected_count}")
    print(f"Excluded from itinerary: {excluded_count}")
    print(f"Requested travel days: {num_days}")
    print(f"Final itinerary attractions: {len(itinerary_df)}")

    for day in range(1, num_days + 1):
        day_stops = itinerary_df[itinerary_df["day"] == day].sort_values("stop_order")
        attractions = day_stops["attraction_name"].astype(str).tolist()
        cities = day_stops["city"].astype(str).tolist()
        total_distance = day_stops["distance_from_prev_km"].sum() if not day_stops.empty else 0.0
        print(f"Day {day}:")
        print(f"- attractions: {attractions}")
        print(f"- cities: {cities}")
        print(f"- total Haversine distance: {total_distance:.2f} km")

    print(f"K-Means cluster sizes: {cluster_sizes}")


@overload
def build_itinerary(
    recommendations_df: pd.DataFrame,
    coordinates_df: pd.DataFrame,
    num_days: int,
    travel_pace: str = "Balanced",
    return_excluded: Literal[False] = False,
) -> pd.DataFrame:
    ...


@overload
def build_itinerary(
    recommendations_df: pd.DataFrame,
    coordinates_df: pd.DataFrame,
    num_days: int,
    travel_pace: str = "Balanced",
    return_excluded: Literal[True] = ...,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ...


@overload
def build_itinerary(
    recommendations_df: pd.DataFrame,
    coordinates_df: pd.DataFrame,
    num_days: int,
    travel_pace: str = "Balanced",
    return_excluded: bool = ...,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    ...


def build_itinerary(
    recommendations_df: pd.DataFrame,
    coordinates_df: pd.DataFrame,
    num_days: int,
    travel_pace: str = "Balanced",
    return_excluded: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generates a day-by-day itinerary from recommended attractions.

    Top-N recommendations are treated as candidates. Geographic feasibility is
    prioritised over recommendation rank when deciding inclusion. K-Means groups
    the selected subset into days, daily capacity is limited to 1-3 stops, and
    Haversine nearest-neighbour routing orders stops within each day.

    Args:
        recommendations_df: Recommendation output dataframe.
        coordinates_df: Coordinates CSV dataframe.
        num_days: Requested trip length.
        travel_pace: "Relaxed", "Balanced", or "Packed".
        return_excluded: If True, return (itinerary_df, excluded_df).

    Returns:
        Itinerary dataframe, or a tuple with excluded recommendations.
    """
    output_cols = [
        "day",
        "stop_order",
        "attraction_uid",
        "attraction_name",
        "city",
        "latitude",
        "longitude",
        "recommendation_score",
        "distance_from_prev_km",
    ]

    empty_excluded = pd.DataFrame(columns=EXCLUDED_OUTPUT_COLS)

    if recommendations_df.empty:
        empty_itinerary = pd.DataFrame(columns=output_cols)
        return (empty_itinerary, empty_excluded) if return_excluded else empty_itinerary

    if num_days <= 0:
        raise ValueError("Number of travel days must be at least 1.")

    pace = travel_pace.strip().title()
    if pace not in {"Relaxed", "Balanced", "Packed"}:
        raise ValueError("travel_pace must be one of: Relaxed, Balanced, Packed.")

    candidates_df = _join_coordinates(recommendations_df, coordinates_df)
    if "rank" not in candidates_df.columns:
        candidates_df["rank"] = range(1, len(candidates_df) + 1)

    if "attraction_uid" in candidates_df.columns:
        candidates_df = candidates_df.drop_duplicates(subset=["attraction_uid"], keep="first").copy()
    else:
        candidates_df = candidates_df.drop_duplicates().copy()

    candidates_df = candidates_df.reset_index(drop=True)
    top_n_candidates = len(candidates_df)
    missing_coords_mask = candidates_df["latitude"].isna() | candidates_df["longitude"].isna()
    excluded_parts: List[pd.DataFrame] = []

    if missing_coords_mask.any():
        excluded_parts.append(
            _build_excluded_df(
                candidates_df[missing_coords_mask],
                "missing_coordinates",
            )
        )

    matched_df = candidates_df[~missing_coords_mask].copy()
    coordinate_matched_count = len(matched_df)

    if matched_df.empty:
        empty_itinerary = pd.DataFrame(columns=output_cols)
        excluded_df = (
            pd.concat(excluded_parts, ignore_index=True)
            if excluded_parts
            else empty_excluded
        )
        return (empty_itinerary, excluded_df) if return_excluded else empty_itinerary

    selected_df, geo_excluded = select_geographic_candidates(matched_df)

    if not geo_excluded.empty:
        excluded_parts.append(geo_excluded)

    limit = 2 if pace == "Relaxed" else 3
    max_total = num_days * limit

    if len(selected_df) > max_total:
        keep_df = selected_df.sort_values("rank").head(max_total)
        trimmed_df = selected_df.sort_values("rank").iloc[max_total:]
        selected_df = keep_df.copy()
        excluded_parts.append(_build_excluded_df(trimmed_df, "not_selected"))

    if selected_df.empty:
        itinerary_df = pd.DataFrame(columns=output_cols)
        excluded_df = (
            pd.concat(excluded_parts, ignore_index=True)
            if excluded_parts
            else empty_excluded
        )
        print_itinerary_debug(
            itinerary_df,
            top_n_candidates,
            num_days,
            [],
            selected_count=0,
            excluded_count=len(excluded_df),
        )
        return (itinerary_df, excluded_df) if return_excluded else itinerary_df

    df_assigned = assign_days(selected_df, num_days)

    # Apply day-level geographic coherence correction
    df_assigned, coherence_excluded = _correct_day_coherence(df_assigned, pace, num_days)
    if coherence_excluded:
        excluded_parts.extend(coherence_excluded)

    df_assigned, capacity_excluded = enforce_day_capacity(
        df_assigned, pace, num_days
    )
    if not capacity_excluded.empty:
        excluded_parts.append(capacity_excluded)

    cluster_sizes = [int((df_assigned["day"] == day).sum()) for day in range(1, num_days + 1)]

    ordered_days = []
    for day in range(1, num_days + 1):
        day_data = df_assigned[df_assigned["day"] == day]
        if day_data.empty:
            continue
        ordered_days.append(order_day(day_data))

    if ordered_days:
        itinerary_df = pd.concat(ordered_days, ignore_index=True)
        itinerary_df["recommendation_score"] = itinerary_df.apply(
            _recommendation_score, axis=1
        )
        final_itinerary = itinerary_df[output_cols]
    else:
        final_itinerary = pd.DataFrame(columns=output_cols)

    final_itinerary.attrs["kmeans_cluster_sizes"] = cluster_sizes

    excluded_df = (
        pd.concat(excluded_parts, ignore_index=True)
        if excluded_parts
        else empty_excluded
    )

    print_itinerary_debug(
        final_itinerary,
        top_n_candidates,
        num_days,
        cluster_sizes,
        selected_count=len(selected_df),
        excluded_count=len(excluded_df),
    )

    return (final_itinerary, excluded_df) if return_excluded else final_itinerary
