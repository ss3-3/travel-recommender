# Module Contribution: Soh Sook Shan (26WMR12721) & Goh Thong En (26WMR12713) - Itinerary Generation (joint work)

# pyrefly: ignore [missing-import]
"""
One-Day Itinerary Generation Module for Travel Destination Recommendation System.

Selects geographically suitable attractions from a Top-N recommendation
candidate pool and orders the selected stops using a Greedy Nearest-Neighbor
Haversine heuristic.
"""

import math
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, cast
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("itinerary")

EXCLUSION_REASON_LABELS: Dict[str, str] = {
    "missing_coordinates": "Missing map coordinates",
    "geographically_incompatible": "Outside the selected geographic group",
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

def build_one_day_itinerary(
    recommendations_df: pd.DataFrame,
    coordinates_df: pd.DataFrame,
    num_stops: int,
    return_excluded: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generates a one-day itinerary from Top-N recommended attractions.

    Top-N recommendations are treated as the candidate pool. The user selects
    the number of destinations to include in the itinerary, where the requested
    number must not exceed the number of recommendations.

    Geographic compatibility is considered before the selected attractions
    are ordered using Greedy Nearest-Neighbor.

    Args:
        recommendations_df: Recommendation output dataframe.
        coordinates_df: Coordinates CSV dataframe.
        num_stops: Requested number of destinations for the one-day itinerary.
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

    empty_excluded = pd.DataFrame(
        columns=EXCLUDED_OUTPUT_COLS
    )

    # Return an empty itinerary if no recommendations are available
    if recommendations_df.empty:
        empty_itinerary = pd.DataFrame(
            columns=output_cols
        )

        if return_excluded:
            return empty_itinerary, empty_excluded

        return empty_itinerary

    # Validate requested number of destinations
    if num_stops < 1:
        raise ValueError(
            "Number of itinerary destinations must be at least 1."
        )

    # Attach coordinates to Top-N recommendations
    candidates_df = _join_coordinates(
        recommendations_df,
        coordinates_df,
    )

    # Add recommendation rank if it is not available
    if "rank" not in candidates_df.columns:
        candidates_df["rank"] = range(
            1,
            len(candidates_df) + 1,
        )

    # Remove duplicate attractions
    if "attraction_uid" in candidates_df.columns:
        candidates_df = candidates_df.drop_duplicates(
            subset=["attraction_uid"],
            keep="first",
        ).copy()
    else:
        candidates_df = candidates_df.drop_duplicates().copy()

    candidates_df = candidates_df.reset_index(drop=True)

    top_n_candidates = len(candidates_df)

    # M cannot exceed Top-N
    if num_stops > top_n_candidates:
        raise ValueError(
            "Number of itinerary destinations cannot exceed "
            "the number of Top-N recommendations."
        )

    excluded_parts: List[pd.DataFrame] = []

    # Identify recommendations with missing coordinates
    missing_coords_mask = (
        candidates_df["latitude"].isna()
        | candidates_df["longitude"].isna()
    )

    if missing_coords_mask.any():
        excluded_parts.append(
            _build_excluded_df(
                candidates_df[missing_coords_mask],
                "missing_coordinates",
            )
        )

    # Keep only attractions with valid coordinates
    matched_df = candidates_df[
        ~missing_coords_mask
    ].copy()

    if matched_df.empty:
        empty_itinerary = pd.DataFrame(
            columns=output_cols
        )

        excluded_df = (
            pd.concat(
                excluded_parts,
                ignore_index=True,
            )
            if excluded_parts
            else empty_excluded
        )

        if return_excluded:
            return empty_itinerary, excluded_df

        return empty_itinerary

    # Set the maximum geographic radius for a one-day itinerary
    max_one_day_radius_km = 50.0

    best_group_indices = []
    best_anchor_rank = float("inf")

    # Try each recommendation as a possible geographic anchor
    for anchor_idx, anchor_row in matched_df.iterrows():
        anchor_lat = float(anchor_row["latitude"])
        anchor_lon = float(anchor_row["longitude"])
        anchor_rank = float(anchor_row["rank"])

        current_group_indices = []

        # Find recommendations located within 50 km of this anchor
        for candidate_idx, candidate_row in matched_df.iterrows():
            distance = haversine_distance(
                anchor_lat,
                anchor_lon,
                float(candidate_row["latitude"]),
                float(candidate_row["longitude"]),
            )

            if distance <= max_one_day_radius_km:
                current_group_indices.append(candidate_idx)

        # Prefer the group containing more nearby recommendations
        if len(current_group_indices) > len(best_group_indices):
            best_group_indices = current_group_indices
            best_anchor_rank = anchor_rank

        # If the group sizes are equal, prefer the better-ranked anchor
        elif (
            len(current_group_indices) == len(best_group_indices)
            and anchor_rank < best_anchor_rank
        ):
            best_group_indices = current_group_indices
            best_anchor_rank = anchor_rank

    # Keep attractions from the strongest geographic group
    selected_df = matched_df.loc[
        best_group_indices
    ].copy()

    # Recommendations outside the selected geographic group are excluded
    excluded_geo_df = matched_df.drop(
        index=best_group_indices
    ).copy()

    if not excluded_geo_df.empty:
        excluded_parts.append(
            _build_excluded_df(
                excluded_geo_df,
                "geographically_incompatible",
            )
        )

    # Prioritise selected attractions according to recommendation rank
    selected_df = selected_df.sort_values(
        "rank"
    )

    # Keep up to M destinations
    if len(selected_df) > num_stops:
        keep_df = selected_df.head(
            num_stops
        )

        removed_df = selected_df.iloc[
            num_stops:
        ]

        selected_df = keep_df.copy()

        excluded_parts.append(
            _build_excluded_df(
                removed_df,
                "not_selected",
            )
        )

    if selected_df.empty:
        empty_itinerary = pd.DataFrame(
            columns=output_cols
        )

        excluded_df = (
            pd.concat(
                excluded_parts,
                ignore_index=True,
            )
            if excluded_parts
            else empty_excluded
        )

        if return_excluded:
            return empty_itinerary, excluded_df

        return empty_itinerary

    # All selected attractions belong to one travel day
    selected_df["day"] = 1

    # Order attractions using Greedy Nearest-Neighbor
    itinerary_df = order_day(
        selected_df
    )

    itinerary_df[
        "recommendation_score"
    ] = itinerary_df.apply(
        _recommendation_score,
        axis=1,
    )

    final_itinerary = itinerary_df[
        output_cols
    ]

    excluded_df = (
        pd.concat(
            excluded_parts,
            ignore_index=True,
        )
        if excluded_parts
        else empty_excluded
    )

    if return_excluded:
        return final_itinerary, excluded_df

    return final_itinerary