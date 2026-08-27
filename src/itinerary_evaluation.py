# Module Contribution: Soh Sook Shan (26WMR12721) & Goh Thong En (26WMR12713) - Itinerary Evaluation Metrics (joint work)

# pyrefly: ignore [missing-import]
"""
Itinerary Evaluation Module for Travel Destination Recommendation System.

This module evaluates the structural and geographic characteristics
of a generated one-day itinerary using four evaluation metrics.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.itinerary import haversine_distance


def avg_consecutive_distance(itinerary_df: pd.DataFrame) -> float | None:
    """
    Calculates the average Haversine distance between consecutive stops.

    A lower value indicates that consecutive attractions are generally
    located closer to each other.

    Args:
        itinerary_df: Generated one-day itinerary DataFrame.

    Returns:
        Average consecutive-stop distance in kilometres.
    """
    if itinerary_df.empty:
        return None

    required_cols = [
        "stop_order",
        "distance_from_prev_km",
    ]

    for col in required_cols:
        if col not in itinerary_df.columns:
            return 0.0

    # Stop 1 has no previous destination, so it is excluded
    consecutive_stops = itinerary_df[
        itinerary_df["stop_order"] > 1
    ]

    if consecutive_stops.empty:
        return None

    return float(
        consecutive_stops["distance_from_prev_km"].mean()
    )


def total_travel_distance(itinerary_df: pd.DataFrame) -> float:
    """
    Calculates the total travel distance of the one-day itinerary.

    Args:
        itinerary_df: Generated one-day itinerary DataFrame.

    Returns:
        Total travel distance in kilometres.
    """
    if itinerary_df.empty:
        return 0.0

    if "distance_from_prev_km" not in itinerary_df.columns:
        return 0.0

    return float(
        itinerary_df["distance_from_prev_km"].fillna(0).sum()
    )


def geographic_compactness(itinerary_df: pd.DataFrame) -> float | None:
    """
    Calculates the geographic compactness of the one-day itinerary.

    The geographic centroid of all selected attractions is calculated.
    The average Haversine distance from each attraction to the centroid
    is then used as the compactness value.

    A lower value indicates that the selected attractions are more
    geographically concentrated.

    Args:
        itinerary_df: Generated one-day itinerary DataFrame.

    Returns:
        Average distance to the geographic centroid in kilometres.
    """
    if itinerary_df.empty or len(itinerary_df) < 2:
        return None

    required_cols = [
        "latitude",
        "longitude",
    ]

    for col in required_cols:
        if col not in itinerary_df.columns:
            return 0.0

    # Calculate geographic centroid
    lat_centroid = itinerary_df["latitude"].mean()
    lon_centroid = itinerary_df["longitude"].mean()

    distances = []

    for _, row in itinerary_df.iterrows():
        distance = haversine_distance(
            float(row["latitude"]),
            float(row["longitude"]),
            float(lat_centroid),
            float(lon_centroid),
        )

        distances.append(distance)

    if not distances:
        return 0.0

    return float(np.mean(distances))


def candidate_carryover_rate(
    itinerary_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
) -> float:
    """
    Calculates the proportion of Top-N recommendation candidates
    that are included in the final one-day itinerary.

    A higher value indicates that more recommended attractions were
    retained after geographic filtering and itinerary selection.

    Args:
        itinerary_df: Generated one-day itinerary DataFrame.
        candidates_df: Top-N recommendation DataFrame.

    Returns:
        Candidate carryover rate between 0.0 and 1.0.
    """
    if candidates_df.empty:
        return 0.0

    if "attraction_uid" not in candidates_df.columns:
        return 0.0

    if "attraction_uid" not in itinerary_df.columns:
        return 0.0

    # Use unique attraction IDs to avoid duplicate counting
    candidate_uids = set(
        candidates_df["attraction_uid"].dropna()
    )

    itinerary_uids = set(
        itinerary_df["attraction_uid"].dropna()
    )

    if len(candidate_uids) == 0:
        return 0.0

    carried_over = candidate_uids.intersection(
        itinerary_uids
    )

    carryover_rate = (
        len(carried_over) / len(candidate_uids)
    )

    return float(carryover_rate)


def evaluate_itinerary(
    itinerary_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Evaluates the generated one-day itinerary.

    Calculates:
        1. Average consecutive-stop distance
        2. Total travel distance
        3. Geographic compactness
        4. Candidate carryover rate

    Args:
        itinerary_df: Generated one-day itinerary DataFrame.
        candidates_df: Top-N recommendation DataFrame.

    Returns:
        Dictionary containing the four itinerary evaluation metrics.
    """
    return {
        "avg_consecutive_distance": avg_consecutive_distance(
            itinerary_df
        ),
        "total_travel_distance": total_travel_distance(
            itinerary_df
        ),
        "geographic_compactness": geographic_compactness(
            itinerary_df
        ),
        "candidate_carryover_rate": candidate_carryover_rate(
            itinerary_df,
            candidates_df,
        ),
    }