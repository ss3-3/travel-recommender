# pyrefly: ignore [missing-import]
"""
Itinerary Evaluation Module for Travel Destination Recommendation System.

This module evaluates the structural and geographic characteristics/quality of the generated itinerary using
five structural and geographic metrics. It contains no machine learning algorithms.
"""

from typing import Dict, Any, Union
import numpy as np
import pandas as pd

from src.itinerary import haversine_distance


def avg_consecutive_distance(itinerary_df: pd.DataFrame) -> Dict[str, float]:
    """
    Measures the average Haversine distance between consecutive attractions.

    This describes the geographic distance between consecutive attractions.
    A lower average consecutive distance indicates a more spatially sequential path.
    Note: This characterizes routing properties, but does not guarantee
    global optimality (e.g., TSP solution).

    Args:
        itinerary_df (pd.DataFrame): Generated itinerary DataFrame containing 'day',
            'stop_order', and 'distance_from_prev_km' columns.

    Returns:
        Dict[str, float]: Dictionary mapping day keys (e.g., 'day_1') to average consecutive-stop
            distance in km, plus an 'overall' average distance for the entire itinerary.
            Returns 0.0 for days or itineraries with <= 1 stop.
    """
    if itinerary_df.empty:
        return {"overall": 0.0}

    # Ensure required columns exist
    for col in ["day", "stop_order", "distance_from_prev_km"]:
        if col not in itinerary_df.columns:
            return {"overall": 0.0}

    results = {}
    
    # Filter to consecutive stops only (transitions from a previous stop)
    consecutive_stops = itinerary_df[itinerary_df["stop_order"] > 1]

    # Group by day to compute per-day averages
    grouped = consecutive_stops.groupby("day")
    for day, group in grouped:
        if not group.empty:
            results[f"day_{day}"] = group["distance_from_prev_km"].mean()
        else:
            results[f"day_{day}"] = 0.0

    # Ensure all days represented in itinerary are present in results (even if 0 or 1 stops)
    all_days = itinerary_df["day"].unique()
    for day in all_days:
        key = f"day_{day}"
        if key not in results:
            results[key] = 0.0

    # Calculate overall average across all transitions
    if not consecutive_stops.empty:
        results["overall"] = consecutive_stops["distance_from_prev_km"].mean()
    else:
        results["overall"] = 0.0

    return results


def total_distance_per_day(itinerary_df: pd.DataFrame) -> Dict[str, float]:
    """
    Sums the travel distance (distance_from_prev_km) for each day.

    This describes the total travel distance associated with attractions assigned to each day, which is useful
    for assessing if a specific day is overloaded with long-distance travel.

    Args:
        itinerary_df (pd.DataFrame): Generated itinerary DataFrame containing 'day'
            and 'distance_from_prev_km' columns.

    Returns:
        Dict[str, float]: Dictionary mapping day keys (e.g., 'day_1') to total travel distance
            in km, plus an 'overall' total distance. Returns 0.0 for empty inputs.
    """
    if itinerary_df.empty or "distance_from_prev_km" not in itinerary_df.columns or "day" not in itinerary_df.columns:
        return {"overall": 0.0}

    results = {}
    
    # Group by day and sum distances
    grouped = itinerary_df.groupby("day")
    for day, group in grouped:
        results[f"day_{day}"] = group["distance_from_prev_km"].sum()

    # Overall total distance
    results["overall"] = itinerary_df["distance_from_prev_km"].sum()
    return results


def compactness_per_day(itinerary_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculates geographic compactness per day.

    This describes the spatial compactness of attractions within each day.
    For each day, the geographic centroid (arithmetic mean latitude and longitude)
    of that day's attractions is computed. Then, the average Haversine distance
    from each attraction to that centroid is calculated. Lower average distances
    indicate tighter spatial clusters of attractions, meaning less overall
    dispersion within that day's activities.

    Args:
        itinerary_df (pd.DataFrame): Generated itinerary DataFrame containing 'day',
            'latitude', and 'longitude' columns.

    Returns:
        Dict[str, float]: Dictionary mapping day keys (e.g., 'day_1') to the average
            distance to centroid in km, plus an 'overall' average distance to centroid.
            Returns 0.0 for empty inputs or single-attraction days.
    """
    if itinerary_df.empty:
        return {"overall": 0.0}

    # Ensure required columns exist
    for col in ["day", "latitude", "longitude"]:
        if col not in itinerary_df.columns:
            return {"overall": 0.0}

    results = {}
    all_distances = []

    grouped = itinerary_df.groupby("day")
    for day, group in grouped:
        if group.empty:
            results[f"day_{day}"] = 0.0
            continue

        # Compute centroid as arithmetic mean of lat and lon
        lat_centroid = group["latitude"].mean()
        lon_centroid = group["longitude"].mean()

        day_distances = []
        for _, row in group.iterrows():
            lat = row["latitude"]
            lon = row["longitude"]
            
            # Compute distance of attraction to its centroid
            dist = haversine_distance(lat, lon, lat_centroid, lon_centroid)
            day_distances.append(dist)
            all_distances.append(dist)

        # Average distance to centroid for the day
        if day_distances:
            results[f"day_{day}"] = np.mean(day_distances)
        else:
            results[f"day_{day}"] = 0.0

    # Overall average distance to centroid across all attractions
    if all_distances:
        results["overall"] = np.mean(all_distances)
    else:
        results["overall"] = 0.0

    return results


def candidate_carryover_rate(itinerary_df: pd.DataFrame, candidates_df: pd.DataFrame) -> float:
    """
    Measures the proportion of coordinate-matched recommendation candidates that are carried over into the final itinerary.

    Values below 1.0 are expected when geographically incompatible or lower-priority
    candidates are excluded from the final day plan.

    Args:
        itinerary_df (pd.DataFrame): Generated itinerary DataFrame containing 'attraction_uid'.
        candidates_df (pd.DataFrame): Recommendation candidates DataFrame.

    Returns:
        float: Candidate carryover rate value bounded between [0.0, 1.0]. Returns 0.0 if candidates_df is empty.
    """
    if candidates_df.empty:
        return 0.0

    # Filter candidates to only those with valid coordinates (i.e. coordinate-matched candidates)
    if "latitude" in candidates_df.columns and "longitude" in candidates_df.columns:
        matched_candidates = candidates_df[
            candidates_df["latitude"].notna() & candidates_df["longitude"].notna()
        ]
    else:
        matched_candidates = candidates_df

    num_candidates = len(matched_candidates)
    if num_candidates == 0:
        return 0.0

    # Count how many coordinate-matched candidates are present in the final itinerary
    if "attraction_uid" not in itinerary_df.columns or "attraction_uid" not in matched_candidates.columns:
        return 0.0

    itinerary_uids = set(itinerary_df["attraction_uid"])
    candidate_uids = set(matched_candidates["attraction_uid"])
    hits = itinerary_uids.intersection(candidate_uids)
    num_itinerary = len(hits)

    rate = num_itinerary / num_candidates
    return min(1.0, max(0.0, rate))


def day_balance(itinerary_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes the distribution balance of attractions assigned across days.

    This describes the distribution of attractions across days to
    prevent user fatigue. A lower standard deviation of stop counts indicates a more balanced
    itinerary.

    Args:
        itinerary_df (pd.DataFrame): Generated itinerary DataFrame containing 'day'.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'per_day_counts' (Dict[str, int]): Map of day keys to stop count.
            - 'std_dev' (float): Standard deviation of stop counts across days.
            - 'min_stops' (int): Minimum number of stops on any day.
            - 'max_stops' (int): Maximum number of stops on any day.
    """
    if itinerary_df.empty or "day" not in itinerary_df.columns:
        return {
            "per_day_counts": {},
            "std_dev": 0.0,
            "min_stops": 0,
            "max_stops": 0
        }

    counts = itinerary_df.groupby("day").size()
    per_day_counts = {f"day_{day}": count for day, count in counts.items()}
    counts_list = list(counts.values)

    if len(counts_list) <= 1:
        std_dev = 0.0
    else:
        # Standard deviation of the population
        std_dev = counts.std(ddof=0)
        if np.isnan(std_dev):
            std_dev = 0.0

    min_stops = int(counts.min())
    max_stops = int(counts.max())

    return {
        "per_day_counts": per_day_counts,
        "std_dev": std_dev,
        "min_stops": min_stops,
        "max_stops": max_stops
    }


def evaluate_itinerary(itinerary_df: pd.DataFrame, candidates_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Orchestrates the evaluation of the structural and geographic quality of a generated itinerary.

    Calculates:
      1. Average consecutive-stop distance
      2. Total travel distance per day
      3. Geographic compactness per day
      4. Candidate carryover rate
      5. Day balance

    Args:
        itinerary_df (pd.DataFrame): Generated itinerary DataFrame.
        candidates_df (pd.DataFrame): Recommendation candidates DataFrame.

    Returns:
        Dict[str, Any]: Consolidated dictionary of structural quality metrics.
    """
    return {
        "avg_consecutive_distance": avg_consecutive_distance(itinerary_df),
        "total_distance_per_day": total_distance_per_day(itinerary_df),
        "compactness_per_day": compactness_per_day(itinerary_df),
        "candidate_carryover_rate": candidate_carryover_rate(itinerary_df, candidates_df),
        "day_balance": day_balance(itinerary_df)
    }
