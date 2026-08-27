# Module Contribution: Goh Thong En (26WMR12713) - User-Based Collaborative Filtering (UBCF)

"""
User-Based Collaborative Filtering recommendation module.

This module implements memory-based User-Based Collaborative Filtering using Cosine Similarity,
Top-K nearest neighbors, and similarity-weighted rating predictions.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from src.preprocessing import validate_columns


def build_user_item_matrix(
    train_df: pd.DataFrame
) -> Tuple[np.ndarray, Dict[int, int], Dict[str, int]]:
    """
    Pivots train_df into a user-item matrix where missing entries are np.nan.

    Args:
        train_df (pd.DataFrame): Training interactions DataFrame.
                                Expected columns: tourist_id, attraction_uid, rating.

    Returns:
        Tuple[np.ndarray, Dict[int, int], Dict[str, int]]:
            - user_item_matrix (np.ndarray): 2D numpy array of shape (n_users, n_items).
            - user_index (Dict[int, int]): Dict mapping tourist_id to row index.
            - attraction_index (Dict[str, int]): Dict mapping attraction_uid to column index.
    """
    validate_columns(train_df, ["tourist_id", "attraction_uid", "rating"])

    # Extract unique users and items
    unique_users = train_df["tourist_id"].unique()
    unique_items = train_df["attraction_uid"].unique()

    # Build index mappings
    user_index = {u_id: idx for idx, u_id in enumerate(unique_users)}
    attraction_index = {a_id: idx for idx, a_id in enumerate(unique_items)}

    # Initialize empty matrix filled with np.nan
    n_users = len(unique_users)
    n_items = len(unique_items)
    user_item_matrix = np.full((n_users, n_items), np.nan, dtype=np.float64)

    # Populate observed ratings
    for _, row in train_df.iterrows():
        u_idx = user_index[row["tourist_id"]]
        i_idx = attraction_index[row["attraction_uid"]]
        user_item_matrix[u_idx, i_idx] = row["rating"]

    return user_item_matrix, user_index, attraction_index

def build_user_similarity_matrix(user_item_matrix: np.ndarray) -> np.ndarray:
    """
    Calculates pairwise user-user cosine similarity from user_item_matrix.

    Cosine similarity is computed using only overlapping rated items.

    Args:
        user_item_matrix (np.ndarray): User-item matrix of shape
            (n_users, n_items).

    Returns:
        np.ndarray: User-user cosine similarity matrix.
    """

    # Create binary mask of observed ratings
    mask = ~np.isnan(user_item_matrix)

    # Replace missing ratings with zero
    matrix_zeroed = np.nan_to_num(
        user_item_matrix,
        nan=0.0,
    )

    # Squared ratings are used for cosine similarity norms
    matrix_squared = matrix_zeroed ** 2

    # Calculate dot products between all user pairs
    dot_products = (
        matrix_zeroed
        @ matrix_zeroed.T
    )

    # Convert observation mask for matrix calculation
    mask_float = mask.astype(float)

    # Calculate norms using only overlapping rated attractions
    norm_sq_i = (
        matrix_squared
        @ mask_float.T
    )

    norm_sq_j = (
        mask_float
        @ matrix_squared.T
    )

    norms_i = np.sqrt(norm_sq_i)
    norms_j = np.sqrt(norm_sq_j)

    denominator = (
        norms_i * norms_j
    )

    # Create similarity matrix
    similarity_matrix = np.zeros_like(
        dot_products,
        dtype=float,
    )

    # Avoid division by zero
    valid_mask = denominator > 0.0

    similarity_matrix[valid_mask] = (
        dot_products[valid_mask]
        / denominator[valid_mask]
    )

    # Keep cosine similarity between 0 and 1
    similarity_matrix = np.clip(
        similarity_matrix,
        0.0,
        1.0,
    )

    # A user is fully similar to themselves
    np.fill_diagonal(
        similarity_matrix,
        1.0,
    )

    return similarity_matrix

def find_nearest_neighbors(
    tourist_id: int,
    user_similarity_matrix: np.ndarray,
    user_index: Dict[int, int],
    k: int = 20
) -> List[Tuple[int, float]]:
    """
    Extracts the top-K nearest neighbors for a target user.

    Excludes target user and filters out non-positive similarity scores.

    Args:
        tourist_id (int): Target user ID.
        user_similarity_matrix (np.ndarray): Pairwise user similarity matrix.
        user_index (Dict[int, int]): Dict mapping tourist_id to row index.
        k (int): Number of nearest neighbors to retrieve (default 20).

    Returns:
        List[Tuple[int, float]]: List of (neighbor_tourist_id, similarity_score) tuples.
    """
    if tourist_id not in user_index:
        return []

    target_row = user_index[tourist_id]
    similarities = user_similarity_matrix[target_row]

    # Map row indexes back to tourist_ids
    reverse_user_index = {idx: u_id for u_id, idx in user_index.items()}

    # Create candidate list of (tourist_id, similarity)
    candidates = []
    for idx, sim in enumerate(similarities):
        if idx != target_row and sim > 0.0:
            candidates.append((reverse_user_index[idx], sim))

    # Sort descending by similarity, tie-breaking by tourist_id ascending
    candidates.sort(key=lambda x: (-x[1], x[0]))

    return candidates[:k]


def predict_ratings(
    neighbors: List[Tuple[int, float]],
    user_item_matrix: np.ndarray,
    user_index: Dict[int, int],
    attraction_index: Dict[str, int]
) -> Dict[str, float]:
    """
    Predicts candidate ratings for attractions based on neighbor ratings.

    Uses similarity-weighted averages. Excludes attractions with no ratings
    from neighbors.

    Args:
        neighbors (List[Tuple[int, float]]): Nearest neighbors with similarity scores.
        user_item_matrix (np.ndarray): 2D array of shape (n_users, n_items).
        user_index (Dict[int, int]): Dict mapping tourist_id to row index.
        attraction_index (Dict[str, int]): Dict mapping attraction_uid to column index.

    Returns:
        Dict[str, float]: Dict mapping attraction_uid to similarity-weighted predicted rating.
    """
    if not neighbors:
        return {}

    # Extract rows of neighbors and their similarity scores
    neighbor_rows = [user_index[n_id] for n_id, _ in neighbors]
    similarities = np.array([sim for _, sim in neighbors])

    # Slice user_item_matrix for neighbors: shape (n_neighbors, n_items)
    neighbor_ratings = user_item_matrix[neighbor_rows, :]

    # Mask of observed ratings: shape (n_neighbors, n_items)
    observed_mask = ~np.isnan(neighbor_ratings)

    # Numerator: dot product of similarities and ratings for each item
    ratings_filled = np.nan_to_num(neighbor_ratings)
    numerator = similarities @ ratings_filled

    # Denominator: sum of absolute similarities of neighbors who rated each item
    abs_similarities = np.abs(similarities)
    denominator = abs_similarities @ observed_mask

    # Generate predictions dictionary
    predictions = {}
    for item_uid, col_idx in attraction_index.items():
        denom_val = denominator[col_idx]
        if denom_val > 0.0:
            predictions[item_uid] = float(numerator[col_idx] / denom_val)

    return predictions


def recommend_attractions_cf(
    tourist_id: int,
    train_df: pd.DataFrame,
    attraction_df: pd.DataFrame,
    user_item_matrix: np.ndarray,
    user_similarity_matrix: np.ndarray,
    user_index: Dict[int, int],
    attraction_index: Dict[str, int],
    k: int = 20,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Generates ranked attraction recommendations for a user using User-Based CF.

    Excludes already visited/rated attractions in train_df and ranks the rest
    descending by predicted rating.

    Args:
        tourist_id (int): Target user ID.
        train_df (pd.DataFrame): Training interactions DataFrame.
        attraction_df (pd.DataFrame): Unique attractions DataFrame.
        user_item_matrix (np.ndarray): Sparse user-item rating matrix.
        user_similarity_matrix (np.ndarray): Pairwise user similarity matrix.
        user_index (Dict[int, int]): Dict mapping tourist_id to row index.
        attraction_index (Dict[str, int]): Dict mapping attraction_uid to column index.
        k (int): Number of neighbors to use (default 20).
        top_n (int): Number of recommendations to return (default 10).

    Returns:
        pd.DataFrame: Ranked recommendations table with columns:
                      attraction_uid, attraction_name, attraction_category, city, predicted_rating, rank.
                      Returns an empty DataFrame with output columns if tourist_id is a cold start.
    """
    output_cols = [
        "attraction_uid",
        "attraction_name",
        "attraction_category",
        "city",
        "predicted_rating",
        "rank",
    ]

    # Cold start check: if user is not in the trained model index.
    if tourist_id not in user_index:
        return pd.DataFrame(columns=output_cols)

    # 1. Find the nearest neighbors.
    neighbors = find_nearest_neighbors(
        tourist_id=tourist_id,
        user_similarity_matrix=user_similarity_matrix,
        user_index=user_index,
        k=k
    )

    if not neighbors:
        return pd.DataFrame(columns=output_cols)

    # 2. Predict ratings for unvisited attractions.
    predictions = predict_ratings(
        neighbors=neighbors,
        user_item_matrix=user_item_matrix,
        user_index=user_index,
        attraction_index=attraction_index
    )

    if not predictions:
        return pd.DataFrame(columns=output_cols)

    # 3. Construct candidate recommendations.
    results = pd.DataFrame(
        list(predictions.items()),
        columns=["attraction_uid", "predicted_rating"]
    )

    # 4. Filter out visited/rated attractions in training data.
    visited_uids = set(
        train_df[train_df["tourist_id"] == tourist_id]["attraction_uid"]
    )
    results_filtered = results[~results["attraction_uid"].isin(visited_uids)]

    # 5. Sort remaining attractions descending by predicted rating.
    results_sorted = results_filtered.sort_values(
        by="predicted_rating", ascending=False
    )
    top_results = results_sorted.head(top_n).copy()

    if top_results.empty:
        return pd.DataFrame(columns=output_cols)

    # 6. Join back descriptive metadata fields for display.
    top_recommendations = pd.merge(
        top_results,
        attraction_df[
            ["attraction_uid", "attraction_name", "attraction_category", "city"]
        ],
        on="attraction_uid",
        how="left",
    )
    
    # 7. Add 1-indexed rank column.
    top_recommendations["rank"] = range(1, len(top_recommendations) + 1)

    return top_recommendations[output_cols]

__all__ = [
    "build_user_item_matrix",
    "build_user_similarity_matrix",
    "find_nearest_neighbors",
    "predict_ratings",
    "recommend_attractions_cf",
]




