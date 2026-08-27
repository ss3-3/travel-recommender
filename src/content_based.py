# Module Contribution: Soh Sook Shan (26WMR12721) - Content-Based Filtering (CBF)

"""
Content-Based Filtering recommendation module.

This module recommends tourist attractions to users by matching the textual and
categorical profiles of attractions they rated highly (rating >= 4.0) against
all other attractions using TF-IDF representation and cosine similarity.
"""

from typing import Dict, Tuple, cast
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_content_column(attraction_df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs a descriptive 'content' text column for each attraction.

    Concatenates category, level, province, city, and main_spots_clean into
    a single lowercase, space-separated string. The category and level fields
    are repeated twice to increase their relative weight in TF-IDF representations.

    Args:
        attraction_df (pd.DataFrame): Cleaned unique attractions DataFrame.

    Returns:
        pd.DataFrame: Attraction DataFrame with an added 'content' column.
    """
    df = attraction_df.copy()

    # Repeat category and level twice to increase their relative weight
    cat = df["attraction_category"].fillna("").astype(str)
    lvl = df["attraction_level"].fillna("").astype(str)
    prov = df["province"].fillna("").astype(str)
    city = df["city"].fillna("").astype(str)
    spots = df["main_spots_clean"].fillna("").astype(str)

    content = cat.str.cat(cast(list[str], [cat, lvl, lvl, prov, city, spots]), sep=" ")

    df["content"] = content.str.strip().str.lower()
    return df


def build_tfidf_matrix(
    content_df: pd.DataFrame
) -> Tuple[TfidfVectorizer, csr_matrix, Dict[str, int]]:
    """
    Fits a TF-IDF Vectorizer on the content column of attractions.

    Args:
        content_df (pd.DataFrame): Attractions DataFrame containing a 'content' column.

    Returns:
        Tuple[TfidfVectorizer, csr_matrix, Dict[str, int]]:
            - vectorizer: Fitted TfidfVectorizer instance.
            - tfidf_matrix: Sparse matrix of shape (n_attractions, n_features).
            - attraction_index: Dict mapping attraction_uid to matrix row index.
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(content_df["content"])

    # Construct mapping from attraction_uid to row index in tfidf_matrix
    attraction_index = {
        uid: idx for idx, uid in enumerate(content_df["attraction_uid"])
    }

    return vectorizer, tfidf_matrix, attraction_index


def build_user_profile(
    tourist_id: int,
    interactions_df: pd.DataFrame,
    tfidf_matrix: csr_matrix,
    attraction_index: Dict[str, int],
    rating_threshold: float = 4.0,
) -> csr_matrix | None:
    """
    Builds a user profile vector representing their content preference.

    The user profile is constructed by taking a rating-weighted average of the
    TF-IDF feature vectors of attractions that the user rated highly (rating >= threshold).

    Args:
        tourist_id (int): Target user ID.
        interactions_df (pd.DataFrame): Cleaned user ratings interactions DataFrame.
        tfidf_matrix (csr_matrix): Sparse TF-IDF attraction feature matrix.
        attraction_index (Dict[str, int]): Dict mapping attraction_uid to matrix row index.
        rating_threshold (float): Threshold above which ratings represent positive preference.

    Returns:
        csr_matrix | None: Rating-weighted average TF-IDF vector, or None if the
                            user has no ratings >= threshold (user cold start).
    """
    # 1. Retrieve user ratings that satisfy the threshold
    user_ratings = interactions_df[
        (interactions_df["tourist_id"] == tourist_id)
        & (interactions_df["rating"] >= rating_threshold)
    ]

    if user_ratings.empty:
        return None

    # 2. Accumulate rating-weighted vectors
    weighted_vector_sum = None
    total_weight = 0.0

    for _, row in user_ratings.iterrows():
        uid = row["attraction_uid"]
        rating = row["rating"]

        if uid in attraction_index:
            row_idx = attraction_index[uid]
            attraction_vector = tfidf_matrix[row_idx]

            weighted_vector = attraction_vector * rating
            if weighted_vector_sum is None:
                weighted_vector_sum = weighted_vector.copy()
            else:
                weighted_vector_sum += weighted_vector

            total_weight += rating

    if weighted_vector_sum is None or total_weight == 0.0:
        return None

    # 3. Compute rating-weighted average profile vector
    profile_vector = weighted_vector_sum / total_weight
    return cast(csr_matrix, profile_vector)


def compute_similarity(
    profile_vector: csr_matrix, tfidf_matrix: csr_matrix
) -> np.ndarray:
    """
    Computes pairwise cosine similarity between user profile vector and all attraction vectors.

    Args:
        profile_vector (csr_matrix): Sparse 1D/2D user profile TF-IDF vector of shape (1, n_features).
        tfidf_matrix (csr_matrix): Sparse TF-IDF attraction feature matrix of shape (n_attractions, n_features).

    Returns:
        np.ndarray: 1D numpy array of similarity scores for all attractions.
    """
    scores = cosine_similarity(profile_vector, tfidf_matrix)
    # Flatten the resulting (1, n_attractions) array into (n_attractions,)
    return cast(np.ndarray, scores.flatten())


def recommend_attractions(
    tourist_id: int,
    interactions_df: pd.DataFrame,
    attraction_df: pd.DataFrame,
    tfidf_matrix: csr_matrix,
    attraction_index: Dict[str, int],
    top_n: int = 10,
    rating_threshold: float = 4.0,
) -> pd.DataFrame:
    """
    Generates ranked attraction recommendations for a user.

    Excludes already visited/rated attractions and ranks the rest descending
    by similarity score. This is the main public entry point of the module.

    Args:
        tourist_id (int): Target user ID.
        interactions_df (pd.DataFrame): User ratings interactions DataFrame.
        attraction_df (pd.DataFrame): Unique attractions DataFrame.
        tfidf_matrix (csr_matrix): Sparse TF-IDF attraction feature matrix.
        attraction_index (Dict[str, int]): Dict mapping attraction_uid to matrix row index.
        top_n (int): Number of recommendations to return.
        rating_threshold (float): Minimum rating threshold for positive user profile (default 4.0).

    Returns:
        pd.DataFrame: Ranked recommendations table with columns:
                      attraction_uid, attraction_name, attraction_category, city, similarity_score, rank.
                      Returns an empty DataFrame with output columns if no profile can be built.
    """
    # Define output schema
    output_cols = [
        "attraction_uid",
        "attraction_name",
        "attraction_category",
        "city",
        "similarity_score",
        "rank",
    ]

    # 1. Build user profile vector
    profile_vector = build_user_profile(
        tourist_id=tourist_id,
        interactions_df=interactions_df,
        tfidf_matrix=tfidf_matrix,
        attraction_index=attraction_index,
        rating_threshold=rating_threshold,
    )

    # Cold start: if user profile cannot be built, return empty DataFrame with schema
    if profile_vector is None:
        return pd.DataFrame(columns=output_cols)

    # 2. Compute similarity scores against all attractions
    similarity_scores = compute_similarity(profile_vector, tfidf_matrix)

    # 3. Create DataFrame of all attraction similarity scores
    results = pd.DataFrame(
        {
            "attraction_uid": attraction_df["attraction_uid"],
            "similarity_score": similarity_scores,
        }
    )

    # 4. Filter out visited/rated attractions to prevent recommending already visited spots.
    # Using a set provides efficient O(1) average-case membership testing.
    visited_uids = set(
        interactions_df[interactions_df["tourist_id"] == tourist_id]["attraction_uid"]
    )
    results_filtered = results[~results["attraction_uid"].isin(visited_uids)]

    # 5. Sort remaining attractions descending by score
    results_sorted = results_filtered.sort_values(
        by="similarity_score", ascending=False
    )
    top_results = results_sorted.head(top_n).copy()

    # 6. Join back descriptive metadata fields for display purposes
    top_recommendations = pd.merge(
        top_results,
        attraction_df[
            ["attraction_uid", "attraction_name", "attraction_category", "city"]
        ],
        on="attraction_uid",
        how="left",
    )

    # 7. Add 1-indexed rank column
    top_recommendations["rank"] = range(1, len(top_recommendations) + 1)

    return top_recommendations[output_cols]


__all__ = [
    "build_content_column",
    "build_tfidf_matrix",
    "build_user_profile",
    "compute_similarity",
    "recommend_attractions",
]
