"""
Verification tests for Phase 4 User-Based Collaborative Filtering.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
SCRATCH_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRATCH_DIR.parent
sys.path.append(str(PROJECT_DIR))

from src.preprocessing import load_dataset, prepare_attractions, prepare_interactions, train_test_split_by_user
from src.collaborative import (
    build_user_item_matrix,
    build_user_similarity_matrix,
    find_nearest_neighbors,
    predict_ratings,
    recommend_attractions_cf,
)

DATA_PATH = PROJECT_DIR / "data" / "tourism_recommendation_dataset_en.csv"


def test_build_user_item_matrix():
    print("Testing build_user_item_matrix...")
    df = load_dataset(str(DATA_PATH))
    interactions_df = prepare_interactions(df)
    train_df, _ = train_test_split_by_user(interactions_df)

    matrix, user_index, attraction_index = build_user_item_matrix(train_df)

    assert isinstance(matrix, np.ndarray), "Matrix must be a numpy ndarray"
    assert matrix.ndim == 2, "Matrix must be 2D"

    n_users = len(user_index)
    n_items = len(attraction_index)
    assert matrix.shape == (n_users, n_items), f"Expected shape ({n_users}, {n_items}), got {matrix.shape}"

    # Verify that missing values are np.nan and not imputed/zero-filled
    # Total ratings in train_df must equal count of non-NaN values in matrix
    non_nan_count = np.isnan(matrix).sum()
    observed_count = (~np.isnan(matrix)).sum()

    print(f"✔ Matrix shape: {matrix.shape}")
    print(f"✔ Observed ratings: {observed_count}, Unobserved (NaN) entries: {non_nan_count}")
    print("✔ build_user_item_matrix passed all checks!")
    return matrix


def test_build_user_similarity_matrix(matrix: np.ndarray):
    print("\nTesting build_user_similarity_matrix...")
    similarity_matrix = build_user_similarity_matrix(matrix)

    n_users = matrix.shape[0]
    assert similarity_matrix.shape == (n_users, n_users), f"Expected shape ({n_users}, {n_users}), got {similarity_matrix.shape}"

    # Verify symmetry
    assert np.allclose(similarity_matrix, similarity_matrix.T), "Similarity matrix must be symmetric"

    # Verify diagonal is 1.0
    diag = np.diagonal(similarity_matrix)
    assert np.allclose(diag, 1.0), "Diagonal elements must be 1.0"

    # Verify similarity bounds [0, 1] for cosine similarity of positive ratings (ratings range 1-5)
    assert similarity_matrix.min() >= 0.0 - 1e-9, f"Similarity values cannot be negative, got minimum {similarity_matrix.min()}"
    assert similarity_matrix.max() <= 1.0 + 1e-9, f"Similarity values cannot exceed 1.0, got maximum {similarity_matrix.max()}"

    print(f"✔ Similarity matrix shape: {similarity_matrix.shape}")
    print(f"✔ Diagonal verification: {diag[:5]}...")
    print("✔ build_user_similarity_matrix passed all checks!")
    return similarity_matrix


def test_find_nearest_neighbors(similarity_matrix: np.ndarray, user_index: dict):
    print("\nTesting find_nearest_neighbors...")
    sample_user = list(user_index.keys())[0]
    k = 20
    neighbors = find_nearest_neighbors(sample_user, similarity_matrix, user_index, k=k)

    assert isinstance(neighbors, list), "Neighbors must be a list"
    assert len(neighbors) <= k, f"Neighbor count cannot exceed {k}"

    # Verify that the target user is not in the neighbor list
    for neighbor_id, sim in neighbors:
        assert neighbor_id != sample_user, "Target user cannot be their own neighbor"
        assert sim > 0.0, f"Similarity must be positive, got {sim}"

    # Verify that the list is sorted in descending order of similarity
    similarities = [sim for _, sim in neighbors]
    is_sorted = all(similarities[i] >= similarities[i + 1] for i in range(len(similarities) - 1))
    assert is_sorted, f"Neighbors are not sorted by similarity descending: {similarities}"

    print(f"✔ Number of neighbors retrieved: {len(neighbors)}")
    print(f"✔ Nearest neighbor: User {neighbors[0][0]} with similarity {neighbors[0][1]:.4f}")
    print("✔ find_nearest_neighbors passed all checks!")
    return neighbors


def test_predict_ratings(similarity_matrix: np.ndarray, user_item_matrix: np.ndarray, user_index: dict, attraction_index: dict):
    print("\nTesting predict_ratings...")
    sample_user = list(user_index.keys())[0]
    neighbors = find_nearest_neighbors(sample_user, similarity_matrix, user_index, k=20)

    predictions = predict_ratings(neighbors, user_item_matrix, user_index, attraction_index)

    assert isinstance(predictions, dict), "Predictions must be returned as a dict"

    # Verify predictions range sanity (ratings must be between 1.0 and 5.0)
    for attraction_uid, pred in predictions.items():
        assert 1.0 - 1e-9 <= pred <= 5.0 + 1e-9, f"Prediction out of bounds [1, 5]: {pred} for {attraction_uid}"

    print(f"✔ Number of predicted attractions: {len(predictions)}")
    print(f"✔ Sample prediction: {list(predictions.keys())[0]} -> {list(predictions.values())[0]:.4f}")
    print("✔ predict_ratings passed all checks!")
    return predictions


def test_recommend_attractions_cf(
    train_df: pd.DataFrame,
    attraction_df: pd.DataFrame,
    matrix: np.ndarray,
    similarity_matrix: np.ndarray,
    user_index: dict,
    attraction_index: dict
):
    print("\nTesting recommend_attractions_cf...")
    sample_user = list(user_index.keys())[0]
    top_n = 5

    recommendations = recommend_attractions_cf(
        tourist_id=sample_user,
        train_df=train_df,
        attraction_df=attraction_df,
        user_item_matrix=matrix,
        user_similarity_matrix=similarity_matrix,
        user_index=user_index,
        attraction_index=attraction_index,
        k=20,
        top_n=top_n
    )

    assert isinstance(recommendations, pd.DataFrame), "Output must be a pandas DataFrame"
    assert len(recommendations) <= top_n, f"Should return at most {top_n} recommendations, got {len(recommendations)}"

    # Schema Verification
    expected_cols = [
        "attraction_uid",
        "attraction_name",
        "attraction_category",
        "city",
        "predicted_rating",
        "rank",
    ]
    assert list(recommendations.columns) == expected_cols, f"Schema mismatch. Expected {expected_cols}, got {list(recommendations.columns)}"

    # Verify visited filtering
    visited_uids = set(train_df[train_df["tourist_id"] == sample_user]["attraction_uid"])
    rec_uids = set(recommendations["attraction_uid"])
    overlap = visited_uids.intersection(rec_uids)
    assert len(overlap) == 0, f"Recommendation contains visited items: {overlap}"

    # Verify ranking
    assert list(recommendations["rank"]) == list(range(1, len(recommendations) + 1)), "Rank column must be 1-indexed sequential"
    ratings = recommendations["predicted_rating"].tolist()
    is_sorted = all(ratings[i] >= ratings[i + 1] for i in range(len(ratings) - 1))
    assert is_sorted, f"Recommendations are not sorted by predicted rating descending: {ratings}"

    print(f"✔ Recommendations shape: {recommendations.shape}")
    print(f"✔ Top recommendation: {recommendations.iloc[0]['attraction_name']} with rating {recommendations.iloc[0]['predicted_rating']:.4f}")
    print("✔ recommend_attractions_cf passed all pipeline checks!")

    # Verify Cold Start behavior
    print("\nTesting cold-start behaviour...")
    cold_user = 999999
    cold_recs = recommend_attractions_cf(
        tourist_id=cold_user,
        train_df=train_df,
        attraction_df=attraction_df,
        user_item_matrix=matrix,
        user_similarity_matrix=similarity_matrix,
        user_index=user_index,
        attraction_index=attraction_index,
        k=20,
        top_n=top_n
    )
    assert isinstance(cold_recs, pd.DataFrame), "Cold start output must be a DataFrame"
    assert cold_recs.empty, "Cold start output must be empty"
    assert list(cold_recs.columns) == expected_cols, f"Schema mismatch for empty recs: {list(cold_recs.columns)}"
    print("✔ recommend_attractions_cf correctly returns empty DataFrame for cold-start user")


if __name__ == "__main__":
    df = load_dataset(str(DATA_PATH))
    attraction_df = prepare_attractions(df)
    interactions_df = prepare_interactions(df)
    train_df, test_df = train_test_split_by_user(interactions_df)

    matrix, user_index, attraction_index = build_user_item_matrix(train_df)
    sim_matrix = test_build_user_similarity_matrix(matrix)
    neighbors = test_find_nearest_neighbors(sim_matrix, user_index)
    predictions = test_predict_ratings(sim_matrix, matrix, user_index, attraction_index)

    test_recommend_attractions_cf(
        train_df=train_df,
        attraction_df=attraction_df,
        matrix=matrix,
        similarity_matrix=sim_matrix,
        user_index=user_index,
        attraction_index=attraction_index
    )

    print("\nAll User-Based Collaborative Filtering verification assertions passed! 🎉")




