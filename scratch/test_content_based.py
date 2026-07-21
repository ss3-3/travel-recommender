import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

# Add project root directory to sys.path to import src
SCRATCH_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRATCH_DIR.parent
sys.path.append(str(PROJECT_DIR))

from src.preprocessing import load_dataset, prepare_attractions, prepare_interactions
from src.content_based import (
    build_content_column,
    build_tfidf_matrix,
    build_user_profile,
    compute_similarity,
    recommend_attractions,
)

DATA_PATH = PROJECT_DIR / "data" / "tourism_recommendation_dataset_en.csv"


def test_content_based_recommendation():
    print("Starting verification of src/content_based.py...")

    # 1. Load and Preprocess Data
    df = load_dataset(str(DATA_PATH))
    attraction_df = prepare_attractions(df)
    interactions_df = prepare_interactions(df)
    num_attractions = len(attraction_df)

    # 2. Verify build_content_column
    print("\n--- Testing build_content_column ---")
    attraction_with_content = build_content_column(attraction_df)
    assert "content" in attraction_with_content.columns, "Should add 'content' column"

    # Verify field repetition and lowercase conversion
    # Let's check a specific row
    sample_row = attraction_with_content.iloc[0]
    category = sample_row["attraction_category"].strip().lower()
    level = sample_row["attraction_level"].strip().lower()
    province = sample_row["province"].strip().lower()
    city = sample_row["city"].strip().lower()
    main_spots = sample_row["main_spots_clean"].strip().lower()

    expected_content = (
        f"{category} {category} {level} {level} {province} {city} {main_spots}".strip()
    )
    # Normalize spaces in case some tokens were empty
    expected_content = " ".join(expected_content.split())
    actual_content = " ".join(sample_row["content"].split())

    assert (
        actual_content == expected_content
    ), f"Content mismatch. Expected:\n'{expected_content}'\nGot:\n'{actual_content}'"
    print(
        "✔ build_content_column successfully concatenated and weighted categories/levels"
    )

    # 3. Verify build_tfidf_matrix
    print("\n--- Testing build_tfidf_matrix ---")
    vectorizer, tfidf_matrix, attraction_index = build_tfidf_matrix(
        attraction_with_content
    )

    assert (
        tfidf_matrix.shape[0] == num_attractions
    ), f"Expected {num_attractions} attraction row vectors, got {tfidf_matrix.shape[0]}"
    assert tfidf_matrix.shape[1] > 0, "Vocabulary size must be greater than 0"
    assert (
        len(attraction_index) == num_attractions
    ), f"Mapping must contain exactly {num_attractions} entries"

    # Verify index mapping lookup matches row count
    for uid, idx in attraction_index.items():
        assert 0 <= idx < num_attractions, f"Mapped row index {idx} out of bounds"
        # Row matching
        assert (
            attraction_df.iloc[idx]["attraction_uid"] == uid
        ), f"Index mismatch for {uid}"
    print(
        "✔ build_tfidf_matrix generated correct TF-IDF shape and unique row index mapping"
    )

    # 4. Verify build_user_profile
    print("\n--- Testing build_user_profile ---")

    # Test valid user with ratings >= 4.0
    # Let's inspect tourist_id 1
    user_ratings = interactions_df[interactions_df["tourist_id"] == 1]
    high_ratings = user_ratings[user_ratings["rating"] >= 4.0]

    profile_vector = build_user_profile(
        tourist_id=1,
        interactions_df=interactions_df,
        tfidf_matrix=tfidf_matrix,
        attraction_index=attraction_index,
        rating_threshold=4.0,
    )

    if not high_ratings.empty:
        assert (
            profile_vector is not None
        ), "Profile vector should be created for user with positive ratings"
        assert isinstance(
            profile_vector, csr_matrix
        ), "Profile vector must be a sparse CSR matrix"
        assert profile_vector.shape == (
            1,
            tfidf_matrix.shape[1],
        ), f"Expected shape (1, {tfidf_matrix.shape[1]}), got {profile_vector.shape}"
        print("✔ build_user_profile generated valid rating-weighted preference vector")
    else:
        print("ℹ Tourist 1 has no rating >= 4.0 in raw logs (unexpected but handled)")

    # Test user cold start (no ratings >= threshold)
    # We construct a fabricated user rating series with ratings all below 4.0
    fake_interactions = pd.DataFrame(
        [
            {
                "tourist_id": 99999,
                "attraction_uid": attraction_df.iloc[0]["attraction_uid"],
                "rating": 3.0,
            },
            {
                "tourist_id": 99999,
                "attraction_uid": attraction_df.iloc[1]["attraction_uid"],
                "rating": 2.5,
            },
        ]
    )
    cold_profile = build_user_profile(
        tourist_id=99999,
        interactions_df=fake_interactions,
        tfidf_matrix=tfidf_matrix,
        attraction_index=attraction_index,
        rating_threshold=4.0,
    )
    assert (
        cold_profile is None
    ), "Should return None for user with no ratings >= threshold"
    print(
        "✔ build_user_profile correctly returns None for cold-start (no positive ratings) users"
    )

    # 5. Verify compute_similarity
    print("\n--- Testing compute_similarity ---")
    # Take tourist 1's profile vector (if exists)
    if profile_vector is not None:
        scores = compute_similarity(profile_vector, tfidf_matrix)
        assert isinstance(
            scores, np.ndarray
        ), "Similarity scores must be returned as a numpy array"
        assert scores.shape == (
            num_attractions,
        ), f"Expected shape ({num_attractions},), got {scores.shape}"
        assert (
            scores.min() >= -1.0 - 1e-9 and scores.max() <= 1.0 + 1e-9
        ), "Cosine similarity must be bounded in [-1, 1]"
        print(
            f"✔ compute_similarity computed similarity array of shape ({num_attractions},) within valid cosine range"
        )

    # 6. Verify recommend_attractions
    print("\n--- Testing recommend_attractions ---")
    top_n = 5
    recommendations = recommend_attractions(
        tourist_id=1,
        interactions_df=interactions_df,
        attraction_df=attraction_df,
        tfidf_matrix=tfidf_matrix,
        attraction_index=attraction_index,
        top_n=top_n,
    )

    # If user profile was built successfully
    if profile_vector is not None:
        assert isinstance(recommendations, pd.DataFrame), "Output must be a DataFrame"
        assert (
            len(recommendations) <= top_n
        ), f"Should return at most {top_n} recommendations"

        # Verify schema
        expected_cols = [
            "attraction_uid",
            "attraction_name",
            "attraction_category",
            "city",
            "similarity_score",
            "rank",
        ]
        assert (
            list(recommendations.columns) == expected_cols
        ), f"Schema mismatch. Got: {list(recommendations.columns)}"

        # Verify visited attractions are excluded
        visited_uids = interactions_df[interactions_df["tourist_id"] == 1][
            "attraction_uid"
        ].tolist()
        recommended_uids = recommendations["attraction_uid"].tolist()
        intersection = set(visited_uids).intersection(set(recommended_uids))
        assert (
            len(intersection) == 0
        ), f"Recommendation includes already visited items: {intersection}"

        # Verify ranking order and rank column
        assert list(recommendations["rank"]) == list(
            range(1, len(recommendations) + 1)
        ), "Rank column must be 1-indexed sequential"
        scores_list = recommendations["similarity_score"].tolist()
        is_sorted_descending = all(
            scores_list[i] >= scores_list[i + 1] for i in range(len(scores_list) - 1)
        )
        assert (
            is_sorted_descending
        ), f"Scores are not sorted in descending order: {scores_list}"
        print(
            "✔ recommend_attractions successfully filtered visited items, sorted scores descending, and joined metadata"
        )
    else:
        # Check that empty DataFrame matches schema
        assert list(recommendations.columns) == [
            "attraction_uid",
            "attraction_name",
            "attraction_category",
            "city",
            "similarity_score",
            "rank",
        ]
        assert (
            recommendations.empty
        ), "Should return empty DataFrame for cold-start user"
        print(
            "✔ recommend_attractions correctly returned empty DataFrame matching schema for profile-less user"
        )

    print("\nAll Content-Based Filtering verification assertions passed! 🎉")


if __name__ == "__main__":
    test_content_based_recommendation()
