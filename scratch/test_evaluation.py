"""
Verification tests for Phase 5 Recommendation System Evaluation.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root directory to sys.path to import src
SCRATCH_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRATCH_DIR.parent
sys.path.append(str(PROJECT_DIR))

from src.preprocessing import load_dataset, prepare_attractions, prepare_interactions, train_test_split_by_user
from src.content_based import build_content_column, build_tfidf_matrix
from src.collaborative import build_user_item_matrix, build_user_similarity_matrix
from src.evaluation import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    coverage,
    extract_ground_truth,
    evaluate_model,
    compare_models,
    run_full_evaluation,
)

DATA_PATH = PROJECT_DIR / "data" / "tourism_recommendation_dataset_en.csv"


def test_unit_metrics():
    print("Testing unit evaluation metrics...")

    # 1. precision_at_k
    p1 = precision_at_k(["A", "B", "C"], {"B", "C", "D"})
    assert abs(p1 - 2/3) < 1e-6, f"Expected 2/3 precision, got {p1}"

    p2 = precision_at_k([], {"A", "B"})
    assert p2 == 0.0, f"Expected 0.0 precision for empty list, got {p2}"

    p3 = precision_at_k(["A", "B"], {"C", "D"})
    assert p3 == 0.0, f"Expected 0.0 precision for zero hits, got {p3}"

    # 2. recall_at_k
    r1 = recall_at_k(["A", "B", "C"], {"B", "C", "D"})
    assert abs(r1 - 2/3) < 1e-6, f"Expected 2/3 recall, got {r1}"

    r2 = recall_at_k([], {"A", "B"})
    assert r2 == 0.0, f"Expected 0.0 recall for empty list, got {r2}"

    r3 = recall_at_k(["A", "B"], set())
    assert r3 == 0.0, f"Expected 0.0 recall for empty test set, got {r3}"

    # 3. f1_at_k
    f1 = f1_at_k(0.5, 0.5)
    assert f1 == 0.5, f"Expected 0.5 F1, got {f1}"

    f2 = f1_at_k(0.0, 0.0)
    assert f2 == 0.0, f"Expected 0.0 F1 for zero precision/recall, got {f2}"

    # 4. coverage
    c1 = coverage(8, 10)
    assert c1 == 0.8, f"Expected 0.8 coverage, got {c1}"

    c2 = coverage(0, 0)
    assert c2 == 0.0, f"Expected 0.0 coverage for empty test, got {c2}"

    # 5. extract_ground_truth
    mock_test = pd.DataFrame([
        {"tourist_id": 1, "attraction_uid": "A"},
        {"tourist_id": 1, "attraction_uid": "B"},
        {"tourist_id": 2, "attraction_uid": "C"}
    ])
    gt1 = extract_ground_truth(1, mock_test)
    assert gt1 == {"A", "B"}, f"Expected {{'A', 'B'}}, got {gt1}"

    gt2 = extract_ground_truth(2, mock_test)
    assert gt2 == {"C"}, f"Expected {{'C'}}, got {gt2}"

    print("✔ Unit metrics tests passed!")


def test_mock_evaluation_loop():
    print("\nTesting mock evaluation loop...")

    # Mock dataset
    mock_train = pd.DataFrame([
        {"tourist_id": 1, "attraction_uid": "X", "rating": 5.0},
        {"tourist_id": 2, "attraction_uid": "Y", "rating": 4.0}
    ])
    mock_test = pd.DataFrame([
        {"tourist_id": 1, "attraction_uid": "A", "rating": 5.0},
        {"tourist_id": 1, "attraction_uid": "B", "rating": 4.0},
        {"tourist_id": 2, "attraction_uid": "C", "rating": 4.0}
    ])

    # Mock recommend function
    # User 1 receives recommendations ["A", "D"] (1 hit out of 2 recommendations, test contains ["A", "B"])
    # User 2 is a cold-start user and receives empty recommendations
    def mock_recommend_fn(tourist_id, train_df, top_n, rating_threshold=4.0):
        if tourist_id == 1:
            return pd.DataFrame({"attraction_uid": ["A", "D"], "score": [4.5, 4.0]})
        else:
            return pd.DataFrame(columns=["attraction_uid"])

    summary_df, per_user_df = evaluate_model(
        recommend_fn=mock_recommend_fn,
        test_users=[1, 2],
        train_df=mock_train,
        test_df=mock_test,
        model_context={},
        top_n=10,
        model_kwargs={"rating_threshold": 4.0}
    )

    # Verify per-user metrics
    user_1_row = per_user_df[per_user_df["tourist_id"] == 1].iloc[0]
    assert user_1_row["precision_at_k"] == 0.5, f"Expected 0.5 precision, got {user_1_row['precision_at_k']}"
    assert user_1_row["recall_at_k"] == 0.5, f"Expected 0.5 recall, got {user_1_row['recall_at_k']}"
    assert user_1_row["f1_at_k"] == 0.5, f"Expected 0.5 F1, got {user_1_row['f1_at_k']}"
    assert user_1_row["hit_count"] == 1, f"Expected 1 hit, got {user_1_row['hit_count']}"
    assert user_1_row["recommended_count"] == 2, f"Expected 2 recs, got {user_1_row['recommended_count']}"

    user_2_row = per_user_df[per_user_df["tourist_id"] == 2].iloc[0]
    assert user_2_row["precision_at_k"] == 0.0
    assert user_2_row["recall_at_k"] == 0.0
    assert user_2_row["recommended_count"] == 0

    # Verify summary metrics
    summary = summary_df.iloc[0]
    assert summary["model_name"] == "mock_recommend_fn"
    assert summary["precision_at_k"] == 0.5  # Only user 1 was covered, so average is 0.5
    assert summary["recall_at_k"] == 0.5
    assert summary["f1_at_k"] == 0.5
    assert summary["coverage"] == 0.5
    assert summary["evaluated_user_count"] == 1
    assert summary["coverage_user_count"] == 1
    assert summary["total_test_user_count"] == 2

    print("✔ Mock model evaluation loop tests passed!")


def test_end_to_end_evaluation():
    print("\nTesting end-to-end evaluation using a sample of the dataset...")

    # Load actual data
    df = load_dataset(str(DATA_PATH))
    attraction_df = prepare_attractions(df)
    interactions_df = prepare_interactions(df)
    train_df, test_df = train_test_split_by_user(interactions_df, test_ratio=0.2, min_interactions=5)

    # 1. Prepare CBF context
    cbf_df = build_content_column(attraction_df)
    vectorizer, tfidf_matrix, attraction_index = build_tfidf_matrix(cbf_df)
    cbf_context = {
        "attraction_df": attraction_df,
        "tfidf_matrix": tfidf_matrix,
        "attraction_index": attraction_index
    }

    # 2. Prepare CF context
    user_item_matrix, user_index, cf_attraction_index = build_user_item_matrix(train_df)
    user_similarity_matrix = build_user_similarity_matrix(user_item_matrix)
    cf_context = {
        "attraction_df": attraction_df,
        "user_item_matrix": user_item_matrix,
        "user_similarity_matrix": user_similarity_matrix,
        "user_index": user_index,
        "attraction_index": cf_attraction_index
    }

    # Select a small slice of users who are present in the test set
    test_users = list(test_df["tourist_id"].unique()[:5])

    # Run evaluation orchestrator
    comparison_df = run_full_evaluation(
        train_df=train_df,
        test_df=test_df,
        test_users=test_users,
        cbf_context=cbf_context,
        cf_context=cf_context,
        top_n=10,
        cbf_kwargs={"rating_threshold": 4.0},
        cf_kwargs={"k": 20}
    )

    # Verify comparison shape and columns
    assert len(comparison_df) == 2, f"Expected 2 rows in comparison table, got {len(comparison_df)}"
    expected_cols = [
        "model_name", "precision_at_k", "recall_at_k", "f1_at_k",
        "coverage", "evaluated_user_count", "coverage_user_count", "total_test_user_count"
    ]
    for col in expected_cols:
        assert col in comparison_df.columns, f"Missing expected column '{col}' in comparison table"

    # Display comparison table
    print("\nSample Evaluation Results (K=10):")
    print(comparison_df.to_string(index=False))
    print("\n✔ End-to-end evaluation run completed successfully!")


if __name__ == "__main__":
    test_unit_metrics()
    test_mock_evaluation_loop()
    test_end_to_end_evaluation()
    print("\nAll Phase 5 Recommendation System Evaluation tests passed! 🎉")
