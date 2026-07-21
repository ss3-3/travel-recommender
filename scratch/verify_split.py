import sys
from pathlib import Path
import pandas as pd

# Add the project directory to sys.path to import src
SCRATCH_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRATCH_DIR.parent
sys.path.append(str(PROJECT_DIR))

from src.preprocessing import (
    load_dataset,
    prepare_interactions,
    train_test_split_by_user
)

DATA_PATH = PROJECT_DIR / "data" / "tourism_recommendation_dataset_en.csv"

def verify_train_test_split():
    print("Starting verification of train_test_split_by_user...")
    
    # 1. Load raw data and extract interactions
    df = load_dataset(str(DATA_PATH))
    interactions = prepare_interactions(df)
    
    # To check for exact row assignment without being confused by duplicate values in the raw dataset,
    # we add a temporary unique 'interaction_id' column to track each row's destination.
    interactions = interactions.copy()
    interactions["interaction_id"] = range(len(interactions))
    
    # 2. Perform train-test split
    train_df, test_df = train_test_split_by_user(
        interactions,
        test_ratio=0.2,
        min_interactions=5,
        random_state=42
    )
    
    # 3. Calculate statistics
    total_count = len(interactions)
    train_count = len(train_df)
    test_count = len(test_df)
    
    # Calculate overlap using the unique interaction_id column
    overlap_ids = set(train_df["interaction_id"]).intersection(set(test_df["interaction_id"]))
    overlap_count = len(overlap_ids)
    
    num_users_train = train_df["tourist_id"].nunique()
    num_users_test = test_df["tourist_id"].nunique()
    
    # 4. Print metrics
    print(f"Total interactions: {total_count}")
    print(f"Training interactions: {train_count}")
    print(f"Testing interactions: {test_count}")
    print(f"Overlap count (by unique row index): {overlap_count}")
    print(f"Number of users in train: {num_users_train}")
    print(f"Number of users in test: {num_users_test}")
    
    # 5. Assertions to confirm correctness
    # Verify train + test == original interactions
    assert train_count + test_count == total_count, (
        f"Sum of splits ({train_count} + {test_count} = {train_count + test_count}) "
        f"does not match total interactions ({total_count})"
    )
    
    # Verify overlap count by index is exactly 0
    assert overlap_count == 0, f"Expected 0 overlap count, got {overlap_count}"
    
    # Verify that the union of split IDs equals the original set of IDs
    all_split_ids = set(train_df["interaction_id"]).union(set(test_df["interaction_id"]))
    assert len(all_split_ids) == total_count, "The split does not cover all original interactions"
    
    # Verify that there are no duplicate indices within the splits
    assert len(train_df["interaction_id"]) == len(set(train_df["interaction_id"])), "Train set contains duplicate row assignments"
    assert len(test_df["interaction_id"]) == len(set(test_df["interaction_id"])), "Test set contains duplicate row assignments"
    
    print("\nAll split assertions passed! The split is correct, fair, and reproducible. 🎉")

if __name__ == "__main__":
    verify_train_test_split()
