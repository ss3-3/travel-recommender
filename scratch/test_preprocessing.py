import sys
from pathlib import Path
import pandas as pd

# Add the project directory to sys.path to import src
SCRATCH_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRATCH_DIR.parent
sys.path.append(str(PROJECT_DIR))

from src.preprocessing import (
    load_dataset,
    validate_columns,
    clean_text,
    prepare_attractions,
    prepare_interactions
)

DATA_PATH = PROJECT_DIR / "data" / "tourism_recommendation_dataset_en.csv"

def test_preprocessing():
    print("Starting verification of src/preprocessing.py...")
    
    # 1. Test load_dataset
    print("\n--- Testing load_dataset ---")
    df = load_dataset(str(DATA_PATH))
    assert isinstance(df, pd.DataFrame), "Loaded data must be a pandas DataFrame"
    assert df.shape[0] == 100000, f"Expected 100,000 rows, got {df.shape[0]}"
    print("✔ load_dataset passed successfully!")

    # Test load_dataset error check
    try:
        load_dataset("non_existent_file.csv")
        assert False, "Should raise FileNotFoundError for missing file"
    except FileNotFoundError:
        print("✔ load_dataset correctly raises FileNotFoundError for missing files")

    # 2. Test validate_columns
    print("\n--- Testing validate_columns ---")
    required = ["tourist_id", "attraction_name", "rating"]
    validate_columns(df, required)
    print("✔ validate_columns successfully validated existing columns")

    # Should raise error for missing column
    try:
        validate_columns(df, ["non_existent_column"])
        assert False, "Should raise ValueError for missing column"
    except ValueError as e:
        print(f"✔ validate_columns correctly raised ValueError: {e}")

    # 3. Test clean_text
    print("\n--- Testing clean_text ---")
    sample_text = "Jing Zhou Gu Cheng ,  Mao Ling ,  Tai Yang Dao"
    cleaned = clean_text(sample_text)
    expected = "jing zhou gu cheng mao ling tai yang dao"
    assert cleaned == expected, f"Expected '{expected}', got '{cleaned}'"
    assert clean_text(None) == "", "Should return empty string for non-string input"
    print("✔ clean_text normalized text correctly")

    # 4. Test prepare_attractions
    print("\n--- Testing prepare_attractions ---")
    attractions = prepare_attractions(df)
    assert isinstance(attractions, pd.DataFrame), "Output must be a DataFrame"
    
    # Verify shape
    assert attractions.shape[0] == 433, f"Expected 433 unique attractions, got {attractions.shape[0]}"
    
    # Verify columns
    expected_cols = [
        'attraction_uid',
        'attraction_name',
        'attraction_category',
        'attraction_level',
        'city',
        'province',
        'ticket_price',
        'main_spots_clean'
    ]
    assert list(attractions.columns) == expected_cols, f"Columns mismatch. Got: {list(attractions.columns)}"
    assert 'season' not in attractions.columns, "Season feature must NOT be present in attraction features"
    
    # Verify that duplicate attraction names are resolved by attraction_uid
    bai_yun_shans = attractions[attractions['attraction_name'] == 'Bai Yun Shan']
    assert len(bai_yun_shans) == 2, f"Expected 2 physical attractions named Bai Yun Shan, got {len(bai_yun_shans)}"
    print("✔ prepare_attractions isolated 433 unique attractions and resolved duplicates")

    # 5. Test prepare_interactions
    print("\n--- Testing prepare_interactions ---")
    interactions = prepare_interactions(df)
    assert isinstance(interactions, pd.DataFrame), "Output must be a DataFrame"
    assert interactions.shape[0] == 100000, f"Expected 100,000 interactions, got {interactions.shape[0]}"
    
    # Verify columns
    expected_int_cols = ['tourist_id', 'attraction_uid', 'rating']
    assert list(interactions.columns) == expected_int_cols, f"Columns mismatch. Got: {list(interactions.columns)}"
    assert set(interactions['attraction_uid']).issubset(set(attractions['attraction_uid'])), "All attraction_uids in interactions must exist in attractions"
    print("✔ prepare_interactions isolated user-item ratings correctly")

    print("\nAll preprocessing verification checks passed successfully! 🎉")

if __name__ == "__main__":
    test_preprocessing()
