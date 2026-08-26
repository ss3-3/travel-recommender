"""
Preprocessing module for the Travel Destination Recommendation System.

This module handles data loading, cleaning, validation, and feature preparation 
for both Content-Based Filtering and Collaborative Filtering models. It implements 
the decisions from the Exploratory Data Analysis (EDA) phase.
"""

from pathlib import Path
import re
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Loads the raw dataset from a CSV file.

    Args:
        csv_path (str): The file path to the raw dataset.

    Returns:
        pd.DataFrame: The loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    # Read and return the CSV file
    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    """
    Validates that all required columns are present in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        required_columns (List[str]): List of required column names.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in dataset: {', '.join(missing_cols)}"
        )


def clean_text(text: Optional[str]) -> str:
    """
    Normalizes textual fields by converting to lowercase, removing punctuation, 
    and trimming whitespace. Designed as a reusable text normalization function 
    for future TF-IDF vectorization.

    Args:
        text (Optional[str]): The raw text string or None.

    Returns:
        str: Cleaned and normalized text.
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase and trim whitespace
    cleaned = text.lower().strip()

    # Replace punctuation with spaces
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)

    # Collapse multiple whitespaces into a single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def prepare_attractions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares cleaned unique attraction characteristics for the Content-Based Filtering model.

    This function:
    1. Replaces missing values in "main_spots" with empty strings (per EDA decision).
    2. Builds a unique attraction key (attraction_uid) by concatenating name, city, and province.
    3. Normalizes text fields.
    4. Extracts static, deduplicated attraction attributes using drop_duplicates.

    Args:
        df (pd.DataFrame): The raw tourism DataFrame.

    Returns:
        pd.DataFrame: Cleaned unique attractions DataFrame containing:
                      attraction_uid, attraction_name, attraction_category,
                      attraction_level, city, province, ticket_price, main_spots_clean.
    """
    # Required columns for attraction features
    required_cols = [
        "attraction_name",
        "attraction_category",
        "attraction_level",
        "city",
        "province",
        "ticket_price",
        "main_spots",
    ]
    validate_columns(df, required_cols)

    # Create a copy to prevent modifying the original DataFrame
    df_copy = df.copy()

    # Fill missing values in main_spots with empty string (expected for independent travelers)
    df_copy["main_spots"] = df_copy["main_spots"].fillna("")

    # Clean text in main_spots
    df_copy["main_spots_clean"] = df_copy["main_spots"].apply(clean_text)

    # Build unique identifier for attractions (attraction_uid) to resolve naming duplicates
    # Format: "Name (City, Province)"
    df_copy["attraction_uid"] = [
        f"{name} ({city}, {province})"
        for name, city, province in zip(
            df_copy["attraction_name"],
            df_copy["city"],
            df_copy["province"],
        )
    ]

    # Clean other categorical fields by trimming whitespace
    for col in ["attraction_category", "attraction_level", "city", "province"]:
        if df_copy[col].dtype == "object":
            df_copy[col] = df_copy[col].astype(str).str.strip()

    # Select only approved features (and name for display purposes)
    output_cols = [
        "attraction_uid",
        "attraction_name",
        "attraction_category",
        "attraction_level",
        "city",
        "province",
        "ticket_price",
        "main_spots_clean",
    ]

    # Deduplicate unique physical attractions based on their unique key
    attractions = df_copy[output_cols].drop_duplicates(subset=["attraction_uid"]).reset_index(drop=True)

    return attractions


def prepare_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the user-item interaction log for the Collaborative Filtering model.

    This function:
    1. Constructs the unique attraction key (attraction_uid) to align with attraction data.
    2. Isolates the interaction triplet: tourist_id, attraction_uid, and rating.

    Args:
        df (pd.DataFrame): The raw tourism DataFrame.

    Returns:
        pd.DataFrame: Cleaned interaction DataFrame with columns:
                      tourist_id, attraction_uid, rating.
    """
    # Required columns for interaction logs
    required_cols = ["tourist_id", "attraction_name", "city", "province", "rating"]
    validate_columns(df, required_cols)

    # Create a copy to prevent modifying the original DataFrame
    df_copy = df.copy()

    # Build the same unique identifier for attractions (attraction_uid)
    df_copy["attraction_uid"] = [
        f"{name} ({city}, {province})"
        for name, city, province in zip(
            df_copy["attraction_name"],
            df_copy["city"],
            df_copy["province"],
        )
    ]

    # Select columns representing interaction data
    output_cols = [
    "tourist_id",
    "attraction_uid",
    "rating",
    ]

    interactions = df_copy[output_cols].copy()

    # Combine repeated ratings from the same user for the same attraction
    interactions = (
        interactions
        .groupby(
            ["tourist_id", "attraction_uid"],
            as_index=False,
        )["rating"]
        .mean()
    )

    return interactions


def train_test_split_by_user(
    interactions_df: pd.DataFrame,
    test_ratio: float = 0.2,
    min_interactions: int = 5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits interactions into train and test sets independently for each user.

    For each tourist_id:
    - If the user has fewer than min_interactions, all their interactions
      are assigned to the training set.
    - If the user has at least min_interactions, approximately test_ratio of their
      interactions are randomly assigned to the test set, and the rest to train.

    Every interaction in interactions_df appears in exactly one of the split sets.

    Args:
        interactions_df (pd.DataFrame): Cleaned interaction DataFrame.
                                        Expected columns: tourist_id, attraction_uid, rating.
        test_ratio (float): Ratio of interactions to assign to the test set per user.
        min_interactions (int): Minimum interactions required to split a user's data.
        random_state (int): Seed for the random number generator.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df) containing:
                                            tourist_id, attraction_uid, rating.
    """
    # Validate required columns in interaction DataFrame
    required_cols = ["tourist_id", "attraction_uid", "rating"]
    validate_columns(interactions_df, required_cols)

    # Set up lists to accumulate row indices for train and test splits
    train_indices = []
    test_indices = []

    # Initialize numpy RandomState generator to guarantee reproducibility
    rng = np.random.RandomState(random_state)

    # Group by tourist_id to perform stratified splitting per user
    grouped = interactions_df.groupby("tourist_id", group_keys=False)

    for _, group in grouped:
        indices = group.index.tolist()
        n_interactions = len(indices)

        if n_interactions < min_interactions:
            # Users with few interactions have all their data retained in training set
            train_indices.extend(indices)
        else:
            # Determine size of test set for this user
            test_size = round(n_interactions * test_ratio)

            # Guarantee that we leave at least one interaction for training
            if test_size >= n_interactions:
                test_size = n_interactions - 1

            # Shuffle indices using the random generator to select items randomly
            shuffled = list(indices)
            rng.shuffle(shuffled)

            # Assign indices to train and test lists
            user_test = shuffled[:test_size]
            user_train = shuffled[test_size:]

            test_indices.extend(user_test)
            train_indices.extend(user_train)

    # Extract rows based on indices and reset indexing
    train_df = interactions_df.loc[train_indices].reset_index(drop=True)
    test_df = interactions_df.loc[test_indices].reset_index(drop=True)

    return train_df, test_df
