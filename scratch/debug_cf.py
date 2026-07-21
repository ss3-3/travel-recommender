from src.preprocessing import load_dataset, prepare_interactions, train_test_split_by_user
from pathlib import Path
from src.collaborative import (
    build_user_item_matrix,
    build_user_similarity_matrix,
    find_nearest_neighbors,
    predict_ratings,
)

BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / "data" / "tourism_recommendation_dataset_en.csv"
raw_df = load_dataset(str(data_path))
interactions_df = prepare_interactions(raw_df)
train_df, _ = train_test_split_by_user(interactions_df)

user_item_matrix, user_index, attraction_index = build_user_item_matrix(train_df)
user_similarity_matrix = build_user_similarity_matrix(user_item_matrix)

tourist_id = 5

neighbors = find_nearest_neighbors(
    tourist_id=tourist_id,
    user_similarity_matrix=user_similarity_matrix,
    user_index=user_index,
    k=20,
)

print("Neighbors:")
print(neighbors[:10])

predictions = predict_ratings(
    neighbors,
    user_item_matrix,
    user_index,
    attraction_index,
)

values = list(predictions.values())

print(f"Prediction count: {len(values)}")
print(f"Min: {min(values)}")
print(f"Max: {max(values)}")
print(f"Mean: {sum(values)/len(values)}")