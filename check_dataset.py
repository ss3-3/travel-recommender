from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "tourism_recommendation_dataset_en.csv"

df = pd.read_csv(DATA_PATH)

print(df.head())
print(df.columns.tolist())
print(df.shape)


print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df["tourist_id"].nunique())
print(df["attraction_name"].nunique())

# Attractions by category
print(df["attraction_category"].value_counts())

# Attractions by province
print(df["province"].value_counts())

# Rating statistics
print(df["rating"].describe())

# Rating distribution
print(df["rating"].value_counts().sort_index())

# Age group distribution
print(df["age_group"].value_counts())

# Gender distribution
print(df["gender"].value_counts())

# Season distribution
print(df["season"].value_counts())