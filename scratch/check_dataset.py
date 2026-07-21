from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "tourism_recommendation_dataset_en.csv"

df = pd.read_csv(DATA_PATH)

print(df.head())
print(df.columns.tolist())
print(df.shape)


print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df["tourist_id"].nunique())
print(df["attraction_name"].nunique())