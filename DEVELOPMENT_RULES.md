# DEVELOPMENT_RULES.md

## Architecture Rules

- Never create new Python files unless explicitly instructed.
- Never create duplicate notebooks.
- Never rename existing modules.
- Never modify folder structure.
- Never introduce new recommendation algorithms.

---

## Notebook Rules

Only update:

01_preprocessing_analysis.ipynb
03_content_based.ipynb
04_collaborative.ipynb
05_evaluation.ipynb
06_itinerary_analysis.ipynb

Never create:

01_eda.ipynb
02_preprocessing.ipynb
02_content_based.ipynb
03_collaborative.ipynb
04_evaluation.ipynb

---

## Source Rules

Only modify existing modules.

Allowed:

preprocessing.py
content_based.py
collaborative.py
evaluation.py
app.py

Do not create:

recommendation.py
main.py
prototype.py
cf.py
cbf.py

---

## Recommendation Rules

This project contains ONLY TWO algorithms.

1. Content-Based Filtering
    TF-IDF + Cosine Similarity

2. User-Based Collaborative Filtering
    KNN + Cosine Similarity

Forbidden:

Hybrid
SVD
Matrix Factorization
ALS
Deep Learning
Neural CF
Autoencoder
Transformer

---

## Application Rules

Prototype MUST use Streamlit.

No CLI.

No Flask.

No FastAPI.

No Django.

No Gradio.

---

## Modification Rules

Update existing files only.

Never create duplicate files.

Never replace public function names.

Never change function signatures unless explicitly instructed.

---

## Documentation Rules

Whenever implementation changes:

Update

PROJECT_PROGRESS.md

AI_DECISIONS.md

README.md (if necessary)

Do not modify architecture documents.