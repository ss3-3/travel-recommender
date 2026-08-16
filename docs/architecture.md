# Software Architecture Design
## Travel Destination Recommendation System using CBF and CF

This document describes the software architecture for the Final Year Project (FYP). The design is intentionally simple: three logical layers, a small `src/` package with single-purpose files, and a Streamlit app that calls those files directly.

---

## Overall Project Architecture

The repository is structured to separate concerns between UI logic, reusable recommendation backend modules, and exploratory research:

```
travel-recommender/
├── app.py                # Streamlit web application (UI layer)
├── src/                  # Reusable implementation modules
│   ├── preprocessing.py  # Data loading, cleaning, validation, train/test splitting
│   ├── content_based.py  # Content-Based Filtering recommendation logic
│   ├── collaborative.py  # Collaborative Filtering recommendation logic
│   └── evaluation.py     # Evaluation metrics and model comparison logic
├── notebooks/            # Notebooks used for analysis, visualisation, and reporting
│   ├── 01_preprocessing_analysis.ipynb # Exploratory Data Analysis & Preprocessing
│   ├── 03_content_based.ipynb # Content-Based Filtering development
│   ├── 04_collaborative.ipynb # Collaborative Filtering development
│   ├── 05_evaluation.ipynb   # Model evaluation and comparison plots
│   └── 06_itinerary_analysis.ipynb # Itinerary generation and evaluation plots
├── scratch/              # Verification scripts to test individual modules
│   ├── test_preprocessing.py
│   ├── test_content_based.py
│   ├── test_collaborative.py
│   └── test_evaluation.py (Phase 5, future)
├── models/               # (Empty) Trained model files are not persisted to disk
├── data/                 # Raw dataset folder
└── docs/                 # Documentation files
```

- **Notebooks** are reserved exclusively for exploratory data analysis (EDA), visualization, and generating evaluation reports.
- **`src/`** contains the production-grade, reusable implementation modules.
- **`scratch/`** contains verification scripts to run tests and assert code correctness during development.

---

## Overall System Architecture

The recommendation system logic is organised into three layers. Each layer only talks to the layer directly below it.

```
┌───────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                  │
│                          app.py (Streamlit)                │
│   - collects user input (tourist_id, algorithm, top_n)    │
│   - displays ranked recommendations directly               │
└───────────────────────────────┬───────────────────────────┘
                                 │
┌───────────────────────────────▼───────────────────────────┐
│                        MODEL LAYER                          │
│         content_based.py            collaborative.py        │
│   - builds attraction features/     - builds user-item      │
│     profiles, computes similarity     matrix, computes      │
│     to user profile (CBF)             similarity/prediction │
│   - outputs ranked list             - outputs ranked list    │
└───────────────────────────────┬───────────────────────────┘
                                 │
┌───────────────────────────────▼───────────────────────────┐
│                          DATA LAYER                          │
│                        preprocessing.py                      │
│   - loads CSV, cleans data, standardizes text,               │
│     constructs attraction composite IDs, and                 │
│     splits train/test per user                               │
└───────────────────────────────────────────────────────────┘

              evaluation.py sits alongside
              all layers as a shared support module
              (not part of the vertical data flow)
```

**Why this shape works for a viva:** each layer maps to one clear question a panel member might ask — "how is the data prepared," "how does each model score attractions," and "how does the user interact with it." You can point at one box per question.

---

## 2. Overall Workflow

1. Raw CSV is loaded and cleaned once.
2. Two independent feature representations are built from the same cleaned data: one for CBF (attraction content strings and TF-IDF feature vectors), one for CF (user-item interaction matrix).
3. Each model produces a ranked list of attractions for a given user, independently of the other.
4. The evaluation module measures both models on the same held-out test data, so their outputs are comparable.
5. Whichever model the user selects in the app, its ranked list is directly retrieved and displayed.
6. Streamlit displays the ranked recommendations to the user.

---

## 3. Data Flow

```
      tourism_recommendation_dataset_en.csv
                        │
                        ▼
    preprocessing.py: load_dataset()
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
prepare_attractions()          prepare_interactions()
        │                               │
        │                               ▼
        │                    train_test_split_by_user()
        │                               │
        │                       ┌───────┴───────┐
        │                       ▼               ▼
        │                    train_df        test_df
        │                       │               │
        ▼                       ▼               ▼
   content_based.py                     collaborative.py
   - build_content_column()             - build_user_item_matrix()
   - build_tfidf_matrix()               - build_user_similarity_matrix()
   - build_user_profile()               - find_nearest_neighbors()
   - compute_similarity()               - predict_ratings()
   - recommend_attractions()            - recommend_attractions_cf()
        │                               │
        └───────────────┬───────────────┘
                        ▼
             app.py (Streamlit App) / notebooks/05_evaluation.ipynb
```

Both models read from the *same* cleaned dataset, so the comparison in Chapter 5 of your report is fair — neither model gets differently prepared data.

`evaluation.py` sits off to the side: it takes the train/test split from `preprocessing.py` and the outputs of both models, and produces the comparison metrics used in the evaluation notebook.

---

## 4. Module Dependency Diagram

```
    preprocessing.py
         ▲
         │
         ├───────────────┐
         │               │
content_based.py   collaborative.py
         ▲               ▲
         │               │
         └───────┬───────┘
                 │
              app.py
```

`evaluation.py` sits off to the side, imported by `notebooks/05_evaluation.ipynb`, and imports/depends on `preprocessing.py`, `content_based.py`, and `collaborative.py`.

**Import rules to keep it clean:**
- `preprocessing.py` has no internal package dependencies.
- `content_based.py` and `collaborative.py` each import `preprocessing.py` (to reuse cleaning/feature functions). They do **not** import each other — this keeps the two models independent, which matters for your "fair comparison" argument.
- `app.py` imports `content_based.py` and `collaborative.py`. It never imports `evaluation.py` — evaluation is an offline research step, not a runtime feature of the app.
- `evaluation.py` imports `preprocessing.py`, `content_based.py`, and `collaborative.py`, and is used only from `notebooks/05_evaluation.ipynb`.

---

## 5. Responsibility of Each File

| File | Responsibility |
|---|---|
| `preprocessing.py` | All data loading, cleaning, text standardization, unique composite attraction ID creation, and train/test stratified splitting. |
| `content_based.py` | Recommendation pipeline using TF-IDF representation, rating-weighted user profiles, and cosine similarity. |
| `collaborative.py` | Recommendation pipeline using memory-based User-Based Collaborative Filtering (KNN) with user similarity and rating prediction. |
| `evaluation.py` | Evaluation metrics (precision@k, recall@k) and side-by-side comparison of the two models. |
| `app.py` | Streamlit UI: collects user input, calls the recommender modules, and renders recommendations. |

---

## 6. Recommended Functions per File

### `preprocessing.py`

| Function | Input | Output | Purpose |
|---|---|---|---|
| `load_dataset(csv_path)` | `csv_path: str` | raw `DataFrame` | Reads raw CSV into memory. |
| `validate_columns(df, required_columns)` | `df: DataFrame`, `required_columns: List[str]` | `None` | Validates that required columns exist in the DataFrame. |
| `clean_text(text)` | `text: str` | `str` | Normalizes text to lowercase, strips punctuation/whitespace. |
| `prepare_attractions(df)` | `df: DataFrame` | `DataFrame` | Cleans missing values, creates unique `attraction_uid` keys, standardizes text. |
| `prepare_interactions(df)` | `df: DataFrame` | `DataFrame` | Extracts `tourist_id`, `attraction_uid`, and `rating`. |
| `train_test_split_by_user(interactions_df, test_ratio, min_interactions, random_state)` | `interactions_df`, parameters | `train_df`, `test_df` | Stratified per-user train/test split. |

### `content_based.py`

| Function | Input | Output | Purpose |
|---|---|---|---|
| `build_content_column(attraction_df)` | unique attractions `DataFrame` | `DataFrame` with `content` column | Concatenates descriptive attributes, repeating category and level twice to increase their weights. |
| `build_tfidf_matrix(content_df)` | `DataFrame` with `content` | vectorizer, sparse matrix, index map | Fits TF-IDF vectorizer and builds sparse feature matrix for all attractions. |
| `build_user_profile(tourist_id, interactions_df, tfidf_matrix, attraction_index, rating_threshold)` | user details, interactions, matrices | user profile vector / `None` | Builds rating-weighted average TF-IDF vector of attractions user rated highly (>= threshold). |
| `compute_similarity(profile_vector, tfidf_matrix)` | profile vector, feature matrix | similarity scores `ndarray` | Calculates cosine similarity between user profile and all attractions. |
| `recommend_attractions(tourist_id, interactions_df, attraction_df, tfidf_matrix, attraction_index, top_n, rating_threshold)` | query parameters, matrices | ranked recommendations `DataFrame` | Public interface to generate top-N recommendations. |

### `collaborative.py`

| Function | Input | Output | Purpose |
|---|---|---|---|
| `build_user_item_matrix(train_df)` | training interactions `DataFrame` | sparse matrix, row/col index maps | Pivots interactions to construct sparse ratings matrix with NaNs for unobserved entries. |
| `build_user_similarity_matrix(user_item_matrix)` | sparse ratings matrix | user similarity matrix `ndarray` | Computes pairwise user-user cosine similarity matrix. |
| `find_nearest_neighbors(tourist_id, user_similarity_matrix, user_index, k)` | user details, similarity matrix, k | list of `(tourist_id, similarity)` | Retrieves top-k most similar users with positive similarity. |
| `predict_ratings(neighbors, user_item_matrix, user_index, attraction_index)` | neighbors, ratings matrix, index maps | predicted ratings `Dict[str, float]` | Computes similarity-weighted average predicted ratings for candidate attractions. |
| `recommend_attractions_cf(tourist_id, train_df, attraction_df, user_item_matrix, user_similarity_matrix, user_index, attraction_index, k, top_n)` | query parameters, matrices | ranked recommendations `DataFrame` | Public interface to generate top-N collaborative filtering recommendations. |

### `evaluation.py`

| Function | Input | Output | Purpose |
|---|---|---|---|
| `precision_at_k(recommended_attractions, actual_attractions)` | recommended list, actual test set | `float` | Computes precision@k for a user. |
| `recall_at_k(recommended_attractions, actual_attractions)` | recommended list, actual test set | `float` | Computes recall@k for a user. |
| `f1_at_k(precision, recall)` | precision, recall | `float` | Computes F1-score@k metric. |
| `coverage(evaluated_user_count, total_test_user_count)` | served user count, total user count | `float` | Computes coverage metric. |
| `evaluate_model(recommend_fn, test_users, train_df, test_df, model_context, top_n, model_kwargs)` | recommend_fn, parameters | summary & per-user DataFrames | Evaluates model metrics across test users. |
| `compare_models(cbf_summary, cf_summary)` | CBF & CF summary DataFrames | comparison `DataFrame` | Concatenates model results side-by-side. |

---

## 7. Execution Order of the Entire Project

**Development / research order:**

1. `notebooks/01_preprocessing_analysis.ipynb` — explore the raw CSV, decide cleaning rules and feature choices, and validate preprocessing outputs. Findings feed directly into `src/preprocessing.py`.
2. Build and test `preprocessing.py` functions inside the notebook or scratch files, then move finalised functions into the `src/` file.
3. Write and verify `src/preprocessing.py` using `scratch/test_preprocessing.py`.
4. Write and prototype recommendation scoring directly in the reusable modules `src/content_based.py` and `src/collaborative.py`.
5. Verify code correctness using `scratch/test_content_based.py` and `scratch/test_collaborative.py`.
6. Use `notebooks/05_evaluation.ipynb` to import both finished models from `src/`, run evaluations from `src/evaluation.py`, and produce the comparison tables/charts.

**Runtime order (when someone runs the finished app):**

1. `app.py` starts → calls `preprocessing.py` to prepare attraction metadata and interactions.
2. User enters a tourist ID, selects the recommendation algorithm (CBF or CF), and configures parameters in the Streamlit UI.
3. `app.py` computes matrices and calls `recommend_attractions()` or `recommend_attractions_cf()` depending on the algorithm.
4. `app.py` directly renders the ranked recommendations table.

---

## 8. How Streamlit Should Communicate with the Recommender Modules

Keep this direct — no API layer, no microservices, no message queue. That would be over-engineering for a single-user academic app.

```python
# app.py — illustrative only, not full code
import streamlit as st
from src import preprocessing, content_based, collaborative

@st.cache_data
def get_cleaned_data(csv_path):
    df = preprocessing.load_dataset(csv_path)
    attraction_df = preprocessing.prepare_attractions(df)
    interactions_df = preprocessing.prepare_interactions(df)
    return attraction_df, interactions_df

@st.cache_resource
def get_cbf_matrices(attraction_df):
    df_with_content = content_based.build_content_column(attraction_df)
    vectorizer, tfidf_matrix, attraction_index = content_based.build_tfidf_matrix(df_with_content)
    return tfidf_matrix, attraction_index

@st.cache_resource
def get_cf_matrices(train_df):
    user_item_matrix, user_index, attraction_index = collaborative.build_user_item_matrix(train_df)
    user_similarity = collaborative.build_user_similarity_matrix(user_item_matrix)
    return user_item_matrix, user_similarity, user_index, attraction_index
```

Two points matter here:

- **`st.cache_data` / `st.cache_resource`** are the only "infrastructure" you need. They stop Streamlit from recomputing the similarity matrices on every widget interaction, which would otherwise make the app feel slow. This is the single most important performance decision in the whole app.
- `app.py` should contain **no scoring logic** — only calls to functions in `src/`, plus `st.` display calls. If a viva panel member asks "where is the CF algorithm," you point to `collaborative.py`, not `app.py`. Keeping logic out of the UI file is what makes the modularity argument credible.

---

## 9. Keeping the Project Modular and Easy to Explain in the Viva

- **One file, one job.** If you ever find yourself writing recommendation-scoring code inside `app.py`, that's a sign it belongs in `content_based.py` or `collaborative.py` instead. Being able to say "every scoring decision lives in exactly one of two files" is a strong, simple answer to give a panel.
- **Write reusable modules directly.** Prototype algorithm ideas and code directly in their respective reusable files under `src/` rather than notebooks, keeping notebooks clean and reserved for visualization, EDA, and evaluation reports. Use `scratch/` verification scripts to run unit tests and check correctness.
- **Keep `evaluation.py` decoupled from `app.py`.** Evaluation is a research artifact for your report, not a feature of the deployed app. Conflating the two tends to confuse markers about what the app "does" versus what your experiment "measured."
-----------------------------------------

# Phase 2 — Preprocessing Pipeline Design (Revised)
## Travel Destination Recommendation System (CBF + User-Based CF)

Revised from Phase 1 EDA findings. All open questions from the previous draft have been resolved using confirmed EDA results; no `[EDA CHECK]` placeholders remain.

---

## 1. Data Flow

```
tourism_recommendation_dataset_en.csv
              │
              ▼
        Preprocessing
   (load → clean → standardize →
    attraction_id → attraction master →
    content column → user-item matrix →
    train/test split)
              │
              ▼
         Clean Dataset
  (attraction_df, user_item_matrix,
        train_df, test_df)
              │
        ┌─────┴─────┐
        ▼           ▼
       CBF          CF
        │           │
        └─────┬─────┘
              ▼
        Recommendation
      (ranked attraction list)
              │
              ▼
          Evaluation
   (precision@k, recall@k,
      CBF vs CF comparison)
              │
              ▼
         Streamlit App
```

---

## 2. Module Dependency

**Research / evaluation path:**
```
                     preprocessing.py
                    /                \
                   ▼                  ▼
        content_based.py        collaborative.py
                   \                  /
                    ▼                ▼
                     evaluation.py
```

**Runtime / app path:**
```
                     preprocessing.py
                    /                \
                   ▼                  ▼
        content_based.py        collaborative.py
                   \                  /
                    ▼                ▼
                       app.py
```

`evaluation.py` and `app.py` both depend on `content_based.py` and `collaborative.py`, but never on each other — evaluation is an offline research step, the app is the runtime deliverable.

---

## 3. Step-by-Step Preprocessing Design

### Step 1 — Load dataset
- **Input:** raw CSV path
- **Output:** raw `DataFrame` (~100,000 interaction rows)
- **Purpose:** single, consistent entry point for reading the data.
- **Why necessary:** every downstream module and both models must start from the same raw snapshot.
- **Used by:** CBF and CF (shared step).

### Step 2 — Clean missing values
- **Input:** raw `DataFrame`
- **Output:** `DataFrame` with only the EDA-confirmed missing columns handled
- **Purpose:** apply the minimum necessary cleaning, not speculative cleaning.
- **Confirmed during Phase 1 EDA:** the only columns with missing values are `group_fee`, `trip_days`, `main_spots`, and `transport_mode`. All other columns (including `ticket_price`) are complete — no imputation is applied to them.
- **Handling:**
  - `main_spots` → fill with `""`. This is the only field that needs an actual fill, because it feeds directly into the CBF content column in Step 6.
  - `group_fee`, `trip_days`, `transport_mode` → left as `NaN`. Neither CBF nor CF uses these fields in Phase 2, so there is nothing to compute — filling them now would be speculative work with no downstream consumer.
- **Used by:** CBF (`main_spots` only). CF does not use any of these four columns.

### Step 3 — Standardize text
- **Input:** cleaned `DataFrame`
- **Output:** `DataFrame` with trimmed, lowercased text in `attraction_category`, `attraction_level`, `city`, `province`, `season`, `main_spots`
- **Purpose:** prevent identical categories being treated as different due to casing/whitespace.
- **Why necessary:** directly affects both `attraction_id` construction (Step 4) and TF-IDF vocabulary quality (Step 6).
- **Used by:** CBF and CF.

### Step 4 — Create attraction ID
- **Input:** standardised `DataFrame`
- **Output:** `DataFrame` with new `attraction_id` column
- **Confirmed during Phase 1 EDA:** the dataset contains 431 unique attraction names but 433 physical attractions — two names are duplicated across different locations (**Bai Yun Shan**, **Hu Kou Pu Bu**), each referring to more than one physical site.
- **Decision:** `attraction_id` is built from `attraction_name + city + province`, not from `attraction_name` alone. This is necessary specifically because of the two confirmed duplicates — using name alone would incorrectly merge two distinct physical attractions into one identity.
- **Used by:** CBF and CF — this is the shared join key that keeps both models referring to the same set of physical attractions.

### Step 5 — Build attraction master table
This table is generated only for Content-Based Filtering.
Collaborative Filtering continues to use the interaction-level dataset.
- **Input:** `DataFrame` with `attraction_id` (interaction-level, ~100,000 rows)
- **Output:** `attraction_df`, one row per unique `attraction_id` — **433 rows**, not 431
- **Purpose:** collapse repeated interaction rows into one descriptive row per physical attraction, aggregating numeric fields (mean `ticket_price`, mean `rating`, mean `visit_duration_hours`) and keeping representative values for descriptive fields.
- **Why necessary:** CBF operates on attractions, not interactions. This step is also where the 431-names-vs-433-attractions distinction becomes concrete in the data — **one row equals one physical attraction**, not one name.
- **Used by:** CBF only.

### Step 6 — Build content column
- **Input:** `attraction_df` from Step 5
- **Output:** `attraction_df` with new `content` text column
- **Construction:** a single concatenation of `attraction_category`, `attraction_level`, `province`, `city`, `season`, `main_spots` (multi-value `season`/`main_spots` entries split into individual tokens), joined once into one lowercase string. No field is repeated or manually re-weighted — TF-IDF's own term-frequency weighting is left to do that job.
- **Why no manual repetition:** TF-IDF already computes term weight from frequency and document rarity. Manually repeating `attraction_category` or `attraction_level` would be an ad hoc adjustment layered on top of an algorithm that already handles weighting — harder to justify and unnecessary.
- **Used by:** CBF only.

### Step 7 — Build user-item rating matrix
- **Input:** `DataFrame` with `attraction_id` (interaction-level)
- **Output:** matrix, `tourist_id` × `attraction_id`, cell value = `rating`
- **Purpose:** produce the structure user-based CF needs for user-user similarity.
- **Similarity method (per AI_DECISIONS.md):** Cosine Similarity. This is fixed, not left open between cosine/Pearson.
- **Used by:** CF only.

### Step 8 — Train/test split by user
- **Input:** interaction-level `DataFrame`, `test_ratio`, `min_interactions`
- **Output:** `train_df`, `test_df`
- **Purpose:** per-user split so evaluation reflects "recommend attractions the user hasn't interacted with yet," rather than a global random split that could leak a user's history across both sets.
- **`min_interactions`:** left as a configurable parameter, not fixed in this architecture document. The default value will be set during implementation, once the actual per-user interaction-count distribution can be checked directly against `train_test_split_by_user`'s behaviour.
- **Used by:** CF (evaluation depends on this split). CBF is evaluated on the same split for a fair comparison.

---

## 4. Preprocessing Output Specification

### For Content-Based Filtering
| Output | Description |
|---|---|
| `attraction_df` | One row per physical attraction (433 rows), with `attraction_id`, descriptive fields, and aggregated numeric fields. |
| `content` column | Single concatenated text field per attraction, per Step 6. |
| TF-IDF input | The `content` column, passed as-is to `TfidfVectorizer` in `content_based.py` (Phase 3). |

### For User-Based Collaborative Filtering
| Output | Description |
|---|---|
| `attraction_id` | Same identifier as CBF (`attraction_name + city + province`) — keeps both models aligned to the same 433 physical attractions. |
| `user_item_matrix` | `tourist_id` × `attraction_id`, values = `rating`. |
| `train_df` / `test_df` | Per-user split, `min_interactions` configurable at implementation time. |

---

## 5. Function Design — `src/preprocessing.py`

No implementation code, per your instruction.

### `load_dataset(csv_path)`
- **Parameters:** `csv_path: str`
- **Returns:** raw `DataFrame`
- **Responsibility:** Reads raw CSV into memory.
- **Dependencies:** none.

### `validate_columns(df, required_columns)`
- **Parameters:** `df: DataFrame`, `required_columns: List[str]`
- **Returns:** `None`
- **Responsibility:** Validates that required columns exist in the DataFrame.
- **Dependencies:** none.

### `clean_text(text)`
- **Parameters:** `text: str`
- **Returns:** `str`
- **Responsibility:** Normalizes text to lowercase, strips punctuation/whitespace.
- **Dependencies:** none.

### `prepare_attractions(df)`
- **Parameters:** `df: DataFrame`
- **Returns:** cleaned attractions `DataFrame`
- **Responsibility:** Cleans missing values, creates unique `attraction_uid` keys, and standardizes text fields.
- **Dependencies:** `validate_columns`, `clean_text`.

### `prepare_interactions(df)`
- **Parameters:** `df: DataFrame`
- **Returns:** cleaned interactions `DataFrame`
- **Responsibility:** Extracts `tourist_id`, `attraction_uid`, and `rating`.
- **Dependencies:** `validate_columns`.

### `train_test_split_by_user(interactions_df, test_ratio, min_interactions, random_state)`
- **Parameters:** `interactions_df: DataFrame`, `test_ratio: float = 0.2`, `min_interactions: int = 5`, `random_state: int = 42`
- **Returns:** `train_df`, `test_df`
- **Responsibility:** Splits interactions into train and test sets independently for each user.
- **Dependencies:** `validate_columns`.

No caching functions or pipeline orchestrators are part of this design — the pipeline is cheap enough to re-run via importing the separate preparation functions directly as needed.

---

## 6. Design Decisions Confirmed

- ✓ `attraction_id` = `attraction_name + city + province` (431 names → 433 physical attractions; duplicates: Bai Yun Shan, Hu Kou Pu Bu)
- ✓ Missing values: `main_spots` → fill `""`; `group_fee`, `trip_days`, `transport_mode` → left as `NaN` (unused in Phase 2)
- ✓ `attraction_df` = one row per physical attraction (433 rows)
- ✓ Content column = single concatenation of category, level, province, city, season, main_spots — no manual repetition/weighting
- ✓ CF similarity method = Cosine Similarity (per AI_DECISIONS.md)
- ✓ No caching layer in Phase 2 — functions are executed dynamically on load
- ✓ `train_test_split_by_user`: `min_interactions` configurable, default fixed at implementation time

---

## 7. Documentation Updates

**`PROJECT_PROGRESS.md`** — suggested entry (I still don't have the actual file to edit directly):
```
## Phase 2 — Preprocessing Pipeline (Design)
Status: Design finalized, pending final confirmation of rating_column
Date: [fill in]
Notes: Pipeline designed around confirmed EDA findings (433 physical
attractions from 431 names; missing-value columns confirmed as group_fee,
trip_days, main_spots, transport_mode). 9 functions specified in
src/preprocessing.py, including single-entry-point preprocess_pipeline().
No caching layer. Implementation not yet started.
```

**`AI_DECISIONS.md`** — new decisions made in this revision, suggested entries:
```
- Decision: attraction_id built from attraction_name + city + province
  Reason: EDA confirmed 2 duplicate attraction names (Bai Yun Shan, Hu Kou
  Pu Bu) referring to different physical attractions; name alone is not
  a safe unique key.

- Decision: CBF content column uses a single concatenation of descriptive
  fields, with no manual field repetition/weighting.
  Reason: TF-IDF's own term-frequency weighting already handles this;
  manual repetition would be an unjustified ad hoc addition.

- Decision: no save/load caching layer in preprocessing.py.
  Reason: not needed at current project scale; adding it would raise
  a scope question in viva without a corresponding production need.
```

------------------------------------------------------

# Phase 3 — Content-Based Filtering Architecture
## Travel Destination Recommendation System

**Status:** Preprocessing (Phase 2) is finalized and treated as a fixed contract. This document designs only the Content-Based Filtering (CBF) module — `src/content_based.py` — which consumes preprocessing's outputs and produces ranked attraction recommendations.

---

## 1. Overview

The Content-Based Filtering module recommends attractions by matching the **attributes of attractions a user already rated highly** against the attributes of all other attractions in the dataset. It does not use other users' behaviour at all — every recommendation for a given user is derived purely from that user's own rating history and the descriptive content of attractions.

This gives CBF two properties that matter for this project's comparison against Collaborative Filtering:

- **No item cold-start problem.** A newly added attraction can be recommended immediately, since its eligibility depends only on its own attributes, not on other users having interacted with it.
- **Fully explainable output.** Every recommendation can be traced back to specific shared attributes (category, level, region, described spots) between the recommended attraction and what the user previously liked — a natural strength when discussing results in the report and viva.

The module's single responsibility is: given a `tourist_id`, produce a ranked list of attractions that user has not yet visited, ordered by content similarity to their known preferences.

---

## 2. Pipeline

```
Attraction Data
        │
        ▼
Build Content Column
        │
        ▼
TF-IDF Vectorizer
        │
        ▼
TF-IDF Matrix
        │
        │
User History ──────► Build User Profile
                     │
                     ▼
              Cosine Similarity
                     │
                     ▼
         Remove Visited Attractions
                     │
                     ▼
             Top-N Recommendations
```

This pipeline runs in two parts with different lifetimes:

- **Corpus-level steps** (content column → Build Content
 → TF-IDF Vectorizer → Attraction Feature Matrix
 → Build User Profile) are computed **once** for the whole attraction set — they don't depend on any specific user.
- **User-level steps** (user profile → similarity → top-N) are computed **per user**, reusing the item feature matrix built once above.

This separation matters for the function design in Section 10: the expensive step (TF-IDF fitting) is not repeated for every recommendation request.

---

## 3. Detailed Explanation of Every Step

### Step 1 — Preprocessed attractions
- **Input:** the Phase 2 attraction output — `attraction_uid`, `attraction_name`, `attraction_category`, `attraction_level`, `city`, `province`, `ticket_price`, `main_spots_clean`.
- **Output:** unchanged — this is the entry contract from preprocessing, not a transformation step.
- **Purpose:** fixed starting point for the CBF module.
- **Why necessary:** establishes that CBF depends only on the already-finalized preprocessing contract, not on any raw data directly.
- **Used by:** `content_based.py` (read-only input; not modified here).

### Step 2 — Build content column
- **Input:** the attraction table from Step 1.
- **Output:** the same attraction table with one new column, `content` — a single text string per attraction.
- **Purpose:** merge the relevant descriptive fields into one field TF-IDF can vectorize.
- **Why necessary:** TF-IDF requires one text field per document (per attraction here); the five source fields must be combined according to a single, fixed rule (Section 4).
- **Note on field normalization:** `main_spots_clean` is already normalized by preprocessing (Phase 2). The remaining fields used in `content` — `attraction_category`, `attraction_level`, `province`, `city` — are not separately cleaned by preprocessing; they are converted to string form and normalized (case, whitespace) at the point of content construction, inside `build_content_column()` itself. This keeps the architecture consistent with what preprocessing actually outputs, rather than assuming cleaning that hasn't happened upstream.
- **Used by:** `content_based.py`, function `build_content_column()`.

### Step 3 — TF-IDF Vectorization
- **Input:** the `content` column across all attractions (433 documents).
- **Output:** a fitted TF-IDF vectorizer and the resulting sparse term matrix.
- **Purpose:** convert the text corpus into a numeric vector space where each attraction is represented by weighted term scores.
- **Why necessary:** cosine similarity (Step 6) and profile averaging (Step 5) both require attractions to be represented as numeric vectors, not raw text.
- **Used by:** `content_based.py`, function `build_tfidf_matrix()`.

### Step 4 — Item Feature Matrix
- **Input:** output of Step 3.
- **Output:** the attraction × term matrix itself, held in memory for reuse across all users.
- **Purpose:** this is the shared representation every user profile and similarity computation is compared against.
- **Why necessary:** computing this once and reusing it for every user avoids re-fitting TF-IDF per recommendation request, which would be wasteful and would also risk each user being scored against a slightly different vocabulary/vector space.
- **Used by:** `content_based.py`, shared across `build_user_profile()` and `compute_similarity()`.

### Step 5 — Build User Profile
- **Input:** a `tourist_id`, that user's interaction history (from Phase 2's CF-facing interaction table — `tourist_id`, `attraction_uid`, `rating`), and the item feature matrix from Step 4.
- **Output:** a single vector in the same TF-IDF space, representing that user's content preference.
- **Purpose:** summarise "what this user likes" as one point in the same space attractions are represented in, so it can be compared to every attraction directly.
- **Why necessary:** CBF has no separate "user model" otherwise — the profile vector *is* the model of the user's taste. Full construction rule in Section 6.
- **Used by:** `content_based.py`, function `build_user_profile()`.

### Step 6 — Cosine Similarity
- **Input:** the user profile vector (Step 5) and the item feature matrix (Step 4).
- **Output:** one similarity score per attraction, for that user.
- **Purpose:** rank every candidate attraction by how closely it matches the user's profile.
- **Why necessary:** this is the actual scoring mechanism of CBF — everything before this step exists to produce the two vectors being compared here. Full justification in Section 7.
- **Used by:** `content_based.py`, function `compute_similarity()`.

### Step 7 — Top-N Recommendation
- **Input:** similarity scores (Step 6), the user's visited-attraction set (from interactions), `N`.
- **Output:** ranked list of `N` attraction recommendations for that user.
- **Purpose:** turn a raw similarity score for every attraction into a final, presentable, non-redundant recommendation list.
- **Why necessary:** without filtering, the top similarity results would likely include attractions the user has already visited (since they are, by construction, similar to their own profile) — these must be excluded before ranking is useful.
- **Used by:** `content_based.py`, function `recommend_attractions()`.

---

## 4. Content Design

The content composition is designed as a **single, configurable construction rule** inside `build_content_column()`, rather than being hardcoded across the module. This matters because the exact field composition is the most likely part of the design to be revisited during Phase 4 experimentation (e.g. testing the effect of removing `city`, or adding `ticket_price` as a bucketed token) — keeping composition confined to one function means such experiments do not require any change to the rest of the architecture.

The current, finalized composition is:

```
content =
    attraction_category
    attraction_level
    province
    city
    main_spots_clean
```

All fields are joined into one lowercase, space-separated string per attraction. `main_spots_clean` is used as-is from preprocessing; the categorical fields (`attraction_category`, `attraction_level`, `province`, `city`) are cast to string and normalized (case, whitespace) at this construction step, since preprocessing does not clean these fields itself — see the normalization note under Step 2.

---

## 5. TF-IDF Design

### Why TF-IDF is selected

- It converts variable-length descriptive text into a fixed-length numeric representation without requiring a pretrained language model or embedding service — appropriate for a 433-attraction, offline, academic-scale dataset.
- It automatically down-weights terms that appear across most attractions (e.g. very common category words) and up-weights terms that are distinctive to fewer attractions — which is exactly the discriminative behaviour content-based similarity needs.
- It is a long-established, well-understood technique, which makes the design easy to justify and explain in a viva compared to a black-box embedding model.

### Input
The `content` column — one string per attraction (433 documents total).

### Output
A sparse attraction × vocabulary-term matrix, where each row is that attraction's TF-IDF vector, and a fitted vectorizer object that defines the vocabulary and IDF weights for the corpus.

### Advantages
- Computationally lightweight at this dataset scale.
- Fully interpretable — each dimension corresponds to an actual vocabulary term, so a similarity score can be traced back to specific shared words.
- Naturally handles the repeated-term weighting design from Section 4 without extra logic.

### Limitations
- TF-IDF is a bag-of-words representation: it has no notion of semantic similarity between different words (e.g. "temple" and "shrine" are treated as entirely unrelated terms, even if conceptually similar).
- It ignores word order and context entirely.
- Similarity quality is sensitive to vocabulary sparsity — attractions with very short or generic `main_spots_clean` text will have thinner, less distinctive vectors.
- The design is dependent on which raw fields are included in `content`; changing that composition changes similarity behaviour without any change to the TF-IDF algorithm itself.

---

## 6. User Profile Design

**Rule:** the user profile is the **rating-weighted average of the TF-IDF vectors of attractions that user rated ≥ 4.0.**

Formally, for a user with rated attractions $\{a_1, a_2, \dots, a_k\}$ where each $a_i$ has rating $r_i \geq 4.0$ and TF-IDF vector $v_i$:

$$
\text{profile} = \frac{\sum_{i=1}^{k} r_i \cdot v_i}{\sum_{i=1}^{k} r_i}
$$

### Reasoning

The dataset uses a 1–5 rating scale. Ratings of 4 and 5 are interpreted as positive preference, following the common practice adopted in explicit-feedback recommender systems. Only positively-rated attractions (rating ≥ 4.0) are considered to represent user preferences. Lower ratings are ignored because they represent neutral or negative experiences rather than preferred travel interests, and including them would define "preference" using data that does not actually indicate preference.

- **Why filter to ≥ 4.0 only:** including low or neutral ratings would dilute the profile with attractions the user did not particularly enjoy, blurring the signal of what they actually prefer. Restricting to clearly-positive ratings keeps the profile focused on genuine preference rather than average behaviour.
- **Why weight by rating rather than treat all qualifying attractions equally:** within the ≥ 4.0 group, a 5.0 rating still represents stronger preference than a 4.0 rating. Weighting by rating lets that difference in preference strength carry through into the profile vector, rather than treating "liked" as a single binary state.
- **Users with no attractions rated ≥ 4.0:** cannot receive a CBF profile under this rule and fall outside CBF's coverage — this is a known and acceptable limitation of CBF (distinct from CF's user cold-start problem), and should be noted as a coverage boundary when reporting evaluation results in Phase 4, not silently patched over.

---

## 7. Similarity Computation

**Method:** Cosine Similarity between the user profile vector and each attraction's TF-IDF vector.

$$
\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\lVert \mathbf{A} \rVert \, \lVert \mathbf{B} \rVert}
$$

where $\mathbf{A}$ is the user profile vector and $\mathbf{B}$ is a candidate attraction's TF-IDF vector.

### Why cosine similarity is suitable here

- TF-IDF vectors vary in magnitude depending on content length and term weights; cosine similarity measures the **angle** between vectors rather than their magnitude, so an attraction with a longer or shorter `content` string is not unfairly penalised or favoured purely due to vector length.
- It is the standard similarity measure for high-dimensional, sparse vector spaces of exactly the kind TF-IDF produces.
- Since TF-IDF weights are non-negative, cosine similarity here is bounded in $[0, 1]$, which is straightforward to rank and interpret.

---

## 8. Recommendation Strategy

1. **Filter visited attractions.** Remove any `attraction_uid` already present in the user's interaction history from the candidate pool — a user should never be recommended an attraction they have already rated.
2. **Rank by similarity.** Sort the remaining candidates by descending cosine similarity score against the user's profile.
3. **Top-N recommendation.** Return the first `N` attractions from the ranked list as the final output. `N` is a configurable parameter of `recommend_attractions()`; the working default is **10**, consistent with a typical top-N evaluation setup for Phase 4.

**Edge case:** if fewer than `N` unseen attractions remain after filtering visited attractions, the function returns all available candidates rather than raising an error or padding the result — this can occur for users who have already interacted with a large share of the 433 attractions.

---

## 9. Output Specification

For each requested `tourist_id`, the CBF module produces a ranked table containing:

| Column | Description |
|---|---|
| `attraction_uid` | Recommended attraction's identifier |
| `attraction_name` | Joined back from the attraction table, for display |
| `attraction_category` | Joined back from the attraction table, for display |
| `city` | Joined back from the attraction table, for display |
| `similarity_score` | Cosine similarity value used for ranking |
| `rank` | Position in the top-N list (1 = most similar) |

Including `attraction_category` and `city` directly in the output means `app.py` can render a result table (e.g. Name / Category / City / Similarity) without a second lookup against the attraction table.

This output is consumed by:
- **`evaluation.py`** — to compute precision@k / recall@k against held-out test interactions.
- **`app.py`** — to display recommendations in the Streamlit interface.

---

## 10. Function Design — `src/content_based.py`

No implementation code, per your instruction.

### `build_content_column(attraction_df)`
- **Parameters:** `attraction_df` — the Phase 2 attraction table
- **Returns:** `attraction_df` with a new `content` column
- **Responsibility:** implement the exact field concatenation and repetition rule from Section 4.
- **Dependencies:** none beyond the finalized preprocessing output.

### `build_tfidf_matrix(attraction_df)`
- **Parameters:** `attraction_df` — output of `build_content_column()`
- **Returns:** the fitted TF-IDF vectorizer and the item feature matrix (attraction × term)
- **Responsibility:** fit TF-IDF once on the full `content` corpus. The fitted vectorizer must be reused for every subsequent user profile and recommendation request, not refitted per request — fit once, transform many. Refitting per request would risk each user being scored against a slightly different vocabulary, and would repeat unnecessary computation on every call.
- **Dependencies:** `build_content_column()`.

### `build_user_profile(tourist_id, interactions_df, tfidf_matrix, attraction_index, rating_threshold)`
- **Parameters:** target user id; the CF-facing interaction table (`tourist_id`, `attraction_uid`, `rating`); the item feature matrix; a mapping from `attraction_uid` to matrix row position; `rating_threshold` — a **configurable parameter**, default **4.0**. Keeping this configurable rather than fixed means that if Phase 4 evaluation later shows a different threshold (e.g. 3.5) performs better, only the parameter value changes — the architecture itself does not need to change.
- **Returns:** a single profile vector in the same vector space as the item feature matrix
- **Responsibility:** select the user's qualifying rated attractions and compute the rating-weighted average per Section 6.
- **Dependencies:** `build_tfidf_matrix()`.

### `compute_similarity(profile_vector, tfidf_matrix)`
- **Parameters:** a user profile vector; the item feature matrix
- **Returns:** one similarity score per attraction
- **Responsibility:** compute cosine similarity between the profile vector and every attraction vector, per Section 7.
- **Dependencies:** `build_user_profile()`, `build_tfidf_matrix()`.

### `recommend_attractions(tourist_id, interactions_df, attraction_df, tfidf_matrix, attraction_index, top_n)`
- **Parameters:** target user id; interaction table; attraction table (for name lookup); item feature matrix; `attraction_uid`-to-row mapping; `top_n` (default 10)
- **Returns:** ranked `DataFrame` per the Output Specification (Section 9)
- **Responsibility:** single entry point — orchestrates `build_user_profile()`, `compute_similarity()`, visited-attraction filtering, ranking, and top-N selection. This is the function `evaluation.py` and `app.py` call; they do not call the lower-level functions directly.
- **Dependencies:** all functions above.

**Public interface note:** `build_content_column()`, `build_tfidf_matrix()`, `build_user_profile()`, and `compute_similarity()` are internal helper functions. External modules (`evaluation.py`, `app.py`) should invoke only `recommend_attractions()`, which acts as the public interface of the Content-Based Filtering module. This keeps the module's external contract to a single function, regardless of how the internal pipeline steps are implemented.

---

## 11. Cold Start Discussion

- **New attraction:** no issue. Its attributes (`attraction_category`, `attraction_level`, `province`, `city`, `main_spots_clean`) are available as soon as it is added to the attraction table, so a TF-IDF vector can be generated for it immediately without requiring any prior user interaction. CBF is inherently free of the item cold-start problem.
- **New user:** a user without historical ratings cannot build a user profile, since `build_user_profile()` has no rated attractions to average over. Therefore, personalized CBF recommendations cannot be generated for a user until they have accumulated sufficient rating history (specifically, at least one attraction rated ≥ the `rating_threshold`). This is a genuine limitation of CBF, distinct from and complementary to CF's own user cold-start behaviour, and should be reported as such rather than worked around silently.

---

## 12. Documentation Updates

**`PROJECT_PROGRESS.md`** — suggested entry:
```
## Phase 3 — Content-Based Filtering (Design)
Status: Design finalized, ready for implementation
Notes: CBF pipeline designed on top of finalized Phase 2 preprocessing
output. Content column, TF-IDF, user profile, cosine similarity, and
top-N recommendation strategy fully specified in
Phase3_ContentBasedFiltering_Design.md. 5 core functions specified in
src/content_based.py. Implementation not yet started.
```

**`AI_DECISIONS.md`** — new/updated entries:
```
- Decision (SUPERSEDES Phase 2 entry): CBF content column repeats
  attraction_category and attraction_level twice each.
  Reason: without repetition, longer or more specific fields
  (main_spots_clean, city) can dominate TF-IDF similarity scoring
  independent of attraction type. Repetition raises the relative
  term-frequency weight of category/level to keep them the dominant
  similarity signal. This explicitly reverses the "no manual
  repetition" decision logged during the Phase 2 revision.

- Decision: user profile = rating-weighted average of TF-IDF vectors
  for attractions rated >= 4.0 by the user.
  Reason: restricting to >= 4.0 keeps the profile focused on genuine
  preference; weighting by rating preserves the difference in
  preference strength between a 5.0 and a 4.0 rating.

- Decision: similarity method = Cosine Similarity between user profile
  vector and attraction TF-IDF vectors.
  Reason: standard, interpretable measure for sparse high-dimensional
  TF-IDF vector spaces; robust to differences in vector magnitude.

- Decision: default Top-N = 10.
  Reason: consistent with a standard top-N evaluation setup planned
  for Phase 4.

- Known limitation logged: users with no attraction rated >= 4.0 have
  no CBF profile and fall outside CBF's coverage. To be reported as a
  coverage boundary in Phase 4 evaluation, not silently patched.

- Decision: content field composition is implemented as a single,
  isolated construction rule inside build_content_column(), and
  rating_threshold in build_user_profile() is a configurable parameter
  (default 4.0), not a hardcoded value.
  Reason: both are the most likely parameters to be revisited during
  Phase 4 experimentation; isolating them means such experiments
  change one parameter/function, not the surrounding architecture.

- Decision: recommend_attractions() is the only public function of
  content_based.py; build_content_column(), build_tfidf_matrix(),
  build_user_profile(), and compute_similarity() are internal helpers.
  Reason: keeps the module's external contract to a single call for
  evaluation.py and app.py, regardless of internal implementation.

- Cold-start behaviour documented: CBF has no item cold-start problem
  (new attractions are vectorizable immediately) but does have a user
  cold-start limitation (a user needs at least one rating >= threshold
  before a profile — and therefore personalized recommendations — can
  be generated). Logged for direct reuse in the Chapter 4 discussion.
  This limitation is expected and will be reflected in the evaluation coverage rather than treated as a system failure.
```

-------------------------------------------------

# Phase 4 — User-Based Collaborative Filtering Architecture

**Status:** This document is the sole authoritative architecture for Phase 4. It **rejects and replaces** the previously produced SVD/matrix-factorization design (`Phase4_CollaborativeFiltering_SVD_Design.md`), which is now void. It supersedes and consolidates the earlier User-Based KNN design (`Phase4_CollaborativeFiltering_Design.md`) into one complete, from-scratch specification. Phases 1–3 (EDA, Preprocessing, Content-Based Filtering) remain finalized and unchanged — nothing in this document redesigns them.

**Explicitly excluded from this design:** Singular Value Decomposition, matrix factorization, latent factors, matrix reconstruction, `svds()`, `TruncatedSVD`, or any other dimensionality-reduction technique. Every prediction in this design is computed directly from observed ratings and user-to-user similarity — no learned latent representation is used anywhere.

---

## 1. Module Responsibility

`src/collaborative.py` has one responsibility: given a `tourist_id`, produce a ranked list of attractions that user has not yet visited, using **only** the rating behaviour of similar users. It has the following boundaries:

- It does **not** read attraction attributes (`attraction_category`, `attraction_level`, `main_spots_clean`, etc.) — those belong to `content_based.py` and are irrelevant to this module.
- It does **not** import from or depend on `content_based.py`. The two CF/CBF modules are independent, so their comparison in Phase 5 is fair (neither influences the other).
- It does **not** modify or re-implement anything from `preprocessing.py`. It consumes `prepare_interactions()` and `train_test_split_by_user()` outputs exactly as Phase 2 finalized them.
- It produces output in the **same shape** as `content_based.py`'s public function, so `evaluation.py` and `app.py` can treat both models through one consistent interface.

---

## 2. Pipeline / Data Flow

```
Train Interactions (train_df, from Phase 2)
              │
              ▼
      Build User-Item Matrix
              │
              ▼
   Compute User Similarity
     (Cosine Similarity)
              │
              ▼
     Find Top-K Neighbours
              │
              ▼
   Predict Ratings (Weighted Average)
              │
              ▼
   Remove Visited Attractions
              │
              ▼
      Rank Candidates
              │
              ▼
    Top-N Recommendations
```

Two lifetimes, as with all previous phases:

- **Corpus-level, computed once:** user-item matrix, user similarity matrix.
- **Per-user, computed on request:** neighbour selection, rating prediction, filtering, ranking, Top-N.

---

## 3. Detailed Step Explanation

### Step 1 — Train interactions
- **Input:** `train_df` — Phase 2's `train_test_split_by_user()` training split (`tourist_id`, `attraction_uid`, `rating`).
- **Output:** unchanged.
- **Purpose:** fixed entry contract; CF must never train on `test_df`, preserving the same evaluation boundary CBF uses.
- **Function:** none — external Phase 2 contract, consumed directly by `build_user_item_matrix()`.

### Step 2 — Build user-item matrix
- **Input:** `train_df`.
- **Output:** a sparse `tourist_id` × `attraction_uid` matrix; cells hold the observed `rating` where an interaction exists, and are left unobserved (not zero-filled) otherwise.
- **Purpose:** represent every user as a vector of their known ratings, comparable to every other user.
- **Why necessary:** this is the only representation user similarity (Step 3) needs — it deliberately contains no attraction attributes.
- **Function:** `build_user_item_matrix()`.

### Step 3 — Compute user similarity (Cosine Similarity)
- **Input:** the user-item matrix from Step 2.
- **Output:** a `tourist_id` × `tourist_id` cosine similarity matrix.
- **Purpose:** quantify how alike any two users' rating behaviour is, using only their observed, overlapping ratings.
- **Why necessary:** Cosine Similarity is the mechanism that replaces attribute similarity entirely — it is computed and reused for every subsequent recommendation request, not recomputed per user.
- **Function:** `build_user_similarity_matrix()`.

### Step 4 — Find Top-K neighbours
- **Input:** a target `tourist_id`, the similarity matrix from Step 3, `K`.
- **Output:** the target user's top-`K` most similar users and their similarity scores.
- **Purpose:** restrict the comparison set to genuinely informative users, rather than using every other user regardless of similarity strength.
- **Function:** `find_nearest_neighbors()`.

### Step 5 — Predict ratings (weighted average)
- **Input:** the target user's neighbour set (Step 4), the user-item matrix (Step 2).
- **Output:** a predicted rating for every attraction at least one neighbour has rated.
- **Purpose:** estimate how the target user would rate attractions they haven't interacted with, using their neighbours' actual ratings, weighted by similarity.
- **Function:** `predict_ratings()`.

### Step 6 — Remove visited attractions
- **Input:** predicted ratings (Step 5), the target user's attractions in `train_df`.
- **Output:** predicted ratings restricted to attractions the user has not yet rated in training.
- **Purpose:** avoid recommending attractions the user already knows; `test_df` attractions are deliberately **not** excluded, since Phase 5 checks whether they are successfully recommended.
- **Function:** `recommend_attractions_cf()`.

### Step 7 — Rank candidates
- **Input:** filtered predicted ratings (Step 6).
- **Output:** candidates sorted by predicted rating, descending.
- **Purpose:** order candidates by relevance before truncating to the final list.
- **Function:** `recommend_attractions_cf()`.

### Step 8 — Top-N recommendations
- **Input:** ranked candidates (Step 7), `N`.
- **Output:** final ranked list of `N` attractions.
- **Purpose:** produce the presentable output consumed by `evaluation.py` and `app.py`.
- **Function:** `recommend_attractions_cf()`.

---

## 4. User-Item Matrix Design (recap, consistent with prior CF design)

| Aspect | Design |
|---|---|
| Rows | `tourist_id` — one row per user present in `train_df` |
| Columns | `attraction_uid` — one column per attraction present in `train_df` |
| Cell values | Observed `rating`, where present |
| Missing values | Left unobserved — never zero-filled, never imputed |
| Density | ~2.3% (≈100,000 interactions ÷ (10,000 users × 433 attractions)) |

Missing values are **not** imputed here (unlike the now-rejected SVD design) — this matrix is used directly, in sparse form, for cosine similarity, which only requires the overlapping non-missing entries of two vectors. No dense reconstruction of the matrix is performed at any point in this design.

---

## 5. Similarity, Neighbour Selection, and Prediction (recap, consistent with prior CF design)

### Cosine Similarity

$$
\text{sim}(u, v) = \frac{\mathbf{r_u} \cdot \mathbf{r_v}}{\lVert \mathbf{r_u} \rVert \, \lVert \mathbf{r_v} \rVert}
$$

Chosen over Pearson Correlation because, at ~2.3% matrix density, most user pairs share very few co-rated attractions — Pearson's mean-centering becomes undefined or unstable under this sparsity, while cosine similarity remains well-defined and degrades gracefully. This is consistent with the Cosine Similarity decision already logged in `AI_DECISIONS.md`.

### Neighbour selection

Top-`K` most similar users, `K` configurable (default **20**), excluding the target user themself and any neighbour with similarity ≤ 0 or undefined similarity (e.g. from a zero-norm rating vector). Ties at the cutoff boundary are broken deterministically by ascending `tourist_id`, for reproducibility.

### Rating prediction

$$
\hat{r}(u, i) = \frac{\sum_{v \in N(u,i)} \text{sim}(u, v) \cdot r(v, i)}{\sum_{v \in N(u,i)} \lvert \text{sim}(u, v) \rvert}
$$

where $N(u,i)$ is the subset of user $u$'s top-`K` neighbours who have actually rated attraction $i$. Weighting by similarity (rather than a plain average) ensures more similar neighbours have proportionally more influence — this is what makes the prediction personalized. No matrix reconstruction, latent vectors, or dimensionality reduction are involved at any point in this formula — every term is either an observed rating or a directly computed similarity score.

---

## 6. Inputs and Outputs Summary

| Function | Inputs | Outputs |
|---|---|---|
| `build_user_item_matrix()` | `train_df` | sparse matrix, `tourist_id`→row index map, `attraction_uid`→column index map |
| `build_user_similarity_matrix()` | user-item matrix | `tourist_id` × `tourist_id` cosine similarity matrix |
| `find_nearest_neighbors()` | `tourist_id`, similarity matrix, user index, `K` | ordered list of (neighbour `tourist_id`, similarity) pairs |
| `predict_ratings()` | `tourist_id`, neighbour list, user-item matrix, attraction index | predicted rating per attraction with ≥1 contributing neighbour |
| `recommend_attractions_cf()` | `tourist_id`, `train_df`, `attraction_df`, similarity matrix, index maps, `K`, `top_n` | ranked `DataFrame`: `attraction_uid`, `attraction_name`, `predicted_rating`, `rank`, optional `city`/`attraction_category` |

---

## 7. Function Responsibilities

### `build_user_item_matrix(train_df)`
- **Responsibility:** pivot `train_df` into the sparse matrix CF needs. Operates only on `train_df` — never the full interaction set or `test_df`.
- **Dependencies:** Phase 2's `train_test_split_by_user()` output.

### `build_user_similarity_matrix(user_item_matrix)`
- **Responsibility:** compute cosine similarity between every pair of users once; the result is reused across all subsequent recommendation requests (fit once, reuse many).
- **Dependencies:** `build_user_item_matrix()`.

### `find_nearest_neighbors(tourist_id, user_similarity_matrix, user_index, k)`
- **Responsibility:** retrieve the target user's top-`k` valid neighbours, applying the exclusion and tie-break rules in Section 5.
- **Dependencies:** `build_user_similarity_matrix()`.

### `predict_ratings(tourist_id, neighbors, user_item_matrix, attraction_index)`
- **Responsibility:** implement the weighted-average prediction formula (Section 5) for every attraction with at least one contributing neighbour.
- **Dependencies:** `find_nearest_neighbors()`, `build_user_item_matrix()`.

### `recommend_attractions_cf(tourist_id, train_df, attraction_df, user_item_matrix, user_similarity_matrix, user_index, attraction_index, k, top_n)`
- **Responsibility:** single public entry point — orchestrates neighbour selection, prediction, visited-attraction filtering, ranking, and Top-N selection, joining display fields from `attraction_df`.
- **Dependencies:** all functions above.
- **Naming:** deliberately distinct from CBF's `recommend_attractions()` to avoid ambiguity if both modules are imported unqualified into the same namespace (e.g. in `app.py`).

**Public interface:** only `recommend_attractions_cf()` should be called by `evaluation.py` or `app.py`. The other four functions are internal to `collaborative.py`.

---

## 8. Validation Strategy

Before or during implementation, the following should be checked:

- **Matrix construction:** every `tourist_id` and `attraction_uid` in `train_df` maps to exactly one row/column index; no duplicate index entries.
- **No leakage:** `build_user_item_matrix()` must never receive `test_df` or the full unsplit interaction table — this should be verifiable by asserting the function's input row count matches `train_df`'s row count, not the full dataset's.
- **Similarity matrix sanity:** the similarity matrix must be square (`tourist_id` × `tourist_id`), symmetric (`sim(u,v) == sim(v,u)`), and have a diagonal of 1.0 (a user's similarity to themself).
- **Neighbour set validity:** `find_nearest_neighbors()` must never return the target user as their own neighbour, and must never return more than `K` neighbours.
- **Prediction range sanity:** predicted ratings should fall within the dataset's actual rating scale (e.g. if ratings are 1–5, a predicted value far outside that range indicates a computation error, since a weighted average of valid ratings cannot exceed the range of its inputs).
- **Output filtering correctness:** no `attraction_uid` present in the target user's `train_df` history should ever appear in `recommend_attractions_cf()`'s output.
- **Output size correctness:** the output row count must be ≤ `top_n`, and equal to `top_n` only when at least `top_n` valid, unvisited, predictable attractions exist for that user (see edge cases below).

---

## 9. Edge Cases

- **User with zero or near-zero ratings in `train_df`.** May produce an empty or very small neighbour set. `find_nearest_neighbors()` should return an empty list rather than error; `recommend_attractions_cf()` should return an empty `DataFrame` rather than substituting a fallback score (e.g. popularity-based). This is a deliberate, reported cold-start limitation (Section 11), not a silent workaround.
- **No neighbours have rated a given candidate attraction.** That attraction is excluded from `predict_ratings()`'s output entirely — it is not defaulted to any value (e.g. global mean), since doing so would introduce exactly the kind of synthetic signal this design deliberately avoids by rejecting imputation-based approaches.
- **Fewer than `top_n` valid, unvisited, predictable attractions exist.** `recommend_attractions_cf()` returns all available candidates rather than padding the result or raising an error.
- **Tied similarity scores at the `K`-th cutoff.** Broken deterministically by ascending `tourist_id`, so neighbour sets — and therefore predictions — are reproducible across runs.
- **A user who has rated every attraction in the dataset.** No unvisited attractions remain; `recommend_attractions_cf()` returns an empty result, which is a valid and expected output, not an error.
- **Zero-norm rating vectors.** A user with exactly one rating, or with all-identical rating values across very few attractions, can produce an undefined cosine similarity against certain other users; such pairs are treated as similarity 0 and excluded from neighbour selection (Section 5), not as an error condition.

---

## 10. Evaluation Compatibility with Phase 5

- **Same train/test boundary as CBF.** Both models are trained on `train_df` and evaluated against `test_df`; `evaluation.py` calls `recommend_attractions_cf()` for each user in the same way it calls CBF's `recommend_attractions()`, so precision@k and recall@k are computed identically for both.
- **Identical output schema.** `recommend_attractions_cf()`'s output columns (`attraction_uid`, `attraction_name`, `predicted_rating`, `rank`, optional `city`/`attraction_category`) mirror CBF's output shape exactly (`similarity_score` there corresponds to `predicted_rating` here as the ranking column) — `evaluation.py` can treat both models' outputs through one shared comparison function rather than writing separate logic per model.
- **Coverage reporting.** Because some users will receive an empty or short recommendation list (Section 9), Phase 5 evaluation should report **coverage** — the percentage of test users for whom a full Top-`N` list was produced — for CF, exactly as was planned for CBF's own ≥4.0-rating coverage boundary. This gives a direct, comparable coverage statistic between the two models in the results chapter.
- **No shared state between models during evaluation.** Since `collaborative.py` does not import `content_based.py` (Section 1), evaluating both models back-to-back in `evaluation.py` cannot cause one model's internal state to leak into the other's results.

---

## 11. Consistency with Previous Phases

- **Preprocessing (Phase 2):** consumes `prepare_interactions()` and `train_test_split_by_user()) exactly as finalized — no changes, no re-implementation, no bypassing of the train/test split.
- **Content-Based Filtering (Phase 3):** independent module, no cross-imports; output schema deliberately mirrors CBF's for downstream consistency; both modules use the same "single public function, internal helpers" boundary pattern (`recommend_attractions()` for CBF, `recommend_attractions_cf()` for CF).
- **Similarity metric:** Cosine Similarity, consistent with the decision already logged in `AI_DECISIONS.md` during the Phase 2 revision, and reaffirmed here.
- **No SVD, no matrix factorization, no latent factors, no dimensionality reduction, no `svds()`/`TruncatedSVD`.** Every value in this design is either an observed rating or a similarity score computed directly from observed ratings. The previously produced SVD document is void and should not be treated as part of the project's architecture.

---

## 12. Documentation Updates

**`PROJECT_PROGRESS.md`** — suggested entry:
```
## Phase 4 — User-Based Collaborative Filtering (Design, FINAL)
Status: Design finalized, ready for implementation
Notes: Phase 4 confirmed as User-Based CF using Cosine Similarity,
Top-K neighbour selection, and similarity-weighted rating prediction.
The previously explored SVD/matrix-factorization variant is rejected
and superseded. Full architecture in
Phase4_UserBasedCF_Design_FINAL.md. 5 functions specified in
src/collaborative.py, with recommend_attractions_cf() as the sole
public interface. Implementation not yet started.
```

**`AI_DECISIONS.md`** — new entries:
```
- Decision (REJECTS prior SVD entry): Phase 4 Collaborative Filtering
  is confirmed as User-Based CF with Cosine Similarity and Top-K
  neighbour-weighted rating prediction. SVD / matrix factorization /
  latent factors / dimensionality reduction are explicitly excluded
  from this project's architecture.
  Reason: project requirement confirmed that only memory-based
  User-Based CF is in scope; the SVD variant explored previously
  violated this requirement and is void.

- Decision: recommend_attractions_cf() is the disambiguated public
  function name for collaborative.py, distinct from
  content_based.py's recommend_attractions().
  Reason: avoids name collision risk if both modules are imported
  unqualified into the same namespace (e.g. app.py).

- Decision: users with zero valid neighbours, or attractions with
  zero contributing neighbours, receive no prediction — no fallback
  or imputed value is substituted in either case.
  Reason: keeps CF's coverage limitations explicit and reportable in
  Phase 5, consistent with how CBF's own coverage boundary is
  reported, and consistent with rejecting imputation-based approaches
  (as used in the now-void SVD design) throughout this module.
```

---------------------------------------------------

# Phase 5 — Recommendation System Evaluation Architecture
## Travel Destination Recommendation System

**Status:** Phases 1–4 (EDA, Preprocessing, Content-Based Filtering, User-Based Collaborative Filtering) are finalized and treated as fixed contracts. This document designs the evaluation module — `src/evaluation.py` — which compares exactly two models: CBF (TF-IDF + Cosine Similarity) and User-Based CF (KNN + Cosine Similarity). There is no hybrid recommender, no SVD, no deep learning, and no additional model anywhere in this design.

---

## 1. Overview

### Purpose of recommendation evaluation

Phase 5 exists to answer this project's central research question with numbers, not impressions: **does Content-Based Filtering or User-Based Collaborative Filtering produce better multi-destination attraction recommendations on this dataset?** Every design decision in this document exists in service of making that comparison as fair and as interpretable as possible.

### Why offline evaluation is necessary

This project has no live, deployed application with real users clicking through recommendations in real time — it has one static historical dataset (~100,000 interactions, 10,000 users, 433 attractions). Online evaluation methods (A/B testing, live click-through measurement) require an audience of active users interacting with a running system, which is outside this project's scope and timeline. Offline evaluation is the standard, practical substitute: a portion of each user's real historical interactions (`test_df`, from Phase 2) is deliberately withheld from training and treated as "ground truth" of what that user was genuinely interested in. Each model is then asked to recommend attractions as if those held-out interactions hadn't happened yet, and evaluation checks whether the model's recommendations would have surfaced them.

### Why both recommenders must use the same train/test split

If CBF and CF were evaluated against different splits of the data, any difference in their metrics could be explained by which data each model happened to see, rather than by a genuine difference in recommendation quality. Reusing Phase 2's `train_test_split_by_user()` output unchanged, for both models, removes this confound entirely — it is the same fairness argument already used to justify a single shared preprocessing contract across Phases 3 and 4, extended now to the comparison itself.

---

## 2. Evaluation Pipeline

```
Train/Test Split (train_df, test_df — from Phase 2)
              │
              ▼
   Generate Recommendations
   (CBF: recommend_attractions()
    CF: recommend_attractions_cf())
              │
              ▼
Filter Already Visited Attractions
  (performed internally by each
   model's own recommend function
   — not reimplemented here)
              │
              ▼
Compare Against Held-Out Test Interactions
              │
              ▼
    Calculate Evaluation Metrics
   (Precision@K, Recall@K, F1@K, Coverage)
              │
              ▼
        Compare CBF vs CF
```

As with previous phases, this separates into two lifetimes:

- **Per-model, computed once per evaluation run:** generating each model's Top-N recommendations for every test user, and aggregating their metrics.
- **Final, computed once:** combining both models' aggregated metrics into a single comparison table.

---

## 3. Detailed Explanation of Every Evaluation Step

### Step 1 — Train/test split
- **Input:** the full preprocessed interaction table.
- **Output:** `train_df`, `test_df` — Phase 2's `train_test_split_by_user()`, reused unchanged.
- **Purpose:** establish the held-out ground truth every metric in this document is computed against.
- **Why necessary:** without withheld data, there is no way to check whether a recommendation is genuinely correct — a model "evaluated" on data it already trained on would trivially appear accurate, providing no real signal about its predictive quality.
- **Function:** none — external Phase 2 contract, consumed directly by `evaluate_model()`.

### Step 2 — Generate recommendations
- **Input:** each test user's `tourist_id`, `train_df`, a shared `top_n`, and each model's own already-finalized hyperparameters (CBF's `rating_threshold`, CF's `k`).
- **Output:** a ranked Top-N recommendation `DataFrame` per user, per model.
- **Purpose:** obtain each model's actual candidate output, generated exactly as it would be used in the real application.
- **Why necessary:** this is the object under evaluation. Critically, `evaluation.py` does not reimplement or duplicate either model's logic — it calls each model's own finalized public function (`content_based.recommend_attractions()` or `collaborative.recommend_attractions_cf()`) directly, so the evaluated behaviour is guaranteed identical to the actual application behaviour. Recommendations are generated independently for each user using only `train_df` — neither model's `recommend_fn` is given access to `test_df` at any point in this step, which is what keeps `test_df` genuinely held out rather than leaking into training or generation.
- **Function:** `evaluate_model()`, via the `recommend_fn` parameter (Section 7).

### Step 3 — Filter already-visited attractions
- **Input:** the recommendation candidates generated in Step 2, the target user's attractions in `train_df`.
- **Output:** a Top-N list containing only attractions the user has not yet rated in training.
- **Purpose:** make explicit, in the pipeline diagram, a filtering step that already happens — this filtering is performed **inside** each model's own `recommend_attractions()` / `recommend_attractions_cf()` (Phase 3 Section 8, Phase 4 Section 8), not duplicated in `evaluation.py`.
- **Why necessary:** without this step, a model could "score" a hit simply by re-recommending something the user already interacted with during training, which would inflate precision/recall without reflecting any real predictive ability.
- **Function:** none within `evaluation.py` — already implemented in each model's own public function; shown here only for pipeline completeness.

### Step 4 — Compare against held-out test interactions
- **Input:** one user's filtered Top-N recommendation list, that same user's actual `attraction_uid`s in `test_df`.
- **Output:** the count of recommended attractions that appear in the user's test interactions (a "hit" count), alongside the total size of the recommendation list and the total size of the user's test set.
- **Purpose:** determine, per user, how many recommended attractions were ones the user actually went on to interact with.
- **Why necessary:** this is the raw comparison every metric in Section 4 is built from.
- **Function:** `precision_at_k()`, `recall_at_k()`.

### Step 5 — Calculate evaluation metrics
- **Input:** hit counts and set sizes, aggregated across all evaluable test users.
- **Output:** Precision@K, Recall@K, F1@K, and Coverage — one value per metric, per model.
- **Purpose:** summarise per-user comparisons into model-level scores that can actually be reported and compared.
- **Why necessary:** individual per-user hit/no-hit results aren't directly interpretable at the level a report or viva panel needs — aggregation is what produces the comparable numbers.
- **Function:** `evaluate_model()`, calling `precision_at_k()`, `recall_at_k()`, `f1_at_k()`, `coverage()` internally.

### Step 6 — Compare CBF vs CF
- **Input:** CBF's aggregated metric set, CF's aggregated metric set.
- **Output:** a single comparison table, one row per model.
- **Purpose:** directly answer this project's core research question.
- **Why necessary:** this is the actual deliverable Phase 5 exists to produce — everything before this step exists to make this final comparison valid.
- **Function:** `compare_models()`.

---

## 4. Evaluation Metrics

### Precision@K

- **Definition:** the proportion of the Top-K recommended attractions that the user actually interacted with (i.e. that appear in their `test_df`).
- **Intuition:** "of what we recommended, how much was actually relevant?"
- **Formula:**
$$
\text{Precision@K} = \frac{\lvert \text{Recommended} \cap \text{Test} \rvert}{\lvert \text{Recommended} \rvert}
$$
  where $\lvert \text{Recommended} \rvert$ is the number of attractions actually returned to that user — not necessarily a fixed $K$, since some users may legitimately receive fewer than $K$ recommendations under either model's known coverage limitations (Phase 3 Section 11, Phase 4 Section 9). Using the actual returned count as the denominator avoids unfairly penalising a model for correctly returning a shorter, valid list rather than padding it. If the recommendation list is empty, Precision@K is defined as 0 (rather than undefined), so the metric is always computable without a special case in the calling code.
- **Interpretation:** higher precision means a larger share of what was shown to the user was something they genuinely went on to engage with.
- **Strengths:** directly measures recommendation quality from the perspective of "was this list worth showing"; simple, standard, and directly comparable between the two models.
- **Weaknesses:** does not account for how many relevant attractions existed for that user in total — a user with only two attractions in `test_df` can never exceed a precision of $2/K$, regardless of how good the model is. This is why Recall@K is needed alongside it, not instead of it.

### Recall@K

- **Definition:** the proportion of the user's actual held-out test interactions that were successfully captured within the Top-K recommendations.
- **Intuition:** "of everything the user actually went on to do, how much did we successfully surface in advance?"
- **Formula:**
$$
\text{Recall@K} = \frac{\lvert \text{Recommended} \cap \text{Test} \rvert}{\lvert \text{Test} \rvert}
$$
  where $\lvert \text{Test} \rvert$ is the total number of attractions in that user's held-out test set. If the recommendation list is empty, Recall@K is defined as 0 for the same reason as Precision@K above.
- **Interpretation:** higher recall means the model captured more of the user's true future interests within its recommendation list.
- **Strengths:** complements Precision@K's blind spot by accounting for the size of the true positive set; directly measures "how much of the ground truth did we find."
- **Weaknesses:** trivially increases as $K$ grows (a model recommending every attraction achieves recall of 1), which is precisely why Top-N must be fixed identically across both models being compared (Section 5) rather than tuned independently per model. It also penalises users with unusually large test sets more heavily for a fixed small $K$, independent of model quality.

### F1-score@K

- **Definition:** the harmonic mean of Precision@K and Recall@K.
- **Intuition:** a single balanced score for situations where neither false positives (irrelevant recommendations) nor false negatives (missed relevant attractions) should be optimised for at the other's expense.
- **Formula:**
$$
\text{F1@K} = \frac{2 \cdot \text{Precision@K} \cdot \text{Recall@K}}{\text{Precision@K} + \text{Recall@K}}
$$
- **Interpretation:** F1@K is only high when both precision and recall are reasonably high; a model that is excellent on one but very weak on the other still scores low overall.
- **Strengths:** gives one comparable number per model, useful for a direct "which model wins" statement in the results chapter and for chart comparisons.
- **Weaknesses:** as a single blended number, it obscures *which* of precision or recall is driving a model's score — it should always be reported alongside its two components, never as a replacement for them.

### Coverage

- **Definition:** the proportion of test users for whom the recommender successfully returned a **non-empty recommendation list** — i.e. users not blocked by that model's own structural limitation (CBF's requirement of at least one rating ≥ its `rating_threshold`; CF's requirement of at least one valid neighbour). Concretely: a recommendation output containing zero rows (an empty `DataFrame`) is considered "not covered" for that user. This is exactly the observable output both models already produce for a cold-start user — CBF's `recommend_attractions()` and CF's `recommend_attractions_cf()` both return an empty `DataFrame` rather than an error or a fallback score when a user cannot be served (Phase 3 Section 11, Phase 4 Section 9) — so Coverage can be computed directly from `len(recommendations) > 0` without any model-specific logic. This is phrased as "non-empty," rather than "≥ 1 recommendation," so the definition remains precise even in a degenerate case such as `top_n = 0` (which should not occur in practice, but the metric's definition should not depend on that assumption holding).
- **Intuition:** "for how many users can this model even be used?" — a model can post excellent precision/recall on the users it manages to serve, while quietly failing to serve a large share of users at all. Precision/Recall/F1 alone cannot reveal this; Coverage exists specifically to expose it.
- **Formula:**
$$
\text{Coverage} = \frac{\text{number of test users with a non-empty recommendation list}}{\text{total number of test users}}
$$
- **Interpretation:** lower coverage means the model's practical usefulness is more limited across the user base, independent of how accurate it is for the users it does reach.
- **Strengths:** turns each model's already-documented cold-start limitation (Phase 3 Section 11, Phase 4 Section 9) into a concrete, comparable number rather than a qualitative statement; particularly informative here since CBF's and CF's coverage limitations arise from entirely different mechanisms (a rating threshold vs. neighbour availability), so their coverage numbers are not expected to behave identically.
- **Weaknesses:** says nothing about the quality of the recommendations that were produced — must always be reported alongside Precision@K/Recall@K/F1@K, not as a substitute for them.

### Additional metrics considered and excluded

Rank-sensitive metrics such as NDCG@K were considered, since they reward placing correct recommendations higher within the Top-N list rather than treating every position equally. This was not included: NDCG is designed for graded relevance judgments (e.g. a 1–5 relevance score per item), and this project's ground truth is binary (an attraction either appears in `test_df` or it does not) — applying NDCG here would require introducing an additional, unjustified assumption about relative relevance that the dataset does not actually support. Precision@K, Recall@K, F1@K, and Coverage are sufficient to answer this project's comparison question without that added complexity.

---

## 5. Comparison Strategy

Fairness between CBF and CF rests on three fixed elements, all reused unchanged from earlier phases:

### Same train/test split

Both models are evaluated against the same `train_df`/`test_df` produced by Phase 2's `train_test_split_by_user()`. Neither model is retrained or re-split independently for this comparison.

### Same Top-N value

Both models are evaluated with an identical `top_n` (default **10**, consistent with the default already established in Phases 3 and 4). Recall@K in particular is highly sensitive to $N$ — comparing CBF at $N=10$ against CF at $N=20$ would invalidate the comparison outright, since the model given the larger $N$ would have a mechanical advantage unrelated to recommendation quality. Each model's *other* hyperparameters (CBF's `rating_threshold`, CF's `k`) remain at their own previously-established defaults, since those are internal properties of each model's design, not evaluation parameters — only `top_n` is held identical across both models specifically because it is what the comparison depends on being equal.

### Same users — each model evaluated independently

CBF and CF have different, independent coverage limitations (a rating threshold vs. neighbour availability), so a given test user may be servable by one model but not the other. Rather than restricting the accuracy metrics to the intersection of users both models can serve, each model is evaluated **independently over its own evaluable users**:

- **Precision@K, Recall@K, and F1@K** for a given model are averaged only over the users *that model* successfully produced a recommendation for. CBF's precision is computed over CBF's evaluable users; CF's precision is computed over CF's evaluable users — these user sets are not required to match.
- **Coverage** is reported separately per model, over the full test user set, to quantify the proportion of users each model was able to serve at all.

This keeps `evaluate_model()` fully self-contained: it evaluates one model against the full test user set and reports that model's own results, without needing to know anything about the other model's outcome first. An intersection-based approach was considered and rejected — it would require running both models to completion before either model's accuracy metrics could be computed (since the intersection depends on both), adding an ordering dependency between the two evaluation runs that this design deliberately avoids. Reporting each model's accuracy metrics over its own evaluable users, with coverage reported alongside as a separate, comparable number, is consistent with how this comparison is typically reported in recommender systems evaluation literature, and keeps `evaluate_model()`'s responsibility limited to one model at a time.

**If a stricter, intersection-only comparison is ever wanted later** (e.g. as a supplementary robustness check, not the primary reported result), that responsibility belongs to `compare_models()`, not `evaluate_model()` — only `compare_models()` has access to both models' results at once. It would need to be extended to accept both models' `per_user_results` tables (not just their `summary` rows), compute the intersection of `tourist_id`s present in both, and recompute Precision@K/Recall@K/F1@K restricted to that intersection. This is explicitly **not** part of the current design's default comparison strategy (which reports each model over its own evaluable users, per above) — it is noted here only so that, if requested in the future, it is clear which function is responsible and what additional input it would require, rather than leaving that as an open question.

---

## 6. Output Specification

`evaluation.py` (together with `05_evaluation.ipynb` and `06_itinerary_analysis.ipynb`) produces:

| Output | Description |
|---|---|
| Per-user metrics table *(intermediate)* | `tourist_id`, `model_name`, `precision_at_k`, `recall_at_k`, `hit_count`, `recommended_count`, `test_count` — one row per evaluable user per model. Useful for inspection/debugging, not the primary report deliverable. |
| Aggregated metrics summary | One row per model: `model_name`, `precision_at_k`, `recall_at_k`, `f1_at_k`, `coverage`, `evaluated_user_count`, `coverage_user_count`, `total_test_user_count`. The direct output of `evaluate_model()`. `coverage_user_count` and `total_test_user_count` are kept alongside the `coverage` ratio so a reader can see, e.g., "6,700 of 10,000" rather than only the ratio "0.67" — the same reasoning applies to `evaluated_user_count`, which records how many users the accuracy metrics were actually averaged over. |
| Comparison table | Both models' summary rows combined into a single side-by-side table — the direct output of `compare_models()`, ready to be copied into the results chapter. |
| Charts *(optional, notebook only)* | e.g. a bar chart comparing Precision@K / Recall@K / F1@K / Coverage between CBF and CF. Visualization is performed only in `05_evaluation.ipynb` — `evaluation.py` never imports `matplotlib` or any plotting library, consistent with the established pattern that the `src/` modules compute numbers and notebooks handle presentation. |

---

## 7. Function Design — `src/evaluation.py`

No implementation code, per your instruction.

### `precision_at_k(recommended_attractions, actual_attractions)`
- **Parameters:** `recommended_attractions` — the list/set of `attraction_uid`s recommended to one user; `actual_attractions` — the set of that user's `attraction_uid`s in `test_df`
- **Returns:** a single float — that user's Precision@K
- **Responsibility:** implement the Precision@K formula (Section 4), using the actual returned recommendation count as the denominator.
- **Dependencies:** none.

### `recall_at_k(recommended_attractions, actual_attractions)`
- **Parameters:** same shape as `precision_at_k()`
- **Returns:** a single float — that user's Recall@K
- **Responsibility:** implement the Recall@K formula.
- **Dependencies:** none.

### `f1_at_k(precision, recall)`
- **Parameters:** an already-computed precision value, an already-computed recall value
- **Returns:** a single float — the harmonic mean
- **Responsibility:** implement the F1@K formula; return 0 (rather than raising a division error) when precision and recall are both 0.
- **Dependencies:** none — operates on already-computed values, does not recompute precision/recall itself.

### `coverage(evaluated_user_count, total_test_user_count)`
- **Parameters:** the number of users for whom the recommender successfully returned at least one recommendation; the total number of test users considered
- **Returns:** a single float — the coverage ratio. `evaluate_model()` retains and reports the two raw counts (`evaluated_user_count`, `total_test_user_count`) alongside this ratio in its summary output, so the ratio is never reported without its denominator visible.
- **Responsibility:** implement the Coverage formula.
- **Dependencies:** none.

### `evaluate_model(recommend_fn, test_users, train_df, test_df, model_context, top_n, model_kwargs)`
- **Parameters:**
  - `recommend_fn` — the model's own finalized public function, passed in as a callable (`content_based.recommend_attractions` or `collaborative.recommend_attractions_cf`)
  - `test_users` — the list of `tourist_id`s to evaluate
  - `train_df` — for visited-attraction filtering (shared by both models)
  - `test_df` — passed to `extract_ground_truth()` per user
  - `top_n` — the shared Top-N value (Section 5)
  - `model_context` — a dict holding the **precomputed data objects** that specific model's `recommend_fn` needs, which do not change between calls (e.g. for CBF: `{"attraction_df": ..., "tfidf_matrix": ..., "vectorizer": ...}`; for CF: `{"attraction_df": ..., "user_item_matrix": ..., "user_similarity_matrix": ..., "user_index": ..., "attraction_index": ...}`)
  - `model_kwargs` — a dict holding that model's own **tunable hyperparameters** (e.g. for CBF: `{"rating_threshold": 4.0}`; for CF: `{"k": 20}`)
- **Returns:** a tuple `(summary, per_user_results)` — `summary` is the single aggregated metrics row described in Section 6; `per_user_results` is the per-user metrics table (`tourist_id`, `precision_at_k`, `recall_at_k`, `hit_count`, `recommended_count`, `test_count`), returned alongside `summary` specifically so `05_evaluation.ipynb` can chart or inspect per-user results without a separate function call, and so that a stricter secondary comparison (Section 5) can be built from it later if ever needed.
- **Responsibility:** loop over `test_users`, call `recommend_fn(tourist_id=tourist_id, train_df=train_df, top_n=top_n, **model_context, **model_kwargs)` for each, retrieve that user's ground truth via `extract_ground_truth(tourist_id, test_df)`, and aggregate the results via `precision_at_k()`, `recall_at_k()`, `f1_at_k()`, and `coverage()`.
- **Why `model_context` and `model_kwargs` are kept as two separate dicts rather than one:** `model_context` holds objects that are built once, upfront, and reused unchanged across the whole evaluation run (matrices, indices, the attraction table); `model_kwargs` holds the small number of named hyperparameters that define *which version* of that model is being evaluated (e.g. which `k`, which `rating_threshold`). Keeping them distinct makes it possible to re-run evaluation with a different hyperparameter value (Phase 4/5 tuning) without touching or rebuilding `model_context` at all — the separation reflects a real difference in how often each part of the input changes, not just a stylistic split.
- **Note on `attraction_df`:** `attraction_df` is included inside `model_context` solely because `recommend_fn` needs it internally (for the `attraction_name`/`city`/`attraction_category` joins already specified in Phase 3 Section 9 and Phase 4 Section 10). `evaluate_model()` itself never reads `attraction_df`'s contents — its own logic only needs the `attraction_uid` column of whatever `recommend_fn` returns, compared against `extract_ground_truth()`'s output. This keeps `evaluate_model()` independent of attraction attributes entirely, consistent with it being a model-agnostic evaluation loop rather than a data-processing step.
- **Dependencies:** `precision_at_k()`, `recall_at_k()`, `f1_at_k()`, `coverage()`, `extract_ground_truth()`, and whichever model's public function is passed in (external, from Phase 3 or Phase 4 — never reimplemented here).

### `extract_ground_truth(tourist_id, test_df)` *(helper)*
- **Parameters:** target `tourist_id`; `test_df`
- **Returns:** the set of `attraction_uid`s that user actually interacted with in the held-out test data
- **Responsibility:** isolate the repeated `test_df[test_df["tourist_id"] == tourist_id]` lookup pattern into one small, reusable helper, since `evaluate_model()` performs this lookup once per user in its loop.
- **Dependencies:** none.

### `compare_models(cbf_summary, cf_summary)`
- **Parameters:**
  - `cbf_summary` (`pd.DataFrame`) — the aggregated summary returned by `evaluate_model()` for CBF
  - `cf_summary` (`pd.DataFrame`) — the aggregated summary returned by `evaluate_model()` for CF
- **Returns:** a comparison `DataFrame` with one row per model and one column per evaluation metric (`model_name`, `precision_at_k`, `recall_at_k`, `f1_at_k`, `coverage`, `evaluated_user_count`, `coverage_user_count`, `total_test_user_count`) — i.e. CBF and CF as two rows, directly comparable column by column.
- **Responsibility:** concatenate both models' summaries into the final side-by-side comparison table (Section 6), preserving the raw counts alongside each ratio so neither `coverage` nor the accuracy metrics are reported as a bare percentage without its denominator visible. No additional metrics are computed here — this function only concatenates the aggregated summaries already returned by `evaluate_model()`; it performs no additional statistical testing (e.g. paired t-test, Wilcoxon signed-rank test), which is out of scope for this project.
- **Dependencies:** `evaluate_model()` outputs for both models (computed beforehand, typically in `05_evaluation.ipynb`).

### `run_full_evaluation(...)` *(optional convenience function)*
- **Parameters:** all inputs needed to run both models' evaluations and the comparison in one call (train/test data, both models' recommend functions and `model_context` dicts, shared `top_n`)
- **Returns:** the final comparison table
- **Responsibility:** orchestrates `evaluate_model()` for CBF, `evaluate_model()` for CF, and `compare_models()`, mirroring the single-entry-point pattern already established by Phase 2's `preprocess_pipeline()`. This is optional — not explicitly required by the function list above, but recommended for consistency with the rest of the project's architecture, and to give `05_evaluation.ipynb` one clean call rather than three separate ones. This function is provided purely as a convenience wrapper and does not introduce any additional evaluation logic beyond calling the three functions above in sequence.
- **Dependencies:** `evaluate_model()`, `compare_models()`.

**Public interface note:** `evaluation.py` should expose `evaluate_model()` and `compare_models()` (or `run_full_evaluation()`, if implemented) as its public interface. `precision_at_k()`, `recall_at_k()`, `f1_at_k()`, `coverage()`, and `extract_ground_truth()` are internal helpers, not intended to be called directly by `05_evaluation.ipynb` or `app.py`.

### Complete function list for `src/evaluation.py`

**Required:**
- `precision_at_k()`
- `recall_at_k()`
- `f1_at_k()`
- `coverage()`
- `extract_ground_truth()`
- `evaluate_model()`
- `compare_models()`

**Optional:**
- `run_full_evaluation()`

No further functions are part of this design. This list is exhaustive so that implementation does not introduce additional functions beyond what this architecture specifies.

---

## 8. Documentation Updates

**`PROJECT_PROGRESS.md`** — suggested entry:
```
## Phase 5 — Recommendation System Evaluation (Design)
Status: Design finalized, ready for implementation
Notes: Evaluation module designed to compare CBF and User-Based CF
using Precision@K, Recall@K, F1@K, and Coverage, over the same
train/test split and same Top-N value established in earlier phases.
No hybrid recommender, no SVD, no additional models. Full architecture
in Phase5_Evaluation_Design.md. 8 functions specified in
src/evaluation.py (7 required + 1 optional orchestrator).
Implementation not yet started.
```

**`AI_DECISIONS.md`** — new entries:
```
- Decision: evaluation metrics are Precision@K, Recall@K, F1@K, and
  Coverage. NDCG@K was considered and excluded.
  Reason: NDCG requires graded relevance judgments; this project's
  ground truth (test_df membership) is binary, so NDCG would require
  an unjustified additional assumption the data does not support.

- Decision: Precision@K/Recall@K/F1@K are averaged over each model's own
  evaluable users independently (not an intersection of users both
  models can serve); Coverage is reported separately, per model, over
  the full test user set.
  Reason: keeps evaluate_model() self-contained — it can evaluate one
  model to completion without needing the other model's results first.
  An intersection-based approach was considered and rejected, since it
  would introduce an ordering dependency between the two models' eval
  runs and complicate evaluate_model()'s responsibility. This matches
  common practice in recommender systems evaluation literature.

- Decision: evaluate_model() receives each model's supporting objects
  and hyperparameters through two separate dicts — model_context
  (precomputed matrices/indices/attraction_df, built once and reused
  unchanged) and model_kwargs (tunable hyperparameters, e.g.
  rating_threshold or k) — rather than listing CBF- or CF-specific
  parameters directly in its own signature, and rather than merging
  both into a single dict.
  Reason: CBF and CF require entirely different supporting objects
  (TF-IDF matrix vs. user-item/similarity matrices); model_context lets
  evaluate_model() stay identical regardless of which model is passed
  in. Separating model_kwargs from model_context additionally allows
  re-running evaluation with a different hyperparameter value without
  rebuilding model_context at all.

- Decision: aggregated metric summaries report raw counts
  (evaluated_user_count, coverage_user_count, total_test_user_count)
  alongside every ratio metric (coverage, precision, recall, f1).
  Reason: a bare ratio like "coverage = 0.67" is ambiguous without its
  denominator (67/100 vs. 6,700/10,000); reporting counts alongside
  ratios removes that ambiguity in the results chapter.

- Decision: Precision@K and Recall@K are explicitly defined as 0 when
  a user's recommendation list is empty, rather than left undefined.
  Reason: makes both metrics always computable without a special case
  in the calling code, and avoids an undefined-division edge case.

- Decision: evaluation.py calls each model's own finalized public
  function (recommend_attractions() / recommend_attractions_cf())
  directly, rather than reimplementing scoring logic.
  Reason: guarantees the evaluated behaviour is identical to the
  actual application behaviour, and keeps evaluation.py fully
  decoupled from either model's internals.

- Decision: Top-N is fixed identically across both models during
  evaluation (default 10); each model's own hyperparameters
  (rating_threshold, k) remain at their previously-established
  defaults and are not altered for evaluation purposes.
  Reason: Recall@K in particular is highly sensitive to N; an unequal
  N between models would invalidate the comparison.
```

**`PROJECT_SPEC.md`** — if this document lists the project's evaluation methodology, it should be updated to name Precision@K, Recall@K, F1@K, and Coverage explicitly as the finalized metric set, and to state that no hybrid, SVD, or deep-learning model is evaluated in this project. I don't have the actual file to edit directly — please add this once reviewed.

---

## 9. Known Limitations

- **Offline evaluation limitations.** These metrics measure whether a model could correctly "guess" interactions the user already went on to have — they give no credit for recommending something genuinely new and appealing that the user never happened to encounter or interact with historically. This is a structural blind spot of any offline evaluation, not specific to either model here.
- **Dependence on historical interactions.** Evaluation quality is bounded by how complete and representative the historical interaction log is. Gaps in the data — attractions a user never had the opportunity to encounter, seasonal effects, regional availability — become gaps in what can be measured, independent of whether a model is actually good or bad.
- **Train/test split bias.** The per-user split (Phase 2) is one particular partition of the data; a different random seed could shift individual users' precision/recall somewhat. Results should be understood as representative of this project's specific split, not as an absolute, seed-independent measurement — repeating evaluation across multiple random splits (e.g. cross-validation) would strengthen this but is out of scope given the project's timeline.
- **Top-N dependency.** Precision@K, Recall@K, and F1@K all shift with the choice of $N$; a different $N$ could plausibly change which model appears to perform better. Results should always be reported alongside the specific $N$ used, not generalised beyond it.
- **Inability to evaluate true user satisfaction.** Historical interaction is used throughout this design as a proxy for "the user liked this recommendation" — but a user's actual subjective satisfaction, budget, timing, travel companions, or reasons for visiting (or not visiting) an attraction are not captured anywhere in this dataset. Offline metrics are a structured, useful proxy for comparing two models against the same imperfect ground truth — they are not a direct measurement of whether real users would be happy with either system.
- **Unobserved does not necessarily mean irrelevant.** Only explicit historical interactions are treated as relevant; every other attraction is implicitly assumed non-relevant for evaluation purposes, although some of those attractions may simply never have been discovered or considered by the user, rather than genuinely rejected. This is a standard, acknowledged assumption in offline recommender systems evaluation, not a flaw specific to this project's design — but it means a "miss" in these metrics does not always mean the recommendation was a poor match.
- **This evaluation assumes historical interactions indicate user preference.** A recorded `rating` is treated throughout this design as a proxy for genuine preference; ratings themselves may be influenced by factors this dataset does not capture (mood at the time of rating, group decisions, price, convenience), so "high rating" and "true preference" are related but not identical. This is a standard, acknowledged assumption in recommender systems evaluation generally, not a limitation specific to CBF or CF individually.

-------------------------------------------------

# Phase 6 — Streamlit Prototype Architecture
## Travel Destination Recommendation System

**Status:** Phases 1–5 (EDA, Preprocessing, Content-Based Filtering, User-Based Collaborative Filtering, Offline Evaluation) are finalized and treated as fixed contracts. This document designs `app.py` — the Streamlit presentation layer. It is **not** a new recommendation algorithm. No CBF/CF logic, similarity computation, or metric calculation is implemented in `app.py` — every number shown on screen is produced by calling an already-finalized public function from `src/`.

**Explicitly excluded from this design:** no hybrid recommender, no SVD/matrix factorization, no deep learning, no neural network, no additional recommendation algorithm of any kind. `app.py` presents the two existing models (CBF, CF) side by side — it does not create a third.

---

## 1. Overview

`app.py` is a Streamlit application that lets a user interactively generate and view attraction recommendations from either finalized model. It is the demonstration and interaction layer sitting on top of Phases 2–4 — every recommendation shown was already computable from a notebook; this phase makes that capability reachable through a browser interface rather than a code cell.

## 2. Purpose of the Prototype

- Give a live, interactive way to demonstrate both recommendation models for the report and viva, rather than relying solely on static notebook output.
- Verify, end-to-end, that Phase 3's and Phase 4's public functions (`recommend_attractions()`, `recommend_attractions_cf()`) work correctly outside a notebook context — a genuine integration check on the finalized modules, not a redesign of them.
- Let a reviewer directly compare CBF and CF for the same user, side by side, which is harder to convey from separate notebook cells.

This is a **prototype**, not a dashboard: its scope is limited to generating and displaying recommendations. It does not attempt to also serve as an evaluation reporting tool — Phase 5's evaluation results remain a notebook/report deliverable, not an in-app feature (see Section 8).

---

## 3. Module Dependency Diagram

```
                        app.py
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
 preprocessing.py   content_based.py  collaborative.py
```

`evaluation.py` is deliberately **not** a dependency of `app.py` — it remains an offline, notebook-only module (Phase 5), with no runtime role in the prototype.

Within `app.py` itself, the two models are accessed independently and never cross-call each other:

```
app.py
        │
        ├──────────────┐
        │              │
        ▼              ▼
Content-Based     Collaborative
   Filtering       Filtering
        │              │
        └──────┬───────┘
               ▼
        Streamlit UI
```

---

## 4. Overall Application Workflow

1. The app starts and loads the cleaned dataset and precomputed model artefacts once (cached, not recomputed on every interaction).
2. The user selects a `tourist_id`, an algorithm (CBF or CF), and a Top-N value in the sidebar.
3. The user clicks "Generate Recommendations."
4. The app calls the corresponding model's existing public function with the user's selections.
5. The app renders the returned ranked table, or a cold-start message if the model could not produce recommendations for that user (Phase 3 Section 11, Phase 4 Section 9).

---

## 5. Application Pipeline

```
User
      │
      ▼
Load Dataset
      │
      ▼
Preprocessing
      │
      ▼
Select Recommendation Algorithm
      │
      ▼
Generate Recommendation
      │
      ▼
Display Recommendation Results
```

As with every previous phase, this separates into two lifetimes:

- **Once per app session, computed and cached:** dataset loading, preprocessing, and both models' precomputed matrices (TF-IDF matrix for CBF; user-item and similarity matrices for CF).
- **Per user interaction, computed on request:** generating and displaying one recommendation list for the currently selected `tourist_id`.

---

## 6. Application Layout

### Sidebar
- **Tourist ID** (Selectbox) — populated from the set of `tourist_id`s present in the cleaned dataset.
- **Recommendation Model** (Radio) — "Content-Based Filtering" / "User-Based Collaborative Filtering."
- **Top N** (Slider) — default 10, consistent with the Top-N default already established in Phases 3–5.
- **Generate Recommendation** (Button) — triggers recommendation generation; nothing is (re)computed on sidebar changes alone.
- *(Optional)* **Reset Selection** (Button) — clears the current selection and displayed results back to the app's initial state. Not required, but a small usability addition some reviewers like to see.

### Main Page
- App title and short description of what the prototype demonstrates.
- Selected user's context (`tourist_id`, chosen algorithm, chosen Top-N) echoed back before results, so the displayed table's provenance is unambiguous.

### Recommendation Table
- The ranked `DataFrame` returned by the selected model's public function, rendered directly — no reformatting logic beyond what Streamlit's table display already provides.

### Status Messages
- A loading indicator while a recommendation is being generated (relevant mainly the first time a model's matrices are being built, since subsequent calls reuse the cache).
- A success state once results are ready.
- An error state only for genuine failures (e.g. an invalid `tourist_id`), distinct from the cold-start case below.

### Cold-Start Behaviour
- If the selected model returns an empty result for that user (CBF: no rating ≥ `rating_threshold`; CF: no valid neighbours — Phase 3 Section 11, Phase 4 Section 9), the app displays a plain-language message explaining *why* no recommendations are available for that specific user and model, rather than an empty table or a generic error. **This is not treated as an application error — it is an expected model limitation**, already documented and accepted in Phase 3 and Phase 4, and the app's role is only to communicate it clearly, not to work around it.

---

## 7. Detailed Explanation of Every Application Step

### Step 1 — Load dataset
- **Input:** the raw CSV path.
- **Output:** the cleaned attraction and interaction datasets, plus `train_df`/`test_df`.
- **Purpose:** provide the same finalized data foundation every other phase already uses.
- **Why necessary:** the app must reuse Phase 2's existing functions exactly as finalized — it must not re-clean, re-split, or reinterpret the data independently.
- **Which function:** `load_data()`, calling `preprocessing.load_dataset()`, `preprocessing.prepare_attractions()`, `preprocessing.prepare_interactions()`, and `preprocessing.train_test_split_by_user()` (cached).

### Step 2 — Preprocessing
- **Input:** the loaded dataset objects from Step 1.
- **Output:** the two models' precomputed artefacts — CBF's TF-IDF matrix and attraction index; CF's user-item and user-similarity matrices and both index mappings.
- **Purpose:** build each model's expensive, reusable objects exactly once per app session, rather than once per button click.
- **Why necessary:** without caching, every recommendation request would rebuild a TF-IDF matrix or a 10,000 × 10,000 similarity matrix from scratch — this is the same "fit once, reuse many" principle already established in Phase 3 (`build_tfidf_matrix()`) and Phase 4 (`build_user_similarity_matrix()`), just enforced here at the application layer via Streamlit's caching.
- **Which function:** `get_cbf_context()`, `get_cf_context()` (both cached; Section 8).

### Step 3 — Select recommendation algorithm
- **Input:** the user's selection (CBF or CF) via the sidebar.
- **Output:** a flag/string indicating which model's public function to call.
- **Purpose:** determine which of the two already-finalized recommendation pipelines to invoke.
- **Why necessary:** `app.py` must dispatch to the correct model without containing any of that model's own logic — this step is purely a routing decision.
- **Which function:** handled inline in `app.py`'s main script logic, feeding into `get_recommendations()`.

### Step 4 — Generate recommendation
- **Input:** `tourist_id`, selected algorithm, `top_n`, and that model's cached context from Step 2.
- **Output:** a ranked recommendation `DataFrame`, exactly as defined in Phase 3 Section 9 or Phase 4 Section 10.
- **Purpose:** obtain the actual recommendation to display.
- **Why necessary:** this is the one step in the whole pipeline that calls into `src/`'s recommendation logic — everything before it exists to prepare for this call, and everything after it exists to display its result.
- **Which function:** `get_recommendations()`, which calls `content_based.recommend_attractions()` or `collaborative.recommend_attractions_cf()` directly — never a reimplementation.

### Step 5 — Display recommendation results
- **Input:** the `DataFrame` from Step 4 (possibly empty).
- **Output:** rendered content on the Streamlit page — either the recommendation table or the cold-start message.
- **Purpose:** present the result to the user in a readable form.
- **Why necessary:** turns a `DataFrame` into something a non-technical reviewer (e.g. a viva panel member) can read and understand at a glance.
- **Which function:** `render_results()`.

---

## 8. Function Design

`app.py`'s responsibility is strictly **orchestration and presentation** — session setup, caching, dispatching to the correct model, and rendering output. It contains no similarity computation, no TF-IDF logic, no neighbour-finding logic, and no metric calculation. Every one of those already exists in `src/` and is called, not reimplemented.

### `load_data()`
- **Responsibility:** load the raw CSV using `preprocessing.load_dataset()`, prepare the attraction and interaction datasets using `preprocessing.prepare_attractions()` and `preprocessing.prepare_interactions()`, then perform `preprocessing.train_test_split_by_user()`. All four calls are cached together (`st.cache_data`) so the raw CSV is read and processed only once regardless of how many times the user interacts with the sidebar.
- **Returns:** `attraction_df`, `interactions_df`, `train_df`, `test_df`.
- **Dependencies:** `preprocessing.load_dataset()`, `preprocessing.prepare_attractions()`, `preprocessing.prepare_interactions()`, `preprocessing.train_test_split_by_user()` (Phase 2, unchanged).

### `get_cbf_context()`
- **Responsibility:** call `content_based.build_content_column()` and `content_based.build_tfidf_matrix()` once, cached (`st.cache_resource`, since these hold non-serializable fitted objects), returning the `model_context` dict CBF's `recommend_attractions()` actually consumes.
- **Returns:** `{attraction_df, tfidf_matrix, attraction_index}`. The fitted vectorizer itself is **not** included here, since `recommend_attractions()` builds a user profile and computes similarity directly from the already-computed `tfidf_matrix` rows — it does not need to vectorize new text at request time, so the vectorizer object has no runtime use in this function and is not part of its context.
- **Dependencies:** `content_based.build_content_column()`, `content_based.build_tfidf_matrix()` (Phase 3, unchanged).

### `get_cf_context()`
- **Responsibility:** call `collaborative.build_user_item_matrix()` and `collaborative.build_user_similarity_matrix()` once, cached (`st.cache_resource`), returning the `model_context` dict CF's `recommend_attractions_cf()` expects.
- **Returns:** `{attraction_df, user_item_matrix, user_similarity_matrix, user_index, attraction_index}`.
- **Dependencies:** `collaborative.build_user_item_matrix()`, `collaborative.build_user_similarity_matrix()` (Phase 4, unchanged).

### `get_recommendations(tourist_id, algorithm, top_n)`
- **Responsibility:** route to `content_based.recommend_attractions()` or `collaborative.recommend_attractions_cf()`, passing the appropriate cached `model_context` (from `get_cbf_context()`/`get_cf_context()`) and that model's own default hyperparameters (`rating_threshold` for CBF, `k` for CF). Returns whatever `DataFrame` the chosen model's public function returns, unmodified — including an empty `DataFrame` in the cold-start case.
- **Dependencies:** `content_based.recommend_attractions()`, `collaborative.recommend_attractions_cf()` (external; never reimplemented here).

### `render_results(recommendations, tourist_id, algorithm)`
- **Responsibility:** pure display logic — render the recommendation table if `recommendations` is non-empty, or the cold-start message (naming the specific model and its own known limitation) if it is empty. The function only displays the `DataFrame` returned by the recommendation module. **No sorting, ranking, filtering, or other post-processing is performed in `app.py`** — the `DataFrame` it receives is already final, exactly as returned by Phase 3 or Phase 4.
- **Dependencies:** none beyond Streamlit's own display primitives.

**On `evaluation.py`:** this design deliberately does not use `evaluation.py` at all, live or otherwise. The prototype's scope is generating and displaying recommendations, not reporting evaluation metrics — Phase 5's results remain a notebook/report deliverable (`05_evaluation.ipynb`), consistent with evaluation being established as an offline-only step with no runtime role.

---

## 9. User Interface Components

| Component | Description |
|---|---|
| Dataset loading | Handled automatically on app start via `load_data()`; no user action required, but a status message can indicate it happened (mainly informative on first load). |
| Tourist ID selection | Selectbox, populated from the set of `tourist_id`s present in the cleaned dataset. |
| Algorithm selection | Radio button: "Content-Based Filtering" / "User-Based Collaborative Filtering." |
| Top-N selection | Slider, default 10 (consistent with the Top-N default already established in Phases 3–5). |
| Generate Recommendation button | Triggers `get_recommendations()` and `render_results()` for the current sidebar selections; recommendations are not regenerated automatically on every widget change, only on explicit request, so the user controls when computation happens. |
| Recommendation table | Displays the returned `DataFrame` — `attraction_uid`, `attraction_name`, `rank`, and optional `city`/`attraction_category`. **Columns displayed depend on the selected algorithm:** CBF's table includes `similarity_score`; CF's table includes `predicted_rating`. The app does not force both score columns into one table regardless of which model was used. |
| Cold-start message | Plain-language text shown in place of an empty table, naming the specific reason (e.g. "This user has no attraction rated ≥ 4.0, so a Content-Based profile cannot be built" or "This user has no similar neighbours in the training data, so a Collaborative prediction cannot be made"). |

---

## 10. Output Specification

For a given `(tourist_id, algorithm, top_n)` selection, the app displays:

| Output | Source |
|---|---|
| Recommendation table | Directly from `content_based.recommend_attractions()` or `collaborative.recommend_attractions_cf()`, per their respective Output Specifications (Phase 3 Section 9, Phase 4 Section 10) — unmodified, with the score column shown depending on which model produced the result (Section 9). |
| Cold-start message | Rendered by `render_results()` when the above is empty; text only, no computation. |
| Selected context summary | The `tourist_id`, algorithm, and `top_n` currently in effect, echoed above the results for clarity. |

---

## 11. Documentation Updates

**`PROJECT_PROGRESS.md`** — suggested entry:
```
## Phase 6 — Streamlit Prototype (Design)
Status: Design finalized, ready for implementation
Notes: app.py designed as a pure presentation/orchestration layer over
the finalized CBF (Phase 3) and CF (Phase 4) modules, using Phase 2's
existing load_dataset()/prepare_attractions()/prepare_interactions()/
train_test_split_by_user() functions unchanged. No recommendation
logic implemented in app.py; both models accessed only through their
existing public functions (recommend_attractions(),
recommend_attractions_cf()). evaluation.py is not used by app.py at
all — evaluation remains a notebook-only, offline deliverable. Full
architecture in Phase6_StreamlitPrototype_Design.md. Implementation
not yet started.
```

**`AI_DECISIONS.md`** — new entries:
```
- Decision: app.py contains no recommendation, similarity, or metric
  computation logic. It calls only the existing public functions
  content_based.recommend_attractions() and
  collaborative.recommend_attractions_cf().
  Reason: keeps the presentation layer strictly decoupled from model
  logic, consistent with the public-interface boundaries already
  established in Phase 3 and Phase 4.

- Decision: each model's precomputed context (TF-IDF matrix and
  attraction index for CBF; user-item matrix, similarity matrix, and
  both index mappings for CF) is built once per app session via
  caching (st.cache_resource), not rebuilt on every recommendation
  request.
  Reason: mirrors the "fit once, reuse many" principle already
  established for both models' own matrix-building steps; without
  this, every button click would rebuild a full similarity matrix.

- Decision: app.py does not use evaluation.py in any capacity — not
  live, and not by reading a precomputed evaluation file either. The
  prototype's scope is limited to generating and displaying
  recommendations.
  Reason: keeps Phase 6 a prototype rather than a dashboard; Phase 5's
  results remain a notebook/report deliverable, avoiding scope creep
  into evaluation-file management (e.g. metrics.csv, save_results()/
  load_results()) that was never part of this project's design.

- Decision: recommendations are only generated on explicit button
  press, not automatically on every sidebar widget change.
  Reason: gives the user clear control over when computation happens,
  and avoids unnecessary repeated calls to the recommendation
  functions while the user is still adjusting their selection.

- Decision: get_cbf_context() returns attraction_df, tfidf_matrix, and
  attraction_index only — the fitted vectorizer is not included.
  Reason: recommend_attractions() builds user profiles and computes
  similarity from the already-computed tfidf_matrix rows; it never
  vectorizes new text at request time, so the vectorizer has no
  runtime use in this function.
```

**`PROJECT_SPEC.md`** — if this document lists the project's deliverables/components, it should be updated to name the Streamlit prototype (`app.py`) as the Phase 6 presentation layer, explicitly noting it introduces no new recommendation algorithm and has no evaluation-reporting responsibility. I don't have the actual file to edit directly — please add this once reviewed.

---

## 12. Known Limitations

- **Offline prototype.** The app runs against a static, locally loaded dataset snapshot — it does not connect to any live data source, and recommendations reflect the dataset as it existed at the time it was loaded, not real-time user behaviour.
- **Single-user interaction.** The prototype is designed for one person exploring the system at a time (e.g. during a demonstration or viva) — it has no concept of multiple simultaneous end-users interacting independently in the way a deployed consumer product would.
- **Dataset loaded locally.** The CSV is read from local disk; there is no remote data pipeline, database connection, or ingestion process.
- **No authentication.** Any `tourist_id` in the dataset can be selected freely — there is no login, identity verification, or access control, which is appropriate for a research prototype but would need to be added before any real deployment.
- **No persistent storage.** Nothing the user does in the app (selections, generated recommendations) is saved between sessions — closing or refreshing the app discards all state.
- **No online/real-time recommendation.** All recommendations are generated from the fixed, offline Phase 2 dataset snapshot; the app cannot incorporate a genuinely new interaction (e.g. a rating submitted through the app itself) into future recommendations, since doing so would require retraining or updating the underlying matrices — out of scope here.

---

## 13. Future Improvements

The following are realistic extensions but are explicitly **out of scope** for this project and are not designed here:

- **User authentication and persistent profiles**, allowing a returning user to be recognized across sessions.
- **A backing database** in place of a static local CSV, enabling the dataset to be updated without redeploying the app.
- **Logging of in-app interactions** (e.g. which recommendations a user viewed) to eventually support incremental retraining — not the same as introducing a new algorithm, simply a data-collection improvement.
- **Deployment to a hosted environment** (rather than local execution), with the associated infrastructure (containerization, environment configuration) that implies.
- **Richer visual analytics within the app** (e.g. simple charts alongside the recommendation table) — presentation polish only, not new computation.

None of the above introduces a new recommendation algorithm, a hybrid model, or any technique excluded from this project's scope.

---