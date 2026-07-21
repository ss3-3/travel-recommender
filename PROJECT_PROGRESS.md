# Project Progress

## Phase 1 — Exploratory Data Analysis (EDA)

**Status:** ✅ Completed
**Completed On:** 2026-07-18

### Summary
- Loaded and analyzed the dataset (100,000 interactions, 10,000 users, 433 physical attractions).
- Verified missing values correspond to independent travellers.
- Identified duplicate attraction names and designed a unique attraction identifier.
- Measured user-item matrix sparsity (97.69%).
- Defined preprocessing and feature engineering strategies for both recommendation models.

Artifacts
✓ notebooks/01_EDA.ipynb
✓ EDA completed
✓ Feature engineering blueprint completed

---

## Phase 2 — Data Preprocessing

**Status:** ✅ Completed
**Completed On:** 2026-07-18

### Summary
- Implemented dataset loading and columns validation check functions.
- Implemented text cleaning and normalizing functions (lowercase, punctuation stripping, extra spacing reduction).
- Constructed unique key identifier `attraction_uid` for each physical attraction to resolve naming duplicates.
- Prepared deduplicated static attraction features (excluding the `season` attribute and filling missing `main_spots` with empty strings).
- Isolated tourist user-item ratings interaction records for Collaborative Filtering.
- Added and verified a reusable, stratified `train_test_split_by_user` function to support fair model comparisons.
- Verified module output shapes and functionalities using `scratch/test_preprocessing.py` and `scratch/verify_split.py`.

Artifacts
✓ src/preprocessing.py
✓ Core Preprocessing module completed
✓ Reusable train-test split utility implemented

---

## Phase 3 — Content-Based Filtering (CBF)

**Status:** ✅ Completed
**Completed On:** 2026-07-19

### Summary
- Implemented `build_content_column` to merge attraction categorical and text attributes, applying category and level repetitions (weighted scoring).
- Implemented `build_tfidf_matrix` to fit and transform the text corpus into a sparse feature matrix.
- Implemented `build_user_profile` to generate user vectors via rating-weighted averages for positive ratings (>= 4.0), handling the cold-start edge case.
- Implemented `compute_similarity` using scikit-learn's cosine similarity pairwise calculator.
- Implemented the public interface `recommend_attractions` to coordinate profiling, scoring, visited filtering, and metadata mapping.
- Verified all modular behaviors, shapes, and metrics using `scratch/test_content_based.py`.

Artifacts
✓ src/content_based.py
✓ Verification script scratch/test_content_based.py completed

### Maintenance Update
**Completed On:** 2026-07-19

- Refined TF-IDF preprocessing by enabling English stop-word removal.
- Optimized visited-attraction filtering using Python sets.
- Added explicit module exports via `__all__`.
- Updated verification scripts to dynamically adapt to dataset size.
- Improved code formatting, comments, and maintainability without changing recommendation behaviour.

---

## Phase 4 — User-Based Collaborative Filtering

**Status:** ✅ Completed
**Completed On:** 2026-07-19

### Summary
- Replaced the obsolete SVD design with memory-based User-Based Collaborative Filtering (KNN) using Cosine Similarity.
- Pivoted training interactions into a sparse `n_users` x `n_attractions` matrix without imputation.
- Implemented `build_user_similarity_matrix` to calculate pairwise user cosine similarities restricted to overlapping rated items.
- Implemented `find_nearest_neighbors` to extract the top-K neighbours per user, with deterministic tie-breaking (by tourist_id) and exclusion of target/negative similarities.
- Implemented similarity-weighted average prediction calculation for candidate attractions in `predict_ratings`.
- Implemented public interface `recommend_attractions_cf` to coordinate candidate rating prediction, visited training spot exclusion, and display metadata joins.
- Created `scratch/test_collaborative.py` to perform a comprehensive validation suite covering matrix construction, cosine similarity computation, nearest neighbour retrieval, weighted rating prediction, visited-attraction filtering, cold-start handling, and empty matrix edge cases.

Artifacts
✓ src/collaborative.py
✓ Verification script scratch/test_collaborative.py completed (validating matrix construction, cosine similarity computation, nearest neighbour retrieval, weighted rating prediction, visited-attraction filtering, cold-start handling, and edge cases)
✓ notebooks/03_collaborative.ipynb

### Maintenance Update
**Completed On:** 2026-07-19

- Completed notebooks/03_collaborative.ipynb
- Notebook now demonstrates the finalized User-Based Collaborative Filtering workflow.
- Notebook imports the implementation from src/collaborative.py instead of duplicating algorithm code.
- Added demonstration of user-item matrix construction, cosine similarity matrix generation, nearest-neighbour retrieval, recommendation generation, cold-start behaviour, and discussion.
- Synchronized `docs/architecture.md` with the current modular project codebase.
- Replaced the planned notebook structure to only show `01_EDA.ipynb` and `05_Evaluation.ipynb`, as Phase 2, 3, and 4 algorithms are fully implemented under `src/` modules.
- Removed `itinerary.py` and itinerary recommendation descriptions from the architecture design since it is out of project scope.
- Removed persisted model file descriptions (e.g. TF-IDF vectorizer, attraction vectors, and similarity matrix) from the architecture and updated descriptions to reflect that the system does not persist trained models.
- Updated `src/` directory layout to show `preprocessing.py`, `content_based.py`, `collaborative.py`, `evaluation.py`, and `utils.py`.
- Updated `scratch/` directory layout to show verification scripts.
- Refined function descriptions for all files in the architecture design to match actual implemented function signatures and responsibilities.

---

## Phase 5 — Model Evaluation

**Status:** ✅ Completed
**Completed On:** 2026-07-19

### Summary
- Implemented core evaluation metrics: `precision_at_k`, `recall_at_k`, `f1_at_k`, and `coverage`.
- Developed `evaluate_model` to orchestrate model evaluation, handling parameters dynamically and filtering arguments using reflection to guarantee compatibility with existing model APIs without modification.
- Implemented `compare_models` to concatenate model results side-by-side.
- Added `run_full_evaluation` as a single orchestrator entry point.
- Created `scratch/test_evaluation.py` to run unit assertions on metrics, loop logic, user cold-starts, and end-to-end dataset sampling.
- Generated `notebooks/04_Evaluation.ipynb` and `notebooks/05_Evaluation.ipynb` to load preprocessed splits, run both models, compare results, and display bar charts.

Artifacts
✓ src/evaluation.py
✓ scratch/test_evaluation.py
✓ notebooks/04_Evaluation.ipynb
✓ notebooks/05_Evaluation.ipynb

---

## Phase 6 — Streamlit Prototype

**Status:** ✅ Completed
**Completed On:** 2026-07-20

### Summary
- Implemented `app.py` as a Streamlit presentation and orchestration layer.
- Configured cached data loading (`load_data`) using `@st.cache_data`.
- Configured cached model context initializations (`get_cbf_context`, `get_cf_context`) using `@st.cache_resource`.
- Built interactive sidebar controls including Tourist ID selection, model algorithm selection (Content-Based vs Collaborative), Top-N slider, and recommendation generation trigger.
- Integrated `recommend_attractions` and `recommend_attractions_cf` directly from `src/` modules without duplicating recommendation logic.
- Implemented `render_results` to render ranked recommendation tables or present model-specific cold-start warning messages when an empty DataFrame is returned.

Artifacts
✓ app.py
