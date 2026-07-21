# AI Decisions

---

## Decision 001
Title:
Collaborative Filtering Algorithm

Decision:
Use User-Based Collaborative Filtering.

Reason:
The research objective is to compare traditional CBF and CF.
SVD introduces additional latent-factor modelling beyond the intended scope.

Status:
Approved

Date:
2026-07-18

Decision 002

Title
Unique Attraction Identifier

Decision

Create attraction_uid by concatenating:

attraction_name
city
province

Status

Approved

Reason

Two attractions share the same name but represent different locations.
Using attraction_name alone causes ambiguity.

Date

2026-07-18



## Phase 3 Implementation Decisions

Decision:
Use `stop_words="english"` in the TF-IDF vectorizer.

Reason:
Remove common English function words that do not contribute meaningful semantic information, producing cleaner feature vectors while preserving the overall recommendation architecture.

---

Decision:
Visited attraction lookup uses a Python `set`.

Reason:
Provides average O(1) membership testing during recommendation filtering while leaving recommendation behaviour unchanged.

---

Decision:
Verification scripts use dynamically detected dataset sizes instead of hardcoded attraction counts.

Reason:
Improves maintainability and prevents verification failures if the dataset changes in future experiments.

---

Decision:
Explicit `__all__` declaration added to `content_based.py`.

Reason:
Defines the intended public API of the module and improves module maintainability.

Date

2026-07-19

---

## Phase 4 Implementation Decisions

Decision:
Phase 4 Collaborative Filtering is implemented as memory-based User-Based Collaborative Filtering (KNN) with Cosine Similarity, and the previously explored SVD/matrix factorization variant is superseded.

Reason:
To strictly conform to project coursework boundaries requiring traditional memory-based collaborative filtering.

Date:
2026-07-19

---

Decision:
Calculate user cosine similarities on overlapping rated items only, leaving unobserved cells as missing (NaN) instead of zero-filling or mean imputing.

Reason:
Preserves the raw observed ratings to prevent introducing synthetic values into the pairwise distance measurements, aligning with the memory-based collaborative filtering architecture.

Date:
2026-07-19

---

Decision:
The primary public recommendation interface is recommend_attractions_cf.

Reason:
Ensures a clean public interface and encapsulates matrix pivoting, similarity calculations, neighbor search, and rating prediction.

Date:
2026-07-19

---

Decision:
Name the public function `recommend_attractions_cf` instead of `recommend_attractions`.

Reason:
Avoids name clashes in downstream modules (e.g. `app.py`) if both collaborative and content-based recommenders are imported unqualified.

Date:
2026-07-19

---

Decision:
Perform Cosine Similarity using a vectorized NumPy formulation where overlapping norms and dot products are computed simultaneously.

Reason:
Significantly improves processing speed, lowering comparison time for 10,000 users from minutes to under 10 seconds.

Date:
2026-07-19

---

Decision:
Exposed K (default: 20) and Top-N (default: 10) as configurable parameters.

Reason:
Enables easy validation and tuning during Phase 5 evaluation.

Date:
2026-07-19

---

Decision:
If a user has no neighbors or no predictions can be made (cold-start), return an empty DataFrame with the output columns.

Reason:
Avoids substituting arbitrary fallbacks and keeps coverage limits visible for evaluation.

Date:
2026-07-19
---

Decision:
Remove the itinerary recommendation module and associated sequencing logic.

Reason:
Itinerary sequencing is out of the project scope, which is focused entirely on comparing traditional Content-Based and Collaborative Filtering algorithms.

Date:
2026-07-19

---

Decision:
Do not persist trained models (such as TF-IDF matrices, attraction features, and similarity matrices) to disk.

Reason:
At the project's scale (~100,000 interactions and 433 attractions), in-memory execution combined with Streamlit UI caching is highly efficient. Eliminating disk persistence reduces code complexity and eliminates file storage overhead.

Date:
2026-07-19

---

## Documentation Decisions

Decision:
Notebook demonstrations reuse the finalized implementation from src/ modules instead of duplicating algorithm implementations.

Reason:
Maintains a single source of truth for all algorithms, improves maintainability, and ensures notebook outputs always reflect the production implementation.

Date:
2026-07-19

---

Decision:
Phase notebooks are used for demonstration, analysis, and presentation only.

Reason:
Keeps implementation inside reusable Python modules while notebooks document experimental workflow and results.

Date:
2026-07-19

---

## Phase 6 Streamlit Prototype Decisions

Decision:
app.py contains no recommendation, similarity, or metric computation logic. It calls only the existing public functions content_based.recommend_attractions() and collaborative.recommend_attractions_cf().

Reason:
Keeps the presentation layer strictly decoupled from model logic, consistent with the public-interface boundaries already established in Phase 3 and Phase 4.

Date:
2026-07-20

---

Decision:
Each model's precomputed context (TF-IDF matrix and attraction index for CBF; user-item matrix, similarity matrix, and index mappings for CF) is built once per app session via caching (st.cache_resource), not rebuilt on every recommendation request.

Reason:
Mirrors the "fit once, reuse many" principle established for matrix-building steps; prevents rebuilding full matrices on every user interaction.

Date:
2026-07-20

---

Decision:
app.py does not use evaluation.py in any capacity (neither live nor by reading precomputed metric files).

Reason:
Keeps Phase 6 a prototype rather than a dashboard; Phase 5's evaluation results remain an offline deliverable, preventing scope creep into evaluation metric reporting.

Date:
2026-07-20

---

Decision:
Recommendations are generated only on explicit button click in the sidebar rather than automatically on every widget change.

Reason:
Provides clear control over computation and avoids unnecessary execution during parameter adjustment.

Date:
2026-07-20