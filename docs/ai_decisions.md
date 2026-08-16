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

Decision:
Recommendations are generated only on explicit button click in the sidebar rather than automatically on every widget change.

Reason:
Provides clear control over computation and avoids unnecessary execution during parameter adjustment.

Date:
2026-07-20

---

## Phase 7 & 8 Itinerary Generation and Evaluation decisions

Decision:
Restore and implement Itinerary Generation (`src/itinerary.py`) and Itinerary Evaluation (`src/itinerary_evaluation.py`) as downstream components. (This supersedes the Phase 4 decision to remove them).

Reason:
Downstream itinerary planning and structural evaluation (measuring carryover, compactness, consecutive distance, and day balance) are crucial to answering the research objectives RO4 and RO5, showing how recommendations are translated into practical day plans.

Date:
2026-08-13

---

## Phase 9 Coordinates validation decisions

Decision:
Perform a 5-tier coordinate geocoding validation run using OpenStreetMap Nominatim and classification logic (`EXACT_MATCH`, `HIERARCHICALLY_CONSISTENT`, `BOUNDARY_AMBIGUOUS`, `TRUE_MISMATCH`, and `ERROR`).

Reason:
Allows the system to automatically distinguish between prefecture/district nesting and administrative level name discrepancies while isolating genuine coordinate errors for manual correction.

Date:
2026-08-14

---

## Phase 10 UI/UX Redesign decisions

Decision:
Overhaul `app.py` to use a top-level horizontal navigation layout (Overview, Recommendations, Itinerary, Evaluation), integrate an inline "search-bar" preferences form, organize itineraries under day tabs, and fix the HTML text rendering bug using dedented markdown templates.

Reason:
Improves system clarity, visual hierarchy, and demonstration value for the final year project defense. By moving settings to a top-level trip preference selector and wrapping day summaries in clean cards with visual connectors, the interface feels like a real-world travel booking app rather than a database admin dashboard.

Date:
2026-08-14

---

## Phase 11 Geographic Coherence Filtering decisions

Decision:
Implement Anchor-Based Region Filtering inside `src/itinerary.py` using the Rank 1 attraction as the reference anchor, drop items exceeding $R_{max}$ (default 200 km), and display the filtered items in a dedicated collapsible expander in the UI. Keep `r_max` parameter backward-compatible by defaulting to a virtually infinite boundary (`100000.0` km).

Reason:
Solves the algorithm design limitation where unconstrained day-clustering forced nationwide destinations (e.g., Shanghai and Xi'an) into the same day. Anchor-based filtering aligns with real-world travel behavior, where tourists focus on a specific destination region per trip and save distant spots for future travel, without breaking the offline evaluation or notebook environments.

Date:
2026-08-14

---

## Phase 12 Capacity-Constrained Geographic Itinerary & Validation decisions

Decision:
Transition from unconstrained day clustering to Farthest-First Seeded Greedy Capacity-Constrained Clustering in `assign_days()`. Initialize day anchors using Farthest-First Traversal (FFT), and greedily assign remaining attractions based on a city-weighted Haversine distance matrix while enforcing `MAX_STOPS_PER_DAY = 3`. Enforce day validation check: $\lceil N / 3 \rceil \le D \le N$.

Reason:
Agglomerative and K-Means clustering algorithms are unconstrained in size, resulting in unrealistic day sizes (e.g. Day 1 having 6 stops). Seeded greedy capacity clustering guarantees that every day contains at least one stop, no day exceeds 3 stops, and day assignments are distributed as evenly as possible. FFT ensures geographically dispersed anchors, and city-weighted Haversine distances group same-city and nearby attractions first. The day validation check prevents generating itineraries when there are too few days to accommodate the attractions or more days than attractions.

Date:
2026-08-14

---

## Phase 13 Geographic Candidate Selection and Balanced Clustering decisions

Decision:
Introduce a geographic candidate selection filtering stage before day clustering. Treat the Top-N recommendation pool as candidates and select a geographically coherent subset (attractions in the same city as the Rank 1 anchor or within `r_max=200` km of it). Exclude distant outliers and present them separately. Enforce day validation range $1 \le D \le N_{selected}$ where $N_{selected}$ is the count of compatible selected attractions, and raise ValueError if violated. Run K-Means on the selected subset, followed by day balancing and local swap improvements.

Reason:
Treating all Top-N recommendations as mandatory stops forced distant, geographically scattered destinations (e.g. Shenzhen, Shenyang, Xi'an on a Shanghai-centered trip) into the daily routes, making them highly unrealistic. Filtering out isolated outliers before day clustering ensures that only regional destinations are assigned, keeping total daily travel distances highly realistic. Flexible day bounds ($D \le N_{selected}$) allow custom travel planning for any number of selected coherent destinations.

Date:
2026-08-14

---

## Phase 14 Type Safety Alignment & Robustness decisions

Decision:
1. Explicitly cast the unpacked outputs of `build_itinerary(...)` to `Tuple[pd.DataFrame, pd.DataFrame]` in `app.py` when running with `return_excluded=True`.
2. Reset indices using `.reset_index(drop=True)` in `build_itinerary`, `select_geographic_candidates`, and `order_day` to make operations robust against duplicate index labels.
3. Group by a temporary clean column name directly in `select_geographic_candidates` instead of grouping a sliced DataFrame by a series of the original full DataFrame length.
4. Remove redundant/unnecessary `float()` typecast wrappers around pandas Series mean/sum operations and list min operations where the arguments are already of type `float`.

Reason:
1. `build_itinerary(...)` returns a union type. Pyright/mypy assumes unpacking a standalone `DataFrame` iterates over columns yielding `Hashable` keys. This results in the unpacked variables being typed as `DataFrame | Hashable`. Explicit casting using `typing.cast` corrects this behavior.
2. In pandas, duplicate index values in input dataframes can cause `.loc` retrieval to return multiple rows (a DataFrame instead of a Series) or crash with `ValueError: cannot reindex on an axis with duplicate labels` when doing index-aligned operations like slicing and grouping. Resetting the index at function entries prevents all duplicate-index crash modes.
3. Grouping a sliced DataFrame by a full-length series triggers index alignment. If the index contains duplicate labels, pandas cannot align the indices and throws an error. Grouping by the column name directly on a clean copy completely avoids index alignment issues.
4. Redundant/unnecessary `float()` conversions clutter the code and trigger static analysis warnings since pandas numeric aggregations (`mean()`, `sum()`) and `min()` on lists of float values already yield standard float types.

Date:
2026-08-15

---

## Phase 15 Unified Itinerary Semantics & Bug Fix decisions

Decision:
1. Remove `r_max > 10000.0` bypass conditions in `build_itinerary()` and `enforce_day_capacity()` in `src/itinerary.py`. The same geographic selection, day clustering, capacity enforcement, and pace balancing pipeline will execute consistently across web application and test suite environments.
2. Pad `kmeans_cluster_sizes` list to the requested trip duration (`num_days`) with zeros in `assign_days()` when the geographically selected candidate count is less than the requested trip duration.
3. Drop the `province` column from input recommendations inside `_join_coordinates()` if present, avoiding duplicate `province_x`/`province_y` columns during `pd.merge` left-joins.
4. Align assertions in `scratch/test_itinerary.py` to allow empty/unpopulated days and correct expected sizes based on geographically filtered candidate counts.

Reason:
1. The dual-mode execution logic based on `r_max > 10000.0` created two different algorithm behaviors: an interactive mode and an offline mode. This bypassed core features (geographic candidate selection and day capacity constraints) in testing environments. Removing the bypasses guarantees that tests and app logic remain 100% consistent.
2. Silently reducing the trip duration from 4 days to 3 when only 3 compatible attractions were selected violated trip duration integrity. Padding the sizes list keeps the requested duration intact, while the UI gracefully displays unpopulated days with clear user guidance rather than forcing distant incompatible recommendations.
3. Input recommendations that explicitly include a `province` column resulted in duplicate name suffixes during left-join, crashing subsequent province checks. Proactively dropping the column prior to joining ensures a clean single `province` attribute exists in the merged DataFrame.
4. In test suites, asserting that the final itinerary length matches the Top-N candidate pool is incorrect because the pipeline is designed to filter out geographically incoherent candidates rather than force them. Updating assertions ensures the test suite describes and verifies the correct unified algorithm.

Date:
2026-08-15