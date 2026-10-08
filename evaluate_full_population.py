"""
Full-population evaluation of CBF and UBCF, at both the recommendation level
and the itinerary level, across every qualifying test user (not a 200-user
subset).

This script is the actual, reproducible source of the numbers reported in
Section 4.2 (Table 3), Section 4.3 (Table 4), Table 6, the Abstract, and the
Conclusion of the paper. It imports the project's real, unmodified src/
modules (src.content_based, src.collaborative, src.itinerary,
src.itinerary_evaluation, src.evaluation) -- the exact same recommendation
and itinerary logic used everywhere else in this project. Nothing in src/ is
changed or reimplemented here.

The one addition is `build_similarity_rows_for_users` below: a memory-
efficient equivalent of src.collaborative.build_user_similarity_matrix,
restricted to computing only the similarity-matrix rows needed for the given
users. The full pairwise cosine-similarity matrix for all ~10,000 users
needs several dense (n_users x n_users) float64 temporaries (~800MB each),
which does not fit in memory when evaluating the full population at once.
This helper computes the identical cosine-similarity formula one target row
at a time instead, using the same vectorised numerator/denominator terms as
the original matrix computation (verified algebraically identical, since
cosine similarity is symmetric and each row depends only on that user's
ratings against every other user's ratings). It is a memory optimisation,
not an algorithm change.

Usage:
    python scripts/evaluate_full_population.py

Outputs (written to results/, created if missing):
    results/full_population_recommendation_metrics.csv  -- per-model summary
    results/full_population_itinerary_results.csv        -- per-user, per-model rows
Also prints a summary to stdout in the same shape as the paper's Table 3,
Table 4, and Table 6, for direct comparison against the submitted numbers.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
# Works whether this file sits in the project root or in scripts/
PROJECT_DIR = _HERE if (_HERE / "src").is_dir() else _HERE.parent
sys.path.append(str(PROJECT_DIR))

from src.preprocessing import load_dataset, prepare_attractions, prepare_interactions, train_test_split_by_user
from src.content_based import build_content_column, build_tfidf_matrix, recommend_attractions
from src.collaborative import build_user_item_matrix, recommend_attractions_cf
from src.itinerary import build_one_day_itinerary, load_coordinates
from src.itinerary_evaluation import evaluate_itinerary
from src.evaluation import evaluate_model


def build_similarity_rows_for_users(user_item_matrix, target_row_indices):
    """Memory-efficient equivalent of build_user_similarity_matrix, computing
    only the rows in `target_row_indices`. See module docstring for details."""
    n_users = user_item_matrix.shape[0]
    mask = ~np.isnan(user_item_matrix)
    matrix_zeroed = np.nan_to_num(user_item_matrix, nan=0.0).astype(np.float64)
    matrix_squared = matrix_zeroed ** 2
    mask_float = mask.astype(np.float64)

    sim_matrix = np.zeros((n_users, n_users), dtype=np.float32)

    for u in target_row_indices:
        v_u = matrix_zeroed[u]
        mask_u = mask_float[u]
        sq_u = matrix_squared[u]

        dot_row = matrix_zeroed @ v_u
        norm_i_row = np.sqrt(matrix_squared @ mask_u)
        norm_j_row = np.sqrt(mask_float @ sq_u)
        denom_row = norm_i_row * norm_j_row

        row = np.zeros(n_users, dtype=np.float64)
        valid = denom_row > 0.0
        row[valid] = dot_row[valid] / denom_row[valid]
        row = np.clip(row, 0.0, 1.0)
        row[u] = 1.0
        sim_matrix[u] = row.astype(np.float32)

    return sim_matrix


def main():
    t_start = time.time()
    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    csv_path = PROJECT_DIR / "data" / "tourism_recommendation_dataset_en.csv"
    coords_path = PROJECT_DIR / "data" / "coordinates.csv"

    df = load_dataset(str(csv_path))
    attraction_df = prepare_attractions(df)
    interactions_df = prepare_interactions(df)
    coordinates_df = load_coordinates(str(coords_path))

    train_df, test_df = train_test_split_by_user(interactions_df, test_ratio=0.2, min_interactions=5, random_state=42)

    cbf_df = build_content_column(attraction_df)
    vectorizer, tfidf_matrix, attraction_index = build_tfidf_matrix(cbf_df)
    user_item_matrix, user_index, cf_attraction_index = build_user_item_matrix(train_df)

    test_users = list(test_df["tourist_id"].unique())
    print(f"Full qualifying test user count: {len(test_users)}")

    target_rows = [user_index[u] for u in test_users if u in user_index]
    print(f"Building UBCF similarity rows for {len(target_rows)} users...")
    t0 = time.time()
    user_similarity_matrix = build_similarity_rows_for_users(user_item_matrix, target_rows)
    print(f"  similarity build took {time.time() - t0:.1f}s")
    import gc
    gc.collect()

    # ---- Part 1: recommendation-level metrics (Table 3) ----
    print("\n=== Recommendation-level evaluation (Precision/Recall/F1/nDCG/Coverage) ===")
    cbf_context = {"attraction_df": attraction_df, "tfidf_matrix": tfidf_matrix, "attraction_index": attraction_index}
    cf_context = {
        "attraction_df": attraction_df, "user_item_matrix": user_item_matrix,
        "user_similarity_matrix": user_similarity_matrix, "user_index": user_index,
        "attraction_index": cf_attraction_index,
    }
    t0 = time.time()
    cbf_summary, cbf_per_user = evaluate_model(recommend_attractions, test_users, train_df, test_df, cbf_context, top_n=10)
    print(f"  CBF eval took {time.time() - t0:.1f}s")
    t0 = time.time()
    cf_summary, cf_per_user = evaluate_model(recommend_attractions_cf, test_users, train_df, test_df, cf_context, top_n=10, model_kwargs={"k": 20})
    print(f"  UBCF eval took {time.time() - t0:.1f}s")

    cbf_summary.insert(0, "model", "CBF")
    cf_summary.insert(0, "model", "UBCF")
    rec_summary = pd.concat([cbf_summary, cf_summary], ignore_index=True)
    rec_summary.to_csv(results_dir / "full_population_recommendation_metrics.csv", index=False)
    print("\nTable 3 comparison (n=%d test users):" % len(test_users))
    print(rec_summary.to_string(index=False))

    # ---- Part 2: itinerary-level metrics (Table 4 / Table 6) ----
    print("\n=== Itinerary-level evaluation (multi-stop rate, carryover, distances) ===")
    requested_destinations = 4
    top_n = 10
    records = []
    t0 = time.time()
    for i, uid in enumerate(test_users):
        for model_name, recs in [
            ("CBF", recommend_attractions(
                tourist_id=uid, interactions_df=train_df, attraction_df=attraction_df,
                tfidf_matrix=tfidf_matrix, attraction_index=attraction_index, top_n=top_n
            )),
            ("UBCF", recommend_attractions_cf(
                tourist_id=uid, train_df=train_df, attraction_df=attraction_df,
                user_item_matrix=user_item_matrix, user_similarity_matrix=user_similarity_matrix,
                user_index=user_index, attraction_index=cf_attraction_index, k=20, top_n=top_n
            )),
        ]:
            if recs.empty:
                records.append({
                    "tourist_id": uid, "model": model_name, "num_candidates": 0,
                    "selected_stops": 0, "carryover_rate": np.nan,
                    "avg_consecutive_distance": np.nan, "total_travel_distance": np.nan,
                    "geographic_compactness": np.nan, "had_candidates": False,
                })
                continue
            itinerary, _excluded = build_one_day_itinerary(recs, coordinates_df, requested_destinations, return_excluded=True)
            metrics = evaluate_itinerary(itinerary, recs)
            records.append({
                "tourist_id": uid, "model": model_name, "num_candidates": len(recs),
                "selected_stops": len(itinerary),
                "carryover_rate": metrics["candidate_carryover_rate"],
                "avg_consecutive_distance": metrics["avg_consecutive_distance"],
                "total_travel_distance": metrics["total_travel_distance"],
                "geographic_compactness": metrics["geographic_compactness"],
                "had_candidates": True,
            })
        if (i + 1) % 2000 == 0:
            print(f"  ...{i + 1}/{len(test_users)} users done, {time.time() - t0:.1f}s elapsed")

    print(f"  itinerary loop took {time.time() - t0:.1f}s")
    results_df = pd.DataFrame(records)
    results_df.to_csv(results_dir / "full_population_itinerary_results.csv", index=False)

    print(f"\nTable 4 / Table 6 comparison (n={len(test_users)} test users):")
    for model_name in ["CBF", "UBCF"]:
        sub = results_df[results_df["model"] == model_name]
        n_multi_stop = (sub["selected_stops"] >= 2).sum()
        print(f"\n--- {model_name} ---")
        print(f"Multi-stop itineraries: {n_multi_stop} / {len(sub)} ({100 * n_multi_stop / len(sub):.1f}%)")
        print(f"Mean selected stops (of {requested_destinations} requested): {sub['selected_stops'].mean():.3f} (SD {sub['selected_stops'].std():.3f})")
        print(f"Mean candidate carryover rate: {100 * sub['carryover_rate'].mean():.2f}% (SD {100 * sub['carryover_rate'].std():.2f}%)")
        print(f"Mean total travel distance: {sub['total_travel_distance'].mean():.2f} km (SD {sub['total_travel_distance'].std():.2f} km)")
        multi = sub[sub["selected_stops"] >= 2]
        print(f"Mean avg consecutive-stop distance (multi-stop only): {multi['avg_consecutive_distance'].mean():.2f} km (SD {multi['avg_consecutive_distance'].std():.2f} km)")
        print(f"Mean geographic compactness (multi-stop only): {multi['geographic_compactness'].mean():.2f} km (SD {multi['geographic_compactness'].std():.2f} km)")

    pivot_stops = results_df.pivot(index="tourist_id", columns="model", values="selected_stops")
    both = pivot_stops.dropna()
    cbf_more = (both["CBF"] > both["UBCF"]).sum()
    ubcf_more = (both["UBCF"] > both["CBF"]).sum()
    tie = (both["CBF"] == both["UBCF"]).sum()
    print(f"\nPaired comparison (both models produced recommendations): {len(both)} users")
    print(f"CBF retained more stops than UBCF: {cbf_more} users")
    print(f"UBCF retained more stops than CBF: {ubcf_more} users")
    print(f"Tie (same number of stops): {tie} users")

    print(f"\nTotal elapsed: {time.time() - t_start:.1f}s")
    print(f"\nResults written to {results_dir}/")


if __name__ == "__main__":
    main()
