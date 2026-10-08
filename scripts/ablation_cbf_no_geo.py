"""
Ablation: CBF with geographic text tokens (province, city) removed from the
content representation.

Addresses reviewer request: "Remove geographic text (city/province) from CBF
vector representations and re-evaluate to empirically prove whether CBF's
spatial clustering superiority stems from location tokens or category tags."

Nothing in src/ is modified. The original build_content_column() is untouched;
this file defines a parallel function that drops `province` and `city` and
keeps attraction_category x2, attraction_level x2 and main_spots_clean, then
runs the SAME full-population protocol as evaluate_full_population.py
(same split, same test users, Top-N = 10, 4 requested stops, 50 km radius).

Usage:  python scripts/ablation_cbf_no_geo.py
Outputs (results/):
    ablation_cbf_no_geo_recommendation_metrics.csv   (CBF original vs CBF no-geo)
    ablation_cbf_no_geo_itinerary_results.csv        (per-user rows, both variants)
    ablation_cbf_no_geo_summary.txt                  (printed summary)
"""
import sys, time
from pathlib import Path
from typing import cast
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from src.preprocessing import load_dataset, prepare_attractions, prepare_interactions, train_test_split_by_user
from src.content_based import build_content_column, build_tfidf_matrix, recommend_attractions
from src.itinerary import build_one_day_itinerary, load_coordinates
from src.itinerary_evaluation import evaluate_itinerary
from src.evaluation import evaluate_model


def build_content_column_no_geo(attraction_df: pd.DataFrame) -> pd.DataFrame:
    """Ablated copy of build_content_column(): province and city removed.
    Keeps category x2, level x2, main_spots_clean (same repetition weighting)."""
    df = attraction_df.copy()
    cat = df["attraction_category"].fillna("").astype(str)
    lvl = df["attraction_level"].fillna("").astype(str)
    spots = df["main_spots_clean"].fillna("").astype(str)
    content = cat.str.cat(cast(list[str], [cat, lvl, lvl, spots]), sep=" ")
    df["content"] = content.str.strip().str.lower()
    return df


def run_variant(name, content_builder, attraction_df, train_df, test_df, test_users, coordinates_df, lines):
    cdf = content_builder(attraction_df)
    vec, tfidf, aidx = build_tfidf_matrix(cdf)
    print(f"[{name}] TF-IDF shape = {tfidf.shape}", flush=True)
    lines.append(f"[{name}] TF-IDF shape = {tfidf.shape}")
    ctx = {"attraction_df": attraction_df, "tfidf_matrix": tfidf, "attraction_index": aidx}
    summ, per_user = evaluate_model(recommend_attractions, test_users, train_df, test_df, ctx, top_n=10)
    summ.insert(0, "model", name)

    records, top10 = [], {}
    for uid in test_users:
        recs = recommend_attractions(tourist_id=uid, interactions_df=train_df, attraction_df=attraction_df,
                                     tfidf_matrix=tfidf, attraction_index=aidx, top_n=10)
        top10[uid] = list(recs["attraction_uid"]) if not recs.empty else []
        if recs.empty:
            records.append({"tourist_id": uid, "model": name, "num_candidates": 0, "selected_stops": 0,
                            "carryover_rate": np.nan, "avg_consecutive_distance": np.nan,
                            "total_travel_distance": np.nan, "geographic_compactness": np.nan})
            continue
        itin, _ = build_one_day_itinerary(recs, coordinates_df, 4, return_excluded=True)
        m = evaluate_itinerary(itin, recs)
        records.append({"tourist_id": uid, "model": name, "num_candidates": len(recs),
                        "selected_stops": len(itin), "carryover_rate": m["candidate_carryover_rate"],
                        "avg_consecutive_distance": m["avg_consecutive_distance"],
                        "total_travel_distance": m["total_travel_distance"],
                        "geographic_compactness": m["geographic_compactness"]})
    return summ, per_user, pd.DataFrame(records), top10, (cdf, vec, tfidf, aidx)


def describe(name, sub, lines):
    n = len(sub)
    multi = sub[sub["selected_stops"] >= 2]
    out = [f"--- {name} (n={n}) ---",
           f"Multi-stop itineraries: {len(multi)} / {n} ({100*len(multi)/n:.1f}%)",
           f"Mean selected stops (of 4): {sub['selected_stops'].mean():.3f} (SD {sub['selected_stops'].std():.3f})",
           f"Mean candidate carryover: {100*sub['carryover_rate'].mean():.2f}% (SD {100*sub['carryover_rate'].std():.2f}%)",
           f"Mean total travel distance: {sub['total_travel_distance'].mean():.2f} km (SD {sub['total_travel_distance'].std():.2f})",
           f"Mean avg consecutive-stop distance (multi-stop): {multi['avg_consecutive_distance'].mean():.2f} km (SD {multi['avg_consecutive_distance'].std():.2f})",
           f"Mean geographic compactness (multi-stop): {multi['geographic_compactness'].mean():.2f} km (SD {multi['geographic_compactness'].std():.2f})"]
    for o in out:
        print(o); lines.append(o)


def main():
    t0 = time.time(); lines = []
    res = PROJECT_DIR / "results"; res.mkdir(exist_ok=True)
    df = load_dataset(str(PROJECT_DIR / "data" / "tourism_recommendation_dataset_en.csv"))
    attraction_df = prepare_attractions(df)
    interactions_df = prepare_interactions(df)
    coordinates_df = load_coordinates(str(PROJECT_DIR / "data" / "coordinates.csv"))
    train_df, test_df = train_test_split_by_user(interactions_df, test_ratio=0.2, min_interactions=5, random_state=42)
    test_users = list(test_df["tourist_id"].unique())
    print(f"Test users: {len(test_users)}"); lines.append(f"Test users: {len(test_users)}")

    s0, pu0, it0, top0, _ = run_variant("CBF (original)", build_content_column, attraction_df, train_df, test_df, test_users, coordinates_df, lines)
    s1, pu1, it1, top1, (cdf1, _, tfidf1, aidx1) = run_variant("CBF (no province/city)", build_content_column_no_geo, attraction_df, train_df, test_df, test_users, coordinates_df, lines)

    rec = pd.concat([s0, s1], ignore_index=True)
    rec.to_csv(res / "ablation_cbf_no_geo_recommendation_metrics.csv", index=False)
    pd.concat([it0, it1], ignore_index=True).to_csv(res / "ablation_cbf_no_geo_itinerary_results.csv", index=False)
    print("\n=== Recommendation-level ==="); print(rec.to_string(index=False))
    lines += ["", "=== Recommendation-level ===", rec.to_string(index=False)]

    print("\n=== Itinerary-level ==="); lines += ["", "=== Itinerary-level ==="]
    describe("CBF (original)", it0, lines); describe("CBF (no province/city)", it1, lines)

    # Extra diagnostic: how much do the Top-10 lists change?
    ov = [len(set(top0[u]) & set(top1[u])) / 10 for u in test_users if top0[u] and top1[u]]
    msg = f"\nMean Top-10 overlap between original and no-geo CBF: {np.mean(ov):.3f}"
    print(msg); lines.append(msg)

    # Paired stops
    p = pd.concat([it0, it1]).pivot(index="tourist_id", columns="model", values="selected_stops")
    a, b = p["CBF (original)"], p["CBF (no province/city)"]
    msg = f"Paired stops: original>no-geo {(a>b).sum()}, no-geo>original {(b>a).sum()}, tie {(a==b).sum()}"
    print(msg); lines.append(msg)

    # Worked example: Tourist ID 1
    lines.append(""); print()
    for name, tf, ai, cd in [("CBF (original)", None, None, None)]:
        pass
    for name, builder in [("CBF (original)", build_content_column), ("CBF (no province/city)", build_content_column_no_geo)]:
        cdf = builder(attraction_df); _, tf, ai = build_tfidf_matrix(cdf)
        recs = recommend_attractions(tourist_id=1, interactions_df=train_df, attraction_df=attraction_df, tfidf_matrix=tf, attraction_index=ai, top_n=10)
        itin, _ = build_one_day_itinerary(recs, coordinates_df, 4, return_excluded=True)
        m = evaluate_itinerary(itin, recs)
        txt = f"Tourist ID 1 | {name}: top10 cities={list(recs['city'])}\n   itinerary stops={len(itin)} metrics={m}"
        print(txt); lines.append(txt)

    (res / "ablation_cbf_no_geo_summary.txt").write_text("\n".join(lines))
    print(f"\nElapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
