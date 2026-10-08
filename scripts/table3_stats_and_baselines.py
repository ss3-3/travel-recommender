"""
Supplementary statistics for Table 3 (reviewer requests). No src/ file is modified.

Parts (run one at a time, `python scripts/table3_stats_and_baselines.py <part>`):
  stats      per-user mean / SD / 95% CI for CBF and UBCF + paired CBF-UBCF difference
  baselines  random and popularity baselines, same split/users/Top-10/metrics
  baselines_ci  per-user mean/SD/95% CI for the baselines (same basis as CBF/UBCF)
  ties       how often identical scores occur among candidate POIs at the Top-10
             boundary, and whether a deterministic tie-break changes the metrics

Same protocol as evaluate_full_population.py: random_state=42, 80/20 per-user
split, 9,744 test users, Top-N = 10, UBCF k = 20.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from src.preprocessing import load_dataset, prepare_attractions, prepare_interactions, train_test_split_by_user
from src.content_based import build_content_column, build_tfidf_matrix, recommend_attractions, build_user_profile, compute_similarity
from src.collaborative import (build_user_item_matrix, recommend_attractions_cf,
                               find_nearest_neighbors, predict_ratings)
from src.evaluation import (evaluate_model, precision_at_k, recall_at_k, f1_at_k,
                            ndcg_at_k, extract_ground_truth)

RES = PROJECT_DIR / "results"
METRICS = ["precision_at_k", "recall_at_k", "f1_at_k", "ndcg_at_k"]
K = 10


def setup():
    df = load_dataset(str(PROJECT_DIR / "data" / "tourism_recommendation_dataset_en.csv"))
    attraction_df = prepare_attractions(df)
    interactions_df = prepare_interactions(df)
    train_df, test_df = train_test_split_by_user(interactions_df, test_ratio=0.2, min_interactions=5, random_state=42)
    test_users = list(test_df["tourist_id"].unique())
    return attraction_df, train_df, test_df, test_users


def sim_rows(user_item_matrix, target_rows):
    """Memory-efficient equivalent of build_user_similarity_matrix (same as evaluate_full_population.py)."""
    n = user_item_matrix.shape[0]
    mask = ~np.isnan(user_item_matrix)
    z = np.nan_to_num(user_item_matrix, nan=0.0).astype(np.float64)
    sq = z ** 2
    mf = mask.astype(np.float64)
    S = np.zeros((n, n), dtype=np.float32)
    for u in target_rows:
        dot = z @ z[u]
        den = np.sqrt(sq @ mf[u]) * np.sqrt(mf @ sq[u])
        row = np.zeros(n)
        ok = den > 0
        row[ok] = dot[ok] / den[ok]
        row = np.clip(row, 0.0, 1.0)
        row[u] = 1.0
        S[u] = row.astype(np.float32)
    return S


def ci_row(x):
    x = np.asarray(x, dtype=float)
    n = len(x); m = x.mean(); sd = x.std(ddof=1); se = sd / np.sqrt(n)
    return m, sd, m - 1.96 * se, m + 1.96 * se


def part_stats():
    from scipy import stats
    attraction_df, train_df, test_df, test_users = setup()
    cbf_df = build_content_column(attraction_df)
    _, tfidf, aidx = build_tfidf_matrix(cbf_df)
    uim, uidx, cf_aidx = build_user_item_matrix(train_df)
    S = sim_rows(uim, [uidx[u] for u in test_users if u in uidx])
    _, pu_cbf = evaluate_model(recommend_attractions, test_users, train_df, test_df,
                               {"attraction_df": attraction_df, "tfidf_matrix": tfidf, "attraction_index": aidx}, top_n=K)
    print("CBF done", flush=True)
    _, pu_cf = evaluate_model(recommend_attractions_cf, test_users, train_df, test_df,
                              {"attraction_df": attraction_df, "user_item_matrix": uim, "user_similarity_matrix": S,
                               "user_index": uidx, "attraction_index": cf_aidx}, top_n=K, model_kwargs={"k": 20})
    pu_cbf.to_csv(RES / "per_user_metrics_cbf.csv", index=False)
    pu_cf.to_csv(RES / "per_user_metrics_ubcf.csv", index=False)

    rows = []
    for name, pu in [("CBF", pu_cbf), ("UBCF", pu_cf)]:
        pu = pu[pu["recommended_count"] > 0]
        for m in METRICS:
            mean, sd, lo, hi = ci_row(pu[m])
            rows.append({"model": name, "metric": m, "n_users": len(pu), "mean": mean, "sd": sd, "ci95_low": lo, "ci95_high": hi})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "table3_mean_sd_ci.csv", index=False)
    print(out.to_string(index=False))

    a = pu_cbf.set_index("tourist_id"); b = pu_cf.set_index("tourist_id")
    common = a.index.intersection(b.index)
    prow = []
    for m in METRICS:
        d = (a.loc[common, m] - b.loc[common, m]).values
        mean, sd, lo, hi = ci_row(d)
        try:
            w = stats.wilcoxon(d[d != 0]).pvalue
        except Exception:
            w = float("nan")
        t = stats.ttest_rel(a.loc[common, m], b.loc[common, m]).pvalue
        prow.append({"metric": m, "mean_diff_CBF_minus_UBCF": mean, "ci95_low": lo, "ci95_high": hi,
                     "paired_t_p": t, "wilcoxon_p": w, "n_pairs": len(common)})
    pdf = pd.DataFrame(prow)
    pdf.to_csv(RES / "table3_paired_difference.csv", index=False)
    print("\nPaired CBF - UBCF:"); print(pdf.to_string(index=False))


def part_baselines():
    attraction_df, train_df, test_df, test_users = setup()
    all_uids = np.array(sorted(attraction_df["attraction_uid"].unique()))
    rated = train_df.groupby("tourist_id")["attraction_uid"].apply(set).to_dict()
    pop = train_df.groupby("attraction_uid").size()
    pop_order = sorted(all_uids, key=lambda u: (-pop.get(u, 0), u))
    truth = {u: extract_ground_truth(u, test_df) for u in test_users}

    def score(rec_by_user):
        P, R, F, N = [], [], [], []
        for u in test_users:
            rec = rec_by_user[u]; t = truth[u]
            p = precision_at_k(rec, t); r = recall_at_k(rec, t)
            P.append(p); R.append(r); F.append(f1_at_k(p, r)); N.append(ndcg_at_k(rec, t, K))
        return np.mean(P), np.mean(R), np.mean(F), np.mean(N)

    rows = []
    # popularity (most-rated in training, excluding user's rated items)
    rec = {u: [x for x in pop_order if x not in rated.get(u, set())][:K] for u in test_users}
    rows.append({"baseline": "Popularity", "seed": "", **dict(zip(["precision_at_k", "recall_at_k", "f1_at_k", "ndcg_at_k"], score(rec)))})
    # random, 20 seeds
    per_seed = []
    for seed in range(20):
        rng = np.random.RandomState(seed)
        rec = {}
        for u in test_users:
            cand = [x for x in all_uids if x not in rated.get(u, set())]
            idx = rng.choice(len(cand), size=min(K, len(cand)), replace=False)
            rec[u] = [cand[i] for i in idx]
        s = score(rec); per_seed.append(s)
        rows.append({"baseline": "Random", "seed": seed, **dict(zip(["precision_at_k", "recall_at_k", "f1_at_k", "ndcg_at_k"], s))})
    arr = np.array(per_seed)
    rows.append({"baseline": "Random (mean of 20 seeds)", "seed": "", **dict(zip(["precision_at_k", "recall_at_k", "f1_at_k", "ndcg_at_k"], arr.mean(axis=0)))})
    rows.append({"baseline": "Random (SD across seeds)", "seed": "", **dict(zip(["precision_at_k", "recall_at_k", "f1_at_k", "ndcg_at_k"], arr.std(axis=0, ddof=1)))})
    # analytical expected random precision / recall
    ep, er = [], []
    for u in test_users:
        cand = set(all_uids) - rated.get(u, set())
        hits_pool = len(truth[u] & cand)
        ep.append(hits_pool / len(cand))
        er.append(min(K, len(cand)) * hits_pool / len(cand) / len(truth[u]))
    rows.append({"baseline": "Random (analytical expectation)", "seed": "", "precision_at_k": np.mean(ep), "recall_at_k": np.mean(er), "f1_at_k": np.nan, "ndcg_at_k": np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "table3_baselines.csv", index=False)
    print(out[~out["baseline"].eq("Random")].to_string(index=False))


def part_ties():
    attraction_df, train_df, test_df, test_users = setup()
    cbf_df = build_content_column(attraction_df)
    _, tfidf, aidx = build_tfidf_matrix(cbf_df)
    uim, uidx, cf_aidx = build_user_item_matrix(train_df)
    S = sim_rows(uim, [uidx[u] for u in test_users if u in uidx])
    rated = train_df.groupby("tourist_id")["attraction_uid"].apply(set).to_dict()
    truth = {u: extract_ground_truth(u, test_df) for u in test_users}
    uids = attraction_df["attraction_uid"]

    def analyse(res, col):
        """res: DataFrame[attraction_uid, col]; returns tie diagnostics + metrics for default vs deterministic order."""
        d = res.sort_values(by=col, ascending=False)  # exactly what src does
        top_def = list(d["attraction_uid"].head(K))
        det = res.sort_values(by=[col, "attraction_uid"], ascending=[False, True])
        top_det = list(det["attraction_uid"].head(K))
        sc = det[col].values
        boundary_tie = len(sc) > K and sc[K - 1] == sc[K]
        n_tied_at_boundary = int((sc == sc[K - 1]).sum()) if boundary_tie else 0
        any_tie_top10 = len(set(np.round(sc[:K], 12))) < min(K, len(sc))
        t = truth_u
        out = {"boundary_tie": boundary_tie, "n_tied_at_boundary": n_tied_at_boundary, "any_tie_in_top10": any_tie_top10,
               "same_top10_set": set(top_def) == set(top_det), "same_top10_order": top_def == top_det}
        for tag, rec in [("default", top_def), ("deterministic", top_det)]:
            p = precision_at_k(rec, t); r = recall_at_k(rec, t)
            out[f"p_{tag}"] = p; out[f"r_{tag}"] = r; out[f"f1_{tag}"] = f1_at_k(p, r); out[f"ndcg_{tag}"] = ndcg_at_k(rec, t, K)
        return out

    rows = []
    for i, u in enumerate(test_users):
        truth_u = truth[u]
        vis = rated.get(u, set())
        # CBF scores via the src functions
        prof = build_user_profile(u, train_df, tfidf, aidx)
        if prof is not None:
            sim = compute_similarity(prof, tfidf)
            res = pd.DataFrame({"attraction_uid": uids, "similarity_score": sim})
            res = res[~res["attraction_uid"].isin(vis)]
            rows.append({"model": "CBF", "tourist_id": u, **analyse(res, "similarity_score")})
        # UBCF scores via the src functions
        nb = find_nearest_neighbors(u, S, uidx, k=20)
        pred = predict_ratings(nb, uim, uidx, cf_aidx) if nb else {}
        if pred:
            res = pd.DataFrame(list(pred.items()), columns=["attraction_uid", "predicted_rating"])
            res = res[~res["attraction_uid"].isin(vis)]
            if not res.empty:
                rows.append({"model": "UBCF", "tourist_id": u, **analyse(res, "predicted_rating")})
        if (i + 1) % 2000 == 0:
            print(f"{i+1}/{len(test_users)}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(RES / "tie_analysis_per_user.csv", index=False)
    summ = []
    for m, g in d.groupby("model"):
        summ.append({"model": m, "n_users": len(g),
                     "pct_boundary_tie(rank10==rank11)": 100 * g["boundary_tie"].mean(),
                     "pct_any_tie_in_top10": 100 * g["any_tie_in_top10"].mean(),
                     "pct_top10_set_changes_under_deterministic_tiebreak": 100 * (~g["same_top10_set"]).mean(),
                     "pct_top10_order_changes": 100 * (~g["same_top10_order"]).mean(),
                     "P@10_default": g["p_default"].mean(), "P@10_deterministic": g["p_deterministic"].mean(),
                     "R@10_default": g["r_default"].mean(), "R@10_deterministic": g["r_deterministic"].mean(),
                     "F1@10_default": g["f1_default"].mean(), "F1@10_deterministic": g["f1_deterministic"].mean(),
                     "nDCG@10_default": g["ndcg_default"].mean(), "nDCG@10_deterministic": g["ndcg_deterministic"].mean()})
    s = pd.DataFrame(summ)
    s.to_csv(RES / "tie_analysis_summary.csv", index=False)
    print(s.T.to_string())


def part_baselines_ci():
    """Per-user mean, SD and 95% CI for the Popularity and Random baselines, so they
    can sit in Table 3 on the same basis as CBF/UBCF (SD = across-user SD).
    Popularity is deterministic. Random is run for 20 seeds; for each seed the
    per-user mean/SD/CI are computed over the 9,744 users, then averaged over seeds
    (the across-seed SD of the mean is reported separately)."""
    attraction_df, train_df, test_df, test_users = setup()
    all_uids = np.array(sorted(attraction_df["attraction_uid"].unique()))
    rated = train_df.groupby("tourist_id")["attraction_uid"].apply(set).to_dict()
    pop = train_df.groupby("attraction_uid").size()
    pop_order = sorted(all_uids, key=lambda u: (-pop.get(u, 0), u))
    truth = {u: extract_ground_truth(u, test_df) for u in test_users}

    def per_user(rec_by_user):
        out = {m: [] for m in METRICS}
        for u in test_users:
            rec = rec_by_user[u]; t = truth[u]
            p = precision_at_k(rec, t); r = recall_at_k(rec, t)
            out["precision_at_k"].append(p); out["recall_at_k"].append(r)
            out["f1_at_k"].append(f1_at_k(p, r)); out["ndcg_at_k"].append(ndcg_at_k(rec, t, K))
        return {m: np.array(v) for m, v in out.items()}

    rows = []
    rec = {u: [x for x in pop_order if x not in rated.get(u, set())][:K] for u in test_users}
    pu = per_user(rec)
    for m in METRICS:
        mean, sd, lo, hi = ci_row(pu[m])
        rows.append({"baseline": "Popularity", "metric": m, "mean": mean, "sd_across_users": sd, "ci95_low": lo, "ci95_high": hi})

    seed_stats = {m: [] for m in METRICS}
    seed_user_vals = {m: [] for m in METRICS}   # per-seed per-user arrays, for the seed-averaged variant
    for seed in range(20):
        rng = np.random.RandomState(seed)
        rec = {}
        for u in test_users:
            cand = [x for x in all_uids if x not in rated.get(u, set())]
            idx = rng.choice(len(cand), size=min(K, len(cand)), replace=False)
            rec[u] = [cand[i] for i in idx]
        pu = per_user(rec)
        for m in METRICS:
            seed_stats[m].append(ci_row(pu[m]))
            seed_user_vals[m].append(pu[m])
    # Stricter variant: average each user's metric over the 20 seeds first, then
    # compute mean / SD / 95% CI across users (CI for expected random performance).
    for m in METRICS:
        avg_user = np.mean(np.vstack(seed_user_vals[m]), axis=0)
        mean, sd, lo, hi = ci_row(avg_user)
        rows.append({"baseline": "Random (user values averaged over 20 seeds)", "metric": m, "mean": mean,
                     "sd_across_users": sd, "ci95_low": lo, "ci95_high": hi})
    for m in METRICS:
        a = np.array(seed_stats[m])  # columns: mean, sd, lo, hi
        rows.append({"baseline": "Random (avg over 20 seeds)", "metric": m, "mean": a[:, 0].mean(),
                     "sd_across_users": a[:, 1].mean(), "ci95_low": a[:, 2].mean(), "ci95_high": a[:, 3].mean(),
                     "sd_of_mean_across_seeds": a[:, 0].std(ddof=1)})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "table3_baselines_sd_ci.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    t0 = time.time()
    {"stats": part_stats, "baselines": part_baselines, "ties": part_ties, "baselines_ci": part_baselines_ci}[sys.argv[1]]()
    print(f"Elapsed {time.time()-t0:.1f}s")
