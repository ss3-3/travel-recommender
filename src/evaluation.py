"""
Recommendation System Evaluation Module.

This module implements offline evaluation metrics (Precision@K, Recall@K, F1@K,
and Coverage) and evaluation orchestrators to compare Content-Based Filtering
and Collaborative Filtering models fairly over held-out test data.
"""

from typing import Callable, Dict, List, Set, Tuple, Union
import pandas as pd


def precision_at_k(recommended_attractions: list, actual_attractions: set) -> float:
    """
    Computes the Precision@K metric for a single user's recommendations.

    Precision@K is the proportion of recommended attractions that appear in the
    user's actual held-out test interactions. Bounded to [0, 1].

    Args:
        recommended_attractions (list): List of recommended attraction_uids.
        actual_attractions (set): Set of true attraction_uids in test_df.

    Returns:
        float: Precision value. Returns 0.0 if the recommendation list is empty.
    """
    if not recommended_attractions:
        return 0.0

    recommended_set = set(recommended_attractions)
    hits = recommended_set.intersection(actual_attractions)

    return len(hits) / len(recommended_attractions)


def recall_at_k(recommended_attractions: list, actual_attractions: set) -> float:
    """
    Computes the Recall@K metric for a single user's recommendations.

    Recall@K is the proportion of actual held-out test interactions that were
    successfully captured within the recommended attractions. Bounded to [0, 1].

    Args:
        recommended_attractions (list): List of recommended attraction_uids.
        actual_attractions (set): Set of true attraction_uids in test_df.

    Returns:
        float: Recall value. Returns 0.0 if recommended_attractions or actual_attractions is empty.
    """
    if not recommended_attractions or not actual_attractions:
        return 0.0

    recommended_set = set(recommended_attractions)
    hits = recommended_set.intersection(actual_attractions)

    return len(hits) / len(actual_attractions)


def f1_at_k(precision: float, recall: float) -> float:
    """
    Computes the F1-score@K metric (harmonic mean of Precision@K and Recall@K).

    Args:
        precision (float): Precision@K score.
        recall (float): Recall@K score.

    Returns:
        float: F1-score value. Returns 0.0 if both precision and recall are 0.0.
    """
    if precision + recall == 0.0:
        return 0.0

    return 2.0 * precision * recall / (precision + recall)


def coverage(evaluated_user_count: int, total_test_user_count: int) -> float:
    """
    Computes the Coverage metric (proportion of test users successfully served).

    Args:
        evaluated_user_count (int): Number of test users with a non-empty recommendation.
        total_test_user_count (int): Total number of test users.

    Returns:
        float: Coverage ratio. Returns 0.0 if total_test_user_count is 0.
    """
    if total_test_user_count == 0:
        return 0.0

    return evaluated_user_count / total_test_user_count

def extract_ground_truth(tourist_id: int, test_df: pd.DataFrame) -> set:
    """
    Isolates the ground-truth attraction_uids for a target user in test_df.

    Args:
        tourist_id (int): Target user ID.
        test_df (pd.DataFrame): Held-out test interactions DataFrame.

    Returns:
        set: Set of attraction_uids the user interacted with.
    """
    user_test = test_df[test_df["tourist_id"] == tourist_id]
    return set(user_test["attraction_uid"])


def evaluate_model(
    recommend_fn: Callable,
    test_users: List[int],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_context: dict,
    top_n: int = 10,
    model_kwargs: Union[dict, None] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates a recommendation algorithm across a list of test users.

    Computes Precision@K, Recall@K, F1@K, and Coverage. Accuracy metrics are
    averaged only over the users successfully served (non-empty recommendations).

    Args:
        recommend_fn (Callable): Recommender function (e.g. recommend_attractions).
        test_users (List[int]): List of user IDs to evaluate.
        train_df (pd.DataFrame): Training interactions DataFrame.
        test_df (pd.DataFrame): Held-out test interactions DataFrame.
        model_context (dict): Supporting objects (matrices, indexes, dataframes).
        top_n (int): Top-N value K for recommendations.
        model_kwargs (dict): Hyperparameters for the recommend function.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - summary_df: A DataFrame containing the aggregated metrics for the model.
            - per_user_df: A DataFrame containing metrics for each user.
    """
    import inspect

    per_user_data = []

    # Check recommend_fn signature to map train_df to the correct argument name
    sig = inspect.signature(recommend_fn)
    train_arg_name = "train_df"
    if "interactions_df" in sig.parameters:
        train_arg_name = "interactions_df"

    model_kwargs = model_kwargs or {}

    for tourist_id in test_users:
        # Prepare arguments dynamically
        kwargs = {
            "tourist_id": tourist_id,
            "top_n": top_n,
            train_arg_name: train_df,
        }
        kwargs.update(model_context)
        kwargs.update(model_kwargs)

        # Filter kwargs to only pass parameters accepted by recommend_fn
        filtered_kwargs = {
            k: v for k, v in kwargs.items()
            if k in sig.parameters
        }

        # Generate recommendations DataFrame
        recs_df = recommend_fn(**filtered_kwargs)

        recommended_list = list(recs_df["attraction_uid"]) if not recs_df.empty else []
        actual_set = extract_ground_truth(tourist_id, test_df)

        # Calculate single-user metrics
        p = precision_at_k(recommended_list, actual_set)
        r = recall_at_k(recommended_list, actual_set)
        f1 = f1_at_k(p, r)
        hit_count = len(set(recommended_list).intersection(actual_set))

        per_user_data.append({
            "tourist_id": tourist_id,
            "precision_at_k": p,
            "recall_at_k": r,
            "f1_at_k": f1,
            "hit_count": hit_count,
            "recommended_count": len(recommended_list),
            "test_count": len(actual_set)
        })

    per_user_df = pd.DataFrame(per_user_data)

    # Separate users who got a recommendation from those who didn't (cold start)
    covered_users_df = per_user_df[per_user_df["recommended_count"] > 0]

    evaluated_user_count = len(covered_users_df)
    total_test_user_count = len(test_users)

    avg_precision = covered_users_df["precision_at_k"].mean() if evaluated_user_count > 0 else 0.0
    avg_recall = covered_users_df["recall_at_k"].mean() if evaluated_user_count > 0 else 0.0
    avg_f1 = covered_users_df["f1_at_k"].mean() if evaluated_user_count > 0 else 0.0

    cov = coverage(evaluated_user_count, total_test_user_count)

    # Determine the model name
    if recommend_fn.__name__ == "recommend_attractions_cf":
        model_name = "Collaborative Filtering"
    elif recommend_fn.__name__ == "recommend_attractions":
        model_name = "Content-Based Filtering"
    else:
        model_name = recommend_fn.__name__

    summary_data = {
        "model_name": model_name,
        "precision_at_k": avg_precision,
        "recall_at_k": avg_recall,
        "f1_at_k": avg_f1,
        "coverage": cov,
        "evaluated_user_count": evaluated_user_count,
        "coverage_user_count": evaluated_user_count,
        "total_test_user_count": total_test_user_count
    }
    summary_df = pd.DataFrame([summary_data])

    return summary_df, per_user_df


def compare_models(cbf_summary: pd.DataFrame, cf_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenates the summary DataFrames of two models side-by-side for comparison.

    Args:
        cbf_summary (pd.DataFrame): Aggregated metrics for Content-Based Filtering.
        cf_summary (pd.DataFrame): Aggregated metrics for Collaborative Filtering.

    Returns:
        pd.DataFrame: Comparison table with one row per model.
    """
    return pd.concat([cbf_summary, cf_summary], ignore_index=True)


def run_full_evaluation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_users: List[int],
    cbf_context: dict,
    cf_context: dict,
    top_n: int = 10,
    cbf_kwargs: Union[dict, None] = None,
    cf_kwargs: Union[dict, None] = None,
) -> pd.DataFrame:
    """
    Runs evaluation for both Content-Based and Collaborative Filtering,
    returning a side-by-side comparison table.

    Args:
        train_df (pd.DataFrame): Training interactions DataFrame.
        test_df (pd.DataFrame): Held-out test interactions DataFrame.
        test_users (List[int]): User IDs to evaluate on.
        cbf_context (dict): Supporting objects for CBF (attraction_df, tfidf_matrix, etc.).
        cf_context (dict): Supporting objects for CF (user_item_matrix, user_similarity_matrix, etc.).
        top_n (int): K for evaluation.
        cbf_kwargs (dict): Hyperparameters for CBF.
        cf_kwargs (dict): Hyperparameters for CF.

    Returns:
        pd.DataFrame: Combined comparison table.
    """
    from src.content_based import recommend_attractions
    from src.collaborative import recommend_attractions_cf

    cbf_summary, _ = evaluate_model(
        recommend_fn=recommend_attractions,
        test_users=test_users,
        train_df=train_df,
        test_df=test_df,
        model_context=cbf_context,
        top_n=top_n,
        model_kwargs=cbf_kwargs
    )

    cf_summary, _ = evaluate_model(
        recommend_fn=recommend_attractions_cf,
        test_users=test_users,
        train_df=train_df,
        test_df=test_df,
        model_context=cf_context,
        top_n=top_n,
        model_kwargs=cf_kwargs
    )

    return compare_models(cbf_summary, cf_summary)


__all__ = [
    "precision_at_k",
    "recall_at_k",
    "f1_at_k",
    "coverage",
    "extract_ground_truth",
    "evaluate_model",
    "compare_models",
    "run_full_evaluation",
]
