"""
Streamlit Prototype Application for Travel Destination Recommendation System.

This application provides an interactive presentation layer for Content-Based Filtering,
User-Based Collaborative Filtering, and model comparison for academic demonstration.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

from src.preprocessing import (
    load_dataset,
    prepare_attractions,
    prepare_interactions,
    train_test_split_by_user,
)
from src.content_based import (
    build_content_column,
    build_tfidf_matrix,
    recommend_attractions,
)
from src.collaborative import (
    build_user_item_matrix,
    build_user_similarity_matrix,
    recommend_attractions_cf,
)


@st.cache_data
def load_data():
    """
    Loads raw data, prepares attraction and interaction features, and splits
    interactions into train and test sets using user-stratified splitting.
    """
    project_dir = Path(__file__).resolve().parent
    csv_path = project_dir / "data" / "tourism_recommendation_dataset_en.csv"

    raw_df = load_dataset(str(csv_path))
    attraction_df = prepare_attractions(raw_df)
    interactions_df = prepare_interactions(raw_df)
    train_df, test_df = train_test_split_by_user(
        interactions_df, test_ratio=0.2, min_interactions=5, random_state=42
    )

    return attraction_df, interactions_df, train_df, test_df


@st.cache_resource
def get_cbf_context(attraction_df: pd.DataFrame):
    """
    Precomputes the TF-IDF feature matrix and attraction index mapping
    for Content-Based Filtering once per application session.
    """
    content_df = build_content_column(attraction_df)
    _, tfidf_matrix, attraction_index = build_tfidf_matrix(content_df)
    return {
        "attraction_df": attraction_df,
        "tfidf_matrix": tfidf_matrix,
        "attraction_index": attraction_index,
    }


@st.cache_resource
def get_cf_context(train_df: pd.DataFrame, attraction_df: pd.DataFrame):
    """
    Precomputes the user-item ratings matrix, user similarity matrix,
    and index mappings for User-Based Collaborative Filtering once per application session.
    """
    user_item_matrix, user_index, attraction_index = build_user_item_matrix(train_df)
    user_similarity_matrix = build_user_similarity_matrix(user_item_matrix)
    return {
        "attraction_df": attraction_df,
        "user_item_matrix": user_item_matrix,
        "user_similarity_matrix": user_similarity_matrix,
        "user_index": user_index,
        "attraction_index": attraction_index,
    }


def get_recommendations(
    tourist_id: int,
    algorithm: str,
    top_n: int,
    train_df: pd.DataFrame,
    cbf_context: dict,
    cf_context: dict,
) -> pd.DataFrame:
    """
    Routes recommendation requests to the selected algorithm's existing public function.
    """
    if algorithm == "Content-Based Filtering":
        return recommend_attractions(
            tourist_id=tourist_id,
            interactions_df=train_df,
            attraction_df=cbf_context["attraction_df"],
            tfidf_matrix=cbf_context["tfidf_matrix"],
            attraction_index=cbf_context["attraction_index"],
            top_n=top_n,
        )
    elif algorithm == "User-Based Collaborative Filtering":
        return recommend_attractions_cf(
            tourist_id=tourist_id,
            train_df=train_df,
            attraction_df=cf_context["attraction_df"],
            user_item_matrix=cf_context["user_item_matrix"],
            user_similarity_matrix=cf_context["user_similarity_matrix"],
            user_index=cf_context["user_index"],
            attraction_index=cf_context["attraction_index"],
            k=20,
            top_n=top_n,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def format_recommendations_table(df: pd.DataFrame, model_type: str) -> pd.DataFrame:
    """
    Formats the recommendation dataframe for clean display with Rank as the first column.
    """
    if df.empty:
        return df

    if model_type == "CBF":
        display_df = df[
            [
                "rank",
                "attraction_name",
                "attraction_category",
                "city",
                "similarity_score",
                "attraction_uid",
            ]
        ].copy()
        display_df.columns = [
            "Rank",
            "Attraction Name",
            "Category",
            "City",
            "Similarity Score",
            "Attraction ID",
        ]
        return display_df
    else:
        display_df = df[
            [
                "rank",
                "attraction_name",
                "attraction_category",
                "city",
                "predicted_rating",
                "attraction_uid",
            ]
        ].copy()
        display_df.columns = [
            "Rank",
            "Attraction Name",
            "Category",
            "City",
            "Predicted Rating",
            "Attraction ID",
        ]
        return display_df


def render_performance_section():
    """
    Renders a compact read-only table showing Phase 5 offline evaluation results.
    """
    st.subheader("Overall Model Performance")
    perf_data = {
        "Model": [
            "Content-Based Filtering",
            "User-Based Collaborative Filtering",
        ],
        "Technique": [
            "TF-IDF + Cosine Similarity",
            "KNN (K=20) + Cosine Similarity",
        ],
        "Precision@10": [0.0049, 0.0046],
        "Recall@10": [0.0237, 0.0227],
        "F1-Score@10": [0.0080, 0.0075],
        "Coverage": ["100.0%", "100.0%"],
    }
    perf_df = pd.DataFrame(perf_data)
    st.dataframe(
        perf_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Precision@10": st.column_config.NumberColumn(format="%.4f"),
            "Recall@10": st.column_config.NumberColumn(format="%.4f"),
            "F1-Score@10": st.column_config.NumberColumn(format="%.4f"),
        },
    )


def render_single_model_results(recommendations: pd.DataFrame, tourist_id: int, algorithm: str):
    """
    Renders recommendations table and explanation for a single model.
    """
    st.subheader(algorithm)

    if recommendations.empty:
        if algorithm == "Content-Based Filtering":
            st.warning(
                f"No recommendations available for Tourist ID {tourist_id} using Content-Based Filtering. "
                "This user has no attractions rated >= 4.0 in the training data (Cold-Start)."
            )
        else:
            st.warning(
                f"No recommendations available for Tourist ID {tourist_id} using User-Based Collaborative Filtering. "
                "This user has no similar neighbors or valid rating predictions (Cold-Start)."
            )
        return

    if algorithm == "Content-Based Filtering":
        st.markdown("CBF recommendations are ranked by Similarity Score.")
        display_df = format_recommendations_table(recommendations, "CBF")
        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Rank": st.column_config.NumberColumn(format="%d"),
                "Similarity Score": st.column_config.NumberColumn(format="%.4f"),
            },
        )
    else:
        st.markdown("CF recommendations are ranked by Predicted Rating.")
        display_df = format_recommendations_table(recommendations, "CF")
        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Rank": st.column_config.NumberColumn(format="%d"),
                "Predicted Rating": st.column_config.NumberColumn(format="%.2f"),
            },
        )


def render_comparison_analysis(cbf_recs: pd.DataFrame, cf_recs: pd.DataFrame):
    """
    Renders comparison summary and insights between CBF and CF models.
    """
    st.subheader("Recommendation Analysis")

    if cbf_recs.empty or cf_recs.empty:
        st.write(
            "Comparison analysis is limited because one or both models returned no recommendations for this tourist profile."
        )
        return

    cbf_uids = set(cbf_recs["attraction_uid"])
    cf_uids = set(cf_recs["attraction_uid"])

    common_uids = cbf_uids.intersection(cf_uids)
    unique_cbf = cbf_uids - cf_uids
    unique_cf = cf_uids - cbf_uids

    st.markdown(
        f"""
• Common recommendations: {len(common_uids)}

• Unique to CBF: {len(unique_cbf)}

• Unique to CF: {len(unique_cf)}
        """
    )
    st.write(
        "Content-Based Filtering recommends attractions based on destination feature similarity, "
        "whereas Collaborative Filtering recommends attractions based on rating patterns of similar users."
    )


def main():
    st.set_page_config(
        page_title="Travel Destination Recommendation System",
        layout="wide",
    )

    st.title("Travel Destination Recommendation System")
    st.write(
        "Interactive academic prototype comparing Content-Based Filtering (TF-IDF + Cosine Similarity) "
        "and User-Based Collaborative Filtering (KNN + Cosine Similarity)."
    )

    render_performance_section()

    # Load dataset & prepare contexts
    with st.spinner("Loading dataset and initializing models..."):
        attraction_df, interactions_df, train_df, test_df = load_data()
        cbf_context = get_cbf_context(attraction_df)
        cf_context = get_cf_context(train_df, attraction_df)

    tourist_ids = sorted(interactions_df["tourist_id"].unique())

    # Sidebar Configuration
    st.sidebar.header("Recommendation Settings")

    selected_tourist_id = st.sidebar.selectbox(
        "Tourist ID",
        options=tourist_ids,
        index=0,
    )

    selected_algorithm = st.sidebar.radio(
        "Recommendation Model",
        options=[
            "Content-Based Filtering",
            "User-Based Collaborative Filtering",
            "Compare Both Models",
        ],
    )

    top_n = st.sidebar.slider(
        "Top N",
        min_value=1,
        max_value=20,
        value=10,
    )

    generate_btn = st.sidebar.button("Generate Recommendation", type="primary")

    if generate_btn:
        with st.spinner("Generating recommendations..."):
            if selected_algorithm == "Compare Both Models":
                cbf_recs = get_recommendations(
                    tourist_id=selected_tourist_id,
                    algorithm="Content-Based Filtering",
                    top_n=top_n,
                    train_df=train_df,
                    cbf_context=cbf_context,
                    cf_context=cf_context,
                )
                cf_recs = get_recommendations(
                    tourist_id=selected_tourist_id,
                    algorithm="User-Based Collaborative Filtering",
                    top_n=top_n,
                    train_df=train_df,
                    cbf_context=cbf_context,
                    cf_context=cf_context,
                )
                st.session_state["cbf_recs"] = cbf_recs
                st.session_state["cf_recs"] = cf_recs
            else:
                recs = get_recommendations(
                    tourist_id=selected_tourist_id,
                    algorithm=selected_algorithm,
                    top_n=top_n,
                    train_df=train_df,
                    cbf_context=cbf_context,
                    cf_context=cf_context,
                )
                st.session_state["recommendations"] = recs

            st.session_state["tourist_id"] = selected_tourist_id
            st.session_state["algorithm"] = selected_algorithm
            st.session_state["top_n"] = top_n

    # Main Area Output
    if "algorithm" in st.session_state and st.session_state["tourist_id"] == selected_tourist_id:
        curr_algo = st.session_state["algorithm"]
        if curr_algo == "Compare Both Models" and "cbf_recs" in st.session_state and "cf_recs" in st.session_state:
            st.success(f"Generated comparison recommendations for Tourist ID {st.session_state['tourist_id']}.")
            render_single_model_results(
                recommendations=st.session_state["cbf_recs"],
                tourist_id=st.session_state["tourist_id"],
                algorithm="Content-Based Filtering",
            )
            render_single_model_results(
                recommendations=st.session_state["cf_recs"],
                tourist_id=st.session_state["tourist_id"],
                algorithm="User-Based Collaborative Filtering",
            )
            render_comparison_analysis(
                cbf_recs=st.session_state["cbf_recs"],
                cf_recs=st.session_state["cf_recs"],
            )
        elif curr_algo != "Compare Both Models" and "recommendations" in st.session_state:
            st.success(f"Generated recommendations for Tourist ID {st.session_state['tourist_id']} using {curr_algo}.")
            render_single_model_results(
                recommendations=st.session_state["recommendations"],
                tourist_id=st.session_state["tourist_id"],
                algorithm=curr_algo,
            )
    else:
        st.info("Select a Tourist ID and Recommendation Model in the sidebar, then click Generate Recommendation.")


if __name__ == "__main__":
    main()