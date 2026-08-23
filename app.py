"""
Streamlit Prototype Application for Travel Destination Recommendation System.

This application provides an interactive presentation layer for Content-Based Filtering,
User-Based Collaborative Filtering, and model comparison for academic demonstration.
"""

import textwrap
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
from src.itinerary import (
    build_one_day_itinerary,
    load_coordinates,
)
from src.itinerary_evaluation import evaluate_itinerary


@st.cache_data
def load_coordinates_df() -> pd.DataFrame:
    """
    Loads coordinates.csv file once and caches it.
    """
    project_dir = Path(__file__).resolve().parent
    coords_path = project_dir / "data" / "coordinates.csv"
    return load_coordinates(str(coords_path))


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


def inject_custom_css():
    """
    Injects custom styles for a refined travel-inspired UI, featuring rounded cards,
    a unified primary accent color, clean typography, and visual spacing.
    """
    st.markdown(
        textwrap.dedent(
            """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;700&display=swap');
            
            :root {
                --primary: #1D4ED8;
                --slate: #64748B;
                --background: #F8F9FB;
                --border: #E5E7EB;
            }
            
            html, body, [class*="css"] {
                font-family: 'Outfit', sans-serif;
            }
            
            .stButton>button {
                border-radius: 8px;
                font-weight: 600;
            }
            
            .stat-card {
                background-color: rgba(29, 78, 216, 0.04);
                border: 1px solid rgba(29, 78, 216, 0.12);
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.01);
            }
            
            .destination-card {
                background-color: #ffffff;
                border: 1px solid #e9ecef;
                border-radius: 10px;
                padding: 18px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.02);
                transition: transform 0.2s, box-shadow 0.2s;
            }
            .destination-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(0,0,0,0.05);
            }
            </style>
            """
        ),
        unsafe_allow_html=True
    )


def render_overview_page(attraction_df: pd.DataFrame, interactions_df: pd.DataFrame):
    """
    Renders the Home / Overview page with project header, visual workflow flowchart,
    and descriptive stats.
    """
    st.markdown("### 🏠 Overview")
    st.write(
        "Welcome to the **Travel Destination Recommendation System**. This system is an interactive academic prototype "
        "designed to provide personalized tourist attraction recommendations and generate one-day multi-destination travel itineraries."
    )
    
    st.markdown("#### System Workflow")
    st.markdown(
        textwrap.dedent(
            """
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin: 20px 0; gap: 10px;">
                <div style="background-color: #f1f8e9; border: 1px solid #dcedc8; border-radius: 8px; padding: 15px; flex: 1; text-align: center; min-width: 150px;">
                    <h5 style="margin: 0 0 5px 0; color: #33691e; font-weight: 700;">1. Recommendation</h5>
                    <p style="margin: 0; font-size: 12px; color: #558b2f;">Predict user preferences via CBF or UBCF models</p>
                </div>
                <div style="font-size: 24px; color: #888888; font-weight: bold;">→</div>
                <div style="background-color: #e8f5e9; border: 1px solid #c8e6c9; border-radius: 8px; padding: 15px; flex: 1; text-align: center; min-width: 150px;">
                    <h5 style="margin: 0 0 5px 0; color: #1b5e20; font-weight: 700;">2. Personalisation</h5>
                    <p style="margin: 0; font-size: 12px; color: #2e7d32;">Extract Top-N destination candidate sets</p>
                </div>
                <div style="font-size: 24px; color: #888888; font-weight: bold;">→</div>
                <div style="background-color: #e8f8f5; border: 1px solid #a3e4d7; border-radius: 8px; padding: 15px; flex: 1; text-align: center; min-width: 150px;">
                    <h5 style="margin: 0 0 5px 0; color: #0e6251; font-weight: 700;">3. Itinerary</h5>
                    <p style="margin: 0; font-size: 12px; color: #117864;">Geographic selection and one-day route sequencing</p>
                </div>
                <div style="font-size: 24px; color: #888888; font-weight: bold;">→</div>
                <div style="background-color: #eaf2f8; border: 1px solid #a9cce3; border-radius: 8px; padding: 15px; flex: 1; text-align: center; min-width: 150px;">
                    <h5 style="margin: 0 0 5px 0; color: #1b4f72; font-weight: 700;">4. Evaluation</h5>
                    <p style="margin: 0; font-size: 12px; color: #2471a3;">Analyze spatial and routing properties</p>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True
    )
    
    st.markdown("#### Dataset Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            textwrap.dedent(
                f"""
                <div class="stat-card">
                    <span style="font-size: 13px; color: #64748B; font-weight: bold; text-transform: uppercase;">Total Interactions Logged</span><br/>
                    <span style="font-size: 30px; font-weight: 700; color: #1D4ED8;">{len(interactions_df):,}</span>
                </div>
                """
            ),
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            textwrap.dedent(
                f"""
                <div class="stat-card">
                    <span style="font-size: 13px; color: #64748B; font-weight: bold; text-transform: uppercase;">Unique Registered Users</span><br/>
                    <span style="font-size: 30px; font-weight: 700; color: #1D4ED8;">{interactions_df['tourist_id'].nunique():,}</span>
                </div>
                """
            ),
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            textwrap.dedent(
                f"""
                <div class="stat-card">
                    <span style="font-size: 13px; color: #64748B; font-weight: bold; text-transform: uppercase;">Unique Verified Attractions</span><br/>
                    <span style="font-size: 30px; font-weight: 700; color: #1D4ED8;">{attraction_df['attraction_uid'].nunique():,}</span>
                </div>
                """
            ),
            unsafe_allow_html=True
        )


def render_performance_section():
    """
    Renders a compact read-only table showing Phase 5 offline evaluation results.
    """
    st.markdown("##### Overall Offline Model Performance")
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


def render_recommendation_cards(df: pd.DataFrame, model_type: str):
    """
    Renders recommendations as clean cards in a three-column grid.
    """
    if df.empty:
        st.warning("No recommendations returned.")
        return

    st.markdown("#### Mapped Attractions")

    cols = st.columns(3)
    for i, (_, row) in enumerate(df.iterrows()):
        rank = row["rank"]
        name = row["attraction_name"]
        category = row["attraction_category"]
        city = row["city"]

        if model_type == "CBF":
            score_label = "Similarity"
            score_val = f"{row['similarity_score']:.4f}"
        else:
            score_label = "Pred. Rating"
            score_val = f"{row['predicted_rating']:.2f}"

        # Assign deep blue border-top for Rank 1-3, neutral border for lower ranks
        border_style = "border-top: 4px solid #1D4ED8;" if rank <= 3 else "border: 1px solid #E5E7EB;"

        col_idx = i % 3
        with cols[col_idx]:
            card_html = f"""
            <div class="destination-card" style="{border_style} border-radius: 10px; padding: 18px; margin-bottom: 15px; background-color: #FFFFFF; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                <div style="font-size: 13px; font-weight: bold; color: #1D4ED8; margin-bottom: 5px;">#{rank}</div>
                <h4 style="margin: 5px 0 10px 0; color: #1E293B; font-size: 15px; font-weight: 700; min-height: 40px; display: flex; align-items: center; line-height: 1.3;">{name}</h4>
                <div style="font-size: 13px; color: #64748B; margin-bottom: 5px;">📍 {city}</div>
                <div style="font-size: 12px; color: #64748B; font-style: italic; margin-bottom: 15px;">Category: {category}</div>
                <div style="border-top: 1px solid #F1F5F9; padding-top: 10px; display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 12px; color: #64748B;">{score_label}</span>
                    <span style="font-size: 13px; font-weight: bold; color: #1E293B;">{score_val}</span>
                </div>
            </div>
            """
            st.markdown(textwrap.dedent(card_html), unsafe_allow_html=True)


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
        render_recommendation_cards(recommendations, "CBF")
        with st.expander("📋 View Raw CBF Recommendation Table"):
            display_df = format_recommendations_table(recommendations, "CBF")
            st.dataframe(display_df, hide_index=True, use_container_width=True)
    else:
        render_recommendation_cards(recommendations, "CF")
        with st.expander("📋 View Raw CF Recommendation Table"):
            display_df = format_recommendations_table(recommendations, "CF")
            st.dataframe(display_df, hide_index=True, use_container_width=True)


def render_comparison_analysis(cbf_recs: pd.DataFrame, cf_recs: pd.DataFrame):
    """
    Renders comparison summary and insights between CBF and CF models.
    """
    st.subheader("Recommendation Comparison")

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
*   **Common recommendations**: `{len(common_uids)}`
*   **Unique to CBF**: `{len(unique_cbf)}`
*   **Unique to CF**: `{len(unique_cf)}`
        """
    )
    st.write(
        "Content-Based Filtering recommends attractions based on destination feature similarity, "
        "whereas Collaborative Filtering recommends attractions based on rating patterns of similar users."
    )

def render_itinerary_ui(
    itinerary_df: pd.DataFrame,
    title: str,
    num_stops: int,
    top_n_candidates: int,
    excluded_df: "pd.DataFrame | None" = None,
):
    """
    Renders the generated one-day itinerary in a visual vertical timeline.
    """
    st.subheader(title)

    excluded_count = (
        len(excluded_df)
        if excluded_df is not None and not excluded_df.empty
        else 0
    )

    st.info(
        f"📋 **Top-N Recommendations**: {top_n_candidates} candidates | "
        f"📍 **Requested Destinations (M)**: {num_stops} | "
        f"🚫 **Final Itinerary Attractions**: {len(itinerary_df)} selected | "
        f"🗓️ **Not Selected**: {excluded_count}"
    )

    if itinerary_df.empty:
        st.warning("No itinerary could be generated from the available recommendation candidates.")
    else:
       # Display the one-day itinerary
        itinerary_sorted = itinerary_df.sort_values("stop_order").reset_index(drop=True)

        total_distance = itinerary_sorted["distance_from_prev_km"].fillna(0).sum()

        route_names = " → ".join(
            itinerary_sorted["attraction_name"].astype(str).tolist()
        )

        st.markdown("### 🗺️ One-Day Route")

        st.markdown(
            f"**Route:** {route_names}"
        )

        st.caption(
            f"{len(itinerary_sorted)} destination(s) | "
            f"Total travel distance: {total_distance:.2f} km"
        )

        # Display each destination according to route order
        for index, row in itinerary_sorted.iterrows():
            stop_order = int(row["stop_order"])
            attraction_name = row["attraction_name"]
            city = row["city"]
            recommendation_score = row["recommendation_score"]

            st.markdown(
                f"**Stop {stop_order}: {attraction_name}**"
            )

            st.caption(
                f"📍 {city} | "
                f"Recommendation Score: {recommendation_score:.4f}"
            )

            # Show distance to the next destination
            if index < len(itinerary_sorted) - 1:
                next_distance = itinerary_sorted.iloc[
                    index + 1
                ]["distance_from_prev_km"]

                st.markdown(
                    f"↓ 🚗 {next_distance:.2f} km"
                )
    # Add a CSV download button
    if not itinerary_df.empty:
        csv_data = itinerary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Download {title} as CSV",
            data=csv_data,
            file_name=f"{title.lower().replace(' ', '_')}_tourist_{st.session_state.get('tourist_id')}.csv",
            mime="text/csv",
            key=f"dl_{title.lower().replace(' ', '_')}",
        )

        with st.expander("📋 View Detailed Itinerary Table"):
            st.dataframe(itinerary_df, hide_index=True, use_container_width=True)

    if excluded_df is not None and not excluded_df.empty:
        with st.expander(
            f"🚫 Recommendations Not Selected for This Itinerary ({len(excluded_df)})",
            expanded=False,
        ):
            st.caption(
                "These attractions were recommended but were not included in the current "
                "one-day itinerary because of geographic suitability or lower priority "
                "among compatible options."
            )
            display_cols = [
                col
                for col in [
                    "attraction_name",
                    "city",
                    "rank",
                    "recommendation_score",
                    "exclusion_reason_label",
                ]
                if col in excluded_df.columns
            ]
            st.dataframe(
                excluded_df[display_cols].rename(
                    columns={
                        "attraction_name": "Attraction",
                        "city": "City",
                        "rank": "Rank",
                        "recommendation_score": "Score",
                        "exclusion_reason_label": "Reason",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )


def render_evaluation_details(evaluation: dict, title: str):
    """
    Renders evaluation metrics for the generated one-day itinerary.
    """
    if not evaluation:
        st.warning("No evaluation metrics available.")
        return

    st.subheader(title)

    # Retrieve one-day itinerary evaluation metrics
    carryover_rate = (
        evaluation.get("candidate_carryover_rate", 0.0) * 100
    )
    avg_consecutive_distance = evaluation.get(
        "avg_consecutive_distance",
        0.0,
    )
    total_travel_distance = evaluation.get(
        "total_travel_distance",
        0.0,
    )
    geographic_compactness = evaluation.get(
        "geographic_compactness",
        0.0,
    )

    # Display N/A when distance-based metrics cannot be calculated
    if avg_consecutive_distance is None:
        avg_consecutive_display = "N/A"
    else:
        avg_consecutive_display = (
            f"{avg_consecutive_distance:.2f} km"
        )

    if geographic_compactness is None:
        compactness_display = "N/A"
    else:
        compactness_display = (
            f"{geographic_compactness:.2f} km"
        )

    st.markdown("##### 🗺️ One-Day Itinerary Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Candidate Carryover Rate",
            f"{carryover_rate:.1f}%",
            help=(
                "Percentage of Top-N recommended attractions "
                "included in the final one-day itinerary."
            ),
        )

    with col2:
        st.metric(
            "Avg Consecutive Distance",
            avg_consecutive_display,
            help=(
                "Average distance between consecutive "
                "attractions in the generated route."
            ),
        )

    with col3:
        st.metric(
            "Total Travel Distance",
            f"{total_travel_distance:.2f} km",
            help=(
                "Total Haversine distance travelled between "
                "all consecutive attractions."
            ),
        )

    with col4:
        st.metric(
            "Geographic Compactness",
            compactness_display,
            help=(
                "Average distance of the selected attractions "
                "from their geographic centroid."
            ),
        )

    st.caption(
        "Lower distance and compactness values indicate a more "
        "geographically concentrated itinerary, while a higher "
        "candidate carryover rate indicates that more Top-N "
        "recommendations were retained."
    )


def main():
    st.set_page_config(
        page_title="Travel Destination Recommendation System",
        layout="wide",
    )

    inject_custom_css()

    # Load dataset & prepare contexts
    with st.spinner("Loading dataset and initializing models..."):
        attraction_df, interactions_df, train_df, test_df = load_data()
        cbf_context = get_cbf_context(attraction_df)
        cf_context = get_cf_context(train_df, attraction_df)

    tourist_ids = sorted(interactions_df["tourist_id"].unique())

    # Initialize navigation choice
    if "nav_choice" not in st.session_state:
        st.session_state["nav_choice"] = "🏠 Overview"

    # Sidebar Navigation Menu
    st.sidebar.markdown("### 🗺️ Navigation")
    nav_options = ["🏠 Overview", "🎯 Recommendations", "🗺️ Itinerary", "📊 Evaluation"]
    try:
        nav_index = nav_options.index(st.session_state["nav_choice"])
    except ValueError:
        nav_index = 0

    nav = st.sidebar.radio(
        "Go to:",
        options=nav_options,
        index=nav_index,
        label_visibility="collapsed"
    )
    st.session_state["nav_choice"] = nav

    # Expander in sidebar for About Project (avoids cluttering)
    st.sidebar.markdown("---")
    with st.sidebar.expander("ℹ️ About This Project"):
        st.write(
            "This system compares Content-Based and Collaborative Filtering models (RO1-RO3) "
            "and generates geographically suitable one-day itineraries (RO4-RO5)."
        )

    # Hero / Header Section on Main Panel
    header_html = """
    <div style="background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%); padding: 25px; border-radius: 12px; margin-bottom: 25px; color: white; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
        <h1 style="margin: 0; font-size: 28px; font-weight: 700; color: white;">Travel Destination Recommendation System</h1>
        <p style="margin: 8px 0 0 0; font-size: 14px; opacity: 0.9; font-weight: 400;">
            Personalised destination recommendations and multi-destination itinerary planning
        </p>
    </div>
    """
    st.markdown(textwrap.dedent(header_html), unsafe_allow_html=True)

    # Page Routing
    if st.session_state["nav_choice"] == "🏠 Overview":
        render_overview_page(attraction_df, interactions_df)

    elif st.session_state["nav_choice"] == "🎯 Recommendations":
        st.markdown("### 🎯 Trip Preferences")
        
        # Trip Preferences Search Form directly in the page
        with st.container(border=True):
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                selected_tourist_id = st.selectbox(
                    "Select Tourist ID",
                    options=tourist_ids,
                    index=0 if "tourist_id" not in st.session_state else tourist_ids.index(st.session_state["tourist_id"])
                )
            with col2:
                selected_algorithm = st.radio(
                    "Select Recommendation Model",
                    options=[
                        "Content-Based Filtering",
                        "User-Based Collaborative Filtering",
                        "Compare Both Models"
                    ],
                    index=st.session_state.get("algorithm_index", 0),
                    horizontal=True
                )
                if selected_algorithm is None:
                    selected_algorithm = "Content-Based Filtering"
                # Map selected_algorithm to index for state persistence
                algo_options = ["Content-Based Filtering", "User-Based Collaborative Filtering", "Compare Both Models"]
                st.session_state["algorithm_index"] = algo_options.index(selected_algorithm)
            with col3:
                top_n = st.slider(
                    "Number of Destinations (Top N)",
                    min_value=1,
                    max_value=20,
                    value=st.session_state.get("top_n", 10)
                )

            col_btn, _ = st.columns([1, 3])
            with col_btn:
                generate_btn = st.button("🔍 Search Destinations", type="primary", use_container_width=True)

        if generate_btn:
            # Clear stale itinerary and evaluation states
            for key in [
                "itinerary",
                "itinerary_cbf",
                "itinerary_cf",
                "itinerary_num_stops",
                "evaluation",
                "evaluation_cbf",
                "evaluation_cf",
            ]:
                if key in st.session_state:
                    del st.session_state[key]

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
                st.rerun()

        # Display results if available
        if "algorithm" in st.session_state and st.session_state["tourist_id"] == selected_tourist_id:
            curr_algo = st.session_state["algorithm"]
            if curr_algo == "Compare Both Models" and "cbf_recs" in st.session_state and "cf_recs" in st.session_state:
                st.success(f"Showing comparison recommendations for Tourist ID {st.session_state['tourist_id']}.")
                
                tab1, tab2 = st.tabs(["Content-Based Filtering", "User-Based Collaborative Filtering"])
                with tab1:
                    render_single_model_results(
                        recommendations=st.session_state["cbf_recs"],
                        tourist_id=st.session_state["tourist_id"],
                        algorithm="Content-Based Filtering",
                    )
                with tab2:
                    render_single_model_results(
                        recommendations=st.session_state["cf_recs"],
                        tourist_id=st.session_state["tourist_id"],
                        algorithm="User-Based Collaborative Filtering",
                    )
                st.write("---")
                render_comparison_analysis(
                    cbf_recs=st.session_state["cbf_recs"],
                    cf_recs=st.session_state["cf_recs"],
                )
            elif curr_algo != "Compare Both Models" and "recommendations" in st.session_state:
                st.success(f"Showing recommendations for Tourist ID {st.session_state['tourist_id']} using {curr_algo}.")
                render_single_model_results(
                    recommendations=st.session_state["recommendations"],
                    tourist_id=st.session_state["tourist_id"],
                    algorithm=curr_algo,
                )
        else:
            st.info("💡 Please set your preferences above and click 'Search Destinations' to view recommendations.")

    elif st.session_state["nav_choice"] == "🗺️ Itinerary":
        st.markdown("### 🗺️ One-Day Itinerary Planner")
        
        # Check if recommendations exist in session state
        rec_len = 0
        curr_algo = st.session_state.get("algorithm")
        if curr_algo:
            if curr_algo == "Compare Both Models" and "cbf_recs" in st.session_state and "cf_recs" in st.session_state:
                rec_len = min(len(st.session_state["cbf_recs"]), len(st.session_state["cf_recs"]))
            elif curr_algo != "Compare Both Models" and "recommendations" in st.session_state:
                rec_len = len(st.session_state["recommendations"])

        if rec_len == 0:
            st.info("🎯 Please search destinations and generate recommendations first on the 'Recommendations' tab before building an itinerary.")
        else:
            st.write(
                "Select geographically suitable attractions from your recommendations "
                "and group them into a one-day travel plan."
            )

            num_stops = st.number_input(
                "Number of Destinations for the One-Day Itinerary (M)",
                min_value=1,
                max_value=rec_len,
                value=1,
                step=1,
                help=(
                    "Select how many destinations you want to include in the one-day itinerary. "
                    "The selected number M must be less than or equal to the number of Top-N recommendations."
                ),
            )

            coordinates_df = load_coordinates_df()

            # Input validation check
            if num_stops < 1:
                st.error("Number of itinerary destinations must be at least 1.")
                # Clear stale itinerary and evaluation states
                for key in [
                    "itinerary", "itinerary_cbf", "itinerary_cf",
                    "excluded", "excluded_cbf", "excluded_cf",
                    "evaluation", "evaluation_cbf", "evaluation_cf",
                    "itinerary_num_stops", "itinerary_error",
                ]:
                    if key in st.session_state:
                        del st.session_state[key]
            elif num_stops > rec_len:
                st.error(
                     f"The number of itinerary destinations cannot exceed "
                     f"the Top-N recommendation size ({rec_len})."
                )
                # Clear stale itinerary and evaluation states
                for key in [
                    "itinerary", "itinerary_cbf", "itinerary_cf",
                    "excluded", "excluded_cbf", "excluded_cf",
                    "evaluation", "evaluation_cbf", "evaluation_cf",
                    "itinerary_num_stops", "itinerary_error",
                ]:
                    if key in st.session_state:
                        del st.session_state[key]
            else:
                if "itinerary_error" in st.session_state:
                    st.error(st.session_state["itinerary_error"])

                itinerary_btn = st.button("Generate One-Day Itinerary", type="primary")

                if itinerary_btn or "itinerary_num_stops" in st.session_state:
                    # Reset itinerary if the requested number of destinations has changed
                    if st.session_state.get("itinerary_num_stops") != num_stops:
                        for key in ["itinerary", "itinerary_cbf", "itinerary_cf", "evaluation", "evaluation_cbf", "evaluation_cf", "itinerary_error"]:
                            if key in st.session_state:
                                del st.session_state[key]

                    if itinerary_btn or "itinerary_num_stops" not in st.session_state:
                        with st.spinner("Generating itinerary and calculating metrics..."):
                            try:
                                if curr_algo == "Compare Both Models":
                                    cbf_recs = st.session_state["cbf_recs"]
                                    cf_recs = st.session_state["cf_recs"]
                                    if not cbf_recs.empty:
                                        itinerary_cbf, excluded_cbf = build_one_day_itinerary(
                                            cbf_recs, coordinates_df, num_stops, return_excluded=True
                                        )
                                        st.session_state["itinerary_cbf"] = itinerary_cbf
                                        st.session_state["excluded_cbf"] = excluded_cbf
                                        st.session_state["evaluation_cbf"] = evaluate_itinerary(
                                            itinerary_cbf, cbf_recs
                                        )
                                    if not cf_recs.empty:
                                        itinerary_cf, excluded_cf = build_one_day_itinerary(
                                            cf_recs, coordinates_df, num_stops, return_excluded=True
                                        )
                                        st.session_state["itinerary_cf"] = itinerary_cf
                                        st.session_state["excluded_cf"] = excluded_cf
                                        st.session_state["evaluation_cf"] = evaluate_itinerary(
                                            itinerary_cf, cf_recs
                                        )
                                else:
                                    recs = st.session_state["recommendations"]
                                    if not recs.empty:
                                        itinerary, excluded = build_one_day_itinerary(
                                            recs, coordinates_df, num_stops, return_excluded=True
                                        )
                                        st.session_state["itinerary"] = itinerary
                                        st.session_state["excluded"] = excluded
                                        st.session_state["evaluation"] = evaluate_itinerary(
                                            itinerary, recs
                                        )
                                st.session_state["itinerary_num_stops"] = num_stops
                                if "itinerary_error" in st.session_state:
                                    del st.session_state["itinerary_error"]
                                st.rerun()
                            except ValueError as e:
                                st.session_state["itinerary_error"] = str(e)
                                # Clear stale itinerary and evaluation states to prevent stale/incorrect views
                                for key in ["itinerary", "itinerary_cbf", "itinerary_cf", "evaluation", "evaluation_cbf", "evaluation_cf", "itinerary_num_stops"]:
                                    if key in st.session_state:
                                        del st.session_state[key]
                                st.rerun()

                # Render the generated itineraries
                if "itinerary_num_stops" in st.session_state:
                    saved_stops = st.session_state["itinerary_num_stops"]
                    if curr_algo == "Compare Both Models":
                        tab1, tab2 = st.tabs(["Content-Based Filtering Itinerary", "User-Based Collaborative Filtering Itinerary"])
                        with tab1:
                            if "itinerary_cbf" in st.session_state:
                                render_itinerary_ui(
                                    st.session_state["itinerary_cbf"],
                                    "Content-Based Filtering Itinerary",
                                    saved_stops,
                                    len(st.session_state.get("cbf_recs", pd.DataFrame())),
                                    excluded_df=st.session_state.get("excluded_cbf", pd.DataFrame()),
                                )
                        with tab2:
                            if "itinerary_cf" in st.session_state:
                                render_itinerary_ui(
                                    st.session_state["itinerary_cf"],
                                    "User-Based Collaborative Filtering Itinerary",
                                    saved_stops,
                                    len(st.session_state.get("cf_recs", pd.DataFrame())),
                                    excluded_df=st.session_state.get("excluded_cf", pd.DataFrame()),
                                )
                    else:
                        if "itinerary" in st.session_state:
                            render_itinerary_ui(
                                st.session_state["itinerary"],
                                f"{curr_algo} Itinerary",
                                saved_stops,
                                len(st.session_state.get("recommendations", pd.DataFrame())),
                                excluded_df=st.session_state.get("excluded", pd.DataFrame()),
                            )

    elif st.session_state["nav_choice"] == "📊 Evaluation":
        st.markdown("### 📊 System Evaluation")
        
        # Part A: Recommendation Performance (RO3)
        st.markdown("#### 🎯 Recommendation Performance (RO3)")
        render_performance_section()
        st.write("---")

        # Part B: Itinerary Quality (RO5)
        st.markdown("#### 🗺️ Itinerary Quality (RO5)")
        if "algorithm" in st.session_state:
            curr_algo = st.session_state["algorithm"]
            if curr_algo == "Compare Both Models":
                if "evaluation_cbf" in st.session_state or "evaluation_cf" in st.session_state:
                    tab1, tab2 = st.tabs(["CBF Itinerary Evaluation Details", "UBCF Itinerary Evaluation Details"])
                    with tab1:
                        if "evaluation_cbf" in st.session_state:
                            render_evaluation_details(st.session_state["evaluation_cbf"], "Content-Based Filtering Itinerary")
                    with tab2:
                        if "evaluation_cf" in st.session_state:
                            render_evaluation_details(st.session_state["evaluation_cf"], "User-Based Collaborative Filtering Itinerary")
                else:
                    st.info("🗺️ Please generate an itinerary first to view structural evaluation diagnostics.")
            else:
                if "evaluation" in st.session_state:
                    render_evaluation_details(st.session_state["evaluation"], f"{curr_algo} Itinerary")
                else:
                    st.info("🗺️ Please generate an itinerary first to view structural evaluation diagnostics.")
        else:
            st.info("🎯 Please generate recommendations and build an itinerary first to inspect evaluation diagnostics.")


if __name__ == "__main__":
    main()
