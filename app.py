"""
Streamlit Prototype Application for Travel Destination Recommendation System.

This application provides an interactive presentation layer for Content-Based Filtering,
User-Based Collaborative Filtering, and model comparison for academic demonstration.
"""

import textwrap
from pathlib import Path
from typing import Tuple
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
from src.itinerary import build_itinerary, load_coordinates
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
        "designed to provide personalized tourist attraction recommendations and generate day-by-day travel itineraries."
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
                    <p style="margin: 0; font-size: 12px; color: #117864;">Geographic day clustering and stop sequencing</p>
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


def render_day_itinerary(grouped, day_num: int):
    """
    Renders a single day's vertical timeline stops, or a warning if empty.
    """
    if day_num not in grouped.groups:
        st.info(f"Day {day_num}: No recommended attractions assigned to this day.")
        return

    group = grouped.get_group(day_num)
    group_sorted = group.sort_values("stop_order")

    day_dist = group_sorted["distance_from_prev_km"].sum()
    stops_count = len(group_sorted)
    stop_names = " → ".join(group_sorted["attraction_name"].tolist())

    # Daily Summary Card
    summary_html = f"""
    <div style="background-color: #F8F9FB; border-left: 5px solid #1D4ED8; padding: 15px; border-radius: 8px; margin-bottom: 25px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div>
                <span style="font-size: 13px; font-weight: 700; color: #1D4ED8; text-transform: uppercase; letter-spacing: 0.5px;">DAY {day_num} ROUTE</span>
                <span style="margin-left: 8px; font-size: 13px; color: #64748B;">({stop_names})</span>
            </div>
            <div style="font-size: 13px; color: #1E293B; font-weight: 500;">
                <b>{stops_count}</b> stops &middot; <b>{day_dist:.2f} km</b> total distance
            </div>
        </div>
    </div>
    """
    st.markdown(textwrap.dedent(summary_html), unsafe_allow_html=True)

    # Timeline list
    for idx, (_, row) in enumerate(group_sorted.iterrows()):
        stop = int(row["stop_order"])
        name = row["attraction_name"]
        city = row["city"]
        score = row["recommendation_score"]
        lat = row["latitude"]
        lon = row["longitude"]

        stop_indicator_line = f"<div style='width: 2px; background-color: #E5E7EB; flex-grow: 1; min-height: 25px;'></div>" if idx < len(group_sorted) - 1 else ""

        stop_html = (
            f'<div style="display: flex; margin-bottom: 0px; align-items: stretch;">'
            f'<div style="display: flex; flex-direction: column; align-items: center; margin-right: 15px;">'
            f'<div style="background-color: #1D4ED8; color: white; border-radius: 50%; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 12px; z-index: 2; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">{stop}</div>'
            f'{stop_indicator_line}'
            f'</div>'
            f'<div style="flex-grow: 1; padding-bottom: 15px; padding-top: 2px;">'
            f'<span style="font-weight: 700; font-size: 15px; color: #1E293B;">{name}</span>'
            f'<span style="color: #64748B; font-size: 13px; margin-left: 5px;">📍 {city}</span><br/>'
            f'<span style="font-size: 12px; color: #64748B;">Score: <b>{score:.4f}</b> | Coordinates: ({lat:.4f}, {lon:.4f})</span>'
            f'</div>'
            f'</div>'
        )
        st.markdown(stop_html, unsafe_allow_html=True)

        if idx < len(group_sorted) - 1:
            next_row = group_sorted.iloc[idx + 1]
            next_dist = next_row["distance_from_prev_km"]
            transition_html = (
                f'<div style="display: flex; align-items: center; margin-left: 13px; margin-top: -15px; margin-bottom: 5px;">'
                f'<div style="width: 2px; background-color: #E5E7EB; height: 35px; margin-right: 20px;"></div>'
                f'<div style="font-size: 12px; color: #64748B; font-weight: 500; background-color: #F8F9FB; padding: 3px 8px; border-radius: 4px; border: 1px solid #E5E7EB; display: inline-flex; align-items: center;">'
                f'🚗 {next_dist:.2f} km'
                f'</div>'
                f'</div>'
            )
            st.markdown(transition_html, unsafe_allow_html=True)


def render_itinerary_ui(
    itinerary_df: pd.DataFrame,
    title: str,
    num_days: int,
    top_n_candidates: int,
    excluded_df: "pd.DataFrame | None" = None,
):
    """
    Renders the day-by-day itinerary in a highly visual vertical timeline.
    """
    st.subheader(title)

    excluded_count = len(excluded_df) if excluded_df is not None and not excluded_df.empty else 0
    st.info(
        f"📋 **Recommended Attractions**: {top_n_candidates} candidates | "
        f"📍 **Final Itinerary Attractions**: {len(itinerary_df)} selected | "
        f"🚫 **Not Selected**: {excluded_count} | "
        f"🗓️ **Travel Days**: {num_days}"
    )

    if itinerary_df.empty:
        st.warning("No itinerary could be generated from the available recommendation candidates.")
    else:
        # Group by day
        grouped = itinerary_df.groupby("day")

        # Decide between tabs and selectbox based on num_days
        if num_days <= 5:
            tab_labels = []
            for day in range(1, num_days + 1):
                count = len(grouped.get_group(day)) if day in grouped.groups else 0
                tab_labels.append(f"Day {day} ({count} stops)")
            
            day_tabs = st.tabs(tab_labels)
            for day_idx, day_tab in enumerate(day_tabs):
                day_num = day_idx + 1
                with day_tab:
                    render_day_itinerary(grouped, day_num)
        else:
            options = []
            for day in range(1, num_days + 1):
                count = len(grouped.get_group(day)) if day in grouped.groups else 0
                options.append(f"Day {day} ({count} stops)")
                
            selected_label = st.selectbox("Select Day to View", options=options, key=f"sel_{title.lower().replace(' ', '_')}")
            day_num = int(selected_label.split(" ")[1])
            render_day_itinerary(grouped, day_num)

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
                "itinerary because of geographic feasibility, daily capacity, or lower "
                "priority among compatible options."
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
    Renders diagnostic metrics, day balance summaries, and per-day tables.
    """
    if not evaluation:
        st.warning("No evaluation metrics available.")
        return

    st.markdown("##### 🗺️ Core Spatial Metrics")
    carryover_rate = evaluation.get("candidate_carryover_rate", 0.0) * 100
    avg_consec = evaluation.get("avg_consecutive_distance", {})
    total_dist = evaluation.get("total_distance_per_day", {})
    compactness = evaluation.get("compactness_per_day", {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Candidate Carryover Rate", f"{carryover_rate:.1f}%", help="Proportion of recommended candidates carried over into the final itinerary.")
    with col2:
        st.metric("Avg Consecutive Distance", f"{avg_consec.get('overall', 0.0):.2f} km", help="Average geodetic distance between consecutive stops.")
    with col3:
        st.metric("Total Travel Distance", f"{total_dist.get('overall', 0.0):.2f} km", help="Sum of travel distances across all days.")
    with col4:
        st.metric("Geographic Compactness", f"{compactness.get('overall', 0.0):.2f} km", help="Average distance of stops to their daily geographic centroid.")

    st.markdown("##### 📊 Day Balance Analysis")
    balance = evaluation.get("day_balance", {})
    if balance:
        per_day_counts = balance.get("per_day_counts", {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Minimum Stops", f"{balance.get('min_stops', 0)}")
        with col2:
            st.metric("Maximum Stops", f"{balance.get('max_stops', 0)}")
        with col3:
            avg_stops = sum(per_day_counts.values()) / len(per_day_counts) if per_day_counts else 0.0
            st.metric("Average Stops", f"{avg_stops:.1f}")
        with col4:
            st.metric("Standard Deviation", f"{balance.get('std_dev', 0.0):.2f}")

        # Structured Table for stop counts per day
        rows_counts = [{"Day": f"Day {d_key.split('_')[1]}", "Stop Count": f"{c} stop" + ("s" if c != 1 else "")} for d_key, c in per_day_counts.items()]
        counts_df = pd.DataFrame(rows_counts)
        st.dataframe(counts_df, hide_index=True, use_container_width=True)

    # Build comparison DataFrame
    day_keys = sorted(list(set(avg_consec.keys()) | set(total_dist.keys()) | set(compactness.keys())))
    
    rows = []
    for key in day_keys:
        display_name = "Overall" if key == "overall" else f"Day {key.split('_')[1]}"
        rows.append({
            "Day/Scope": display_name,
            "Avg Consecutive Distance (km)": avg_consec.get(key, 0.0),
            "Total Travel Distance (km)": total_dist.get(key, 0.0),
            "Geographic Compactness (km to centroid)": compactness.get(key, 0.0)
        })

    metrics_df = pd.DataFrame(rows)
    if len(metrics_df) > 1:
        overall_row = metrics_df[metrics_df["Day/Scope"] == "Overall"]
        other_rows = metrics_df[metrics_df["Day/Scope"] != "Overall"]
        metrics_df = pd.concat([other_rows, overall_row], ignore_index=True)

    with st.expander("📋 View Detailed Spatial Breakdown Table"):
        st.dataframe(
            metrics_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Avg Consecutive Distance (km)": st.column_config.NumberColumn(format="%.2f km"),
                "Total Travel Distance (km)": st.column_config.NumberColumn(format="%.2f km"),
                "Geographic Compactness (km to centroid)": st.column_config.NumberColumn(format="%.2f km"),
            }
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
            "and generates clustered day-by-day itineraries (RO4-RO5)."
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
                "itinerary_num_days",
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
        st.markdown("### 🗺️ Day-by-Day Itinerary Planner")
        
        # Check if recommendations exist in session state
        rec_len = 0
        curr_algo = st.session_state.get("algorithm")
        if curr_algo:
            if curr_algo == "Compare Both Models" and "cbf_recs" in st.session_state and "cf_recs" in st.session_state:
                rec_len = max(len(st.session_state["cbf_recs"]), len(st.session_state["cf_recs"]))
            elif curr_algo != "Compare Both Models" and "recommendations" in st.session_state:
                rec_len = len(st.session_state["recommendations"])

        if rec_len == 0:
            st.info("🎯 Please search destinations and generate recommendations first on the 'Recommendations' tab before building an itinerary.")
        else:
            st.write(
                "Select geographically feasible attractions from your recommendations "
                "and group them into a day-by-day travel plan."
            )

            col_days, col_pace = st.columns(2)
            with col_days:
                num_days = st.number_input(
                    "Number of Travel Days",
                    min_value=1,
                    max_value=rec_len,
                    value=min(3, rec_len),
                    step=1,
                    help="Requested trip length. Some days may remain empty if fewer compatible attractions are selected.",
                )
            with col_pace:
                travel_pace = st.selectbox(
                    "Travel Pace",
                    options=["Relaxed", "Balanced", "Packed"],
                    index=1,
                    help="Relaxed prefers 1-2 stops/day, Balanced prefers 1-2 with occasional 3, Packed prefers 2-3.",
                )

            # Calculate active count for validation (attractions with valid coordinates)
            coordinates_df = load_coordinates_df()
            from src.itinerary import attach_coordinates
            if curr_algo == "Compare Both Models":
                cbf_recs = st.session_state.get("cbf_recs", pd.DataFrame())
                cf_recs = st.session_state.get("cf_recs", pd.DataFrame())
                cbf_cand = attach_coordinates(cbf_recs, coordinates_df)
                cf_cand = attach_coordinates(cf_recs, coordinates_df)
                active_count = min(
                    cbf_cand["attraction_uid"].nunique() if "attraction_uid" in cbf_cand.columns else len(cbf_cand),
                    cf_cand["attraction_uid"].nunique() if "attraction_uid" in cf_cand.columns else len(cf_cand),
                )
            else:
                recs = st.session_state.get("recommendations", pd.DataFrame())
                active_candidates = attach_coordinates(recs, coordinates_df)
                active_count = (
                    active_candidates["attraction_uid"].nunique()
                    if "attraction_uid" in active_candidates.columns
                    else len(active_candidates)
                )

            # Input validation check
            if num_days < 1:
                st.error("Number of travel days must be at least 1.")
                # Clear stale itinerary and evaluation states to prevent stale/incorrect views
                for key in [
                    "itinerary", "itinerary_cbf", "itinerary_cf",
                    "excluded", "excluded_cbf", "excluded_cf",
                    "evaluation", "evaluation_cbf", "evaluation_cf",
                    "itinerary_num_days", "itinerary_travel_pace", "itinerary_error",
                ]:
                    if key in st.session_state:
                        del st.session_state[key]
            elif num_days > active_count:
                st.error(
                    f"Only {active_count} recommendation candidates are available, so a maximum of "
                    f"{active_count} travel days can be populated without duplicating attractions. "
                    "Please increase Top-N or reduce the number of travel days."
                )
                # Clear stale itinerary and evaluation states to prevent stale/incorrect views
                for key in [
                    "itinerary", "itinerary_cbf", "itinerary_cf",
                    "excluded", "excluded_cbf", "excluded_cf",
                    "evaluation", "evaluation_cbf", "evaluation_cf",
                    "itinerary_num_days", "itinerary_travel_pace", "itinerary_error",
                ]:
                    if key in st.session_state:
                        del st.session_state[key]
            else:
                if "itinerary_error" in st.session_state:
                    st.error(st.session_state["itinerary_error"])

                itinerary_btn = st.button("Generate Itinerary", type="primary")

                if itinerary_btn or "itinerary_num_days" in st.session_state:
                    # Reset itinerary if number of days has changed since last generation
                    if st.session_state.get("itinerary_num_days") != num_days:
                        for key in ["itinerary", "itinerary_cbf", "itinerary_cf", "evaluation", "evaluation_cbf", "evaluation_cf", "itinerary_error"]:
                            if key in st.session_state:
                                del st.session_state[key]

                    if itinerary_btn or "itinerary_num_days" not in st.session_state:
                        with st.spinner("Generating itinerary and calculating metrics..."):
                            try:
                                if curr_algo == "Compare Both Models":
                                    cbf_recs = st.session_state["cbf_recs"]
                                    cf_recs = st.session_state["cf_recs"]
                                    if not cbf_recs.empty:
                                        itinerary_cbf, excluded_cbf = build_itinerary(
                                            cbf_recs, coordinates_df, num_days, travel_pace=travel_pace, return_excluded=True
                                        )
                                        st.session_state["itinerary_cbf"] = itinerary_cbf
                                        st.session_state["excluded_cbf"] = excluded_cbf
                                        st.session_state["evaluation_cbf"] = evaluate_itinerary(
                                            itinerary_cbf, cbf_recs
                                        )
                                    if not cf_recs.empty:
                                        itinerary_cf, excluded_cf = build_itinerary(
                                            cf_recs, coordinates_df, num_days, travel_pace=travel_pace, return_excluded=True
                                        )
                                        st.session_state["itinerary_cf"] = itinerary_cf
                                        st.session_state["excluded_cf"] = excluded_cf
                                        st.session_state["evaluation_cf"] = evaluate_itinerary(
                                            itinerary_cf, cf_recs
                                        )
                                else:
                                    recs = st.session_state["recommendations"]
                                    if not recs.empty:
                                        itinerary, excluded = build_itinerary(
                                            recs, coordinates_df, num_days, travel_pace=travel_pace, return_excluded=True
                                        )
                                        st.session_state["itinerary"] = itinerary
                                        st.session_state["excluded"] = excluded
                                        st.session_state["evaluation"] = evaluate_itinerary(
                                            itinerary, recs
                                        )
                                st.session_state["itinerary_num_days"] = num_days
                                st.session_state["itinerary_travel_pace"] = travel_pace
                                if "itinerary_error" in st.session_state:
                                    del st.session_state["itinerary_error"]
                                st.rerun()
                            except ValueError as e:
                                st.session_state["itinerary_error"] = str(e)
                                # Clear stale itinerary and evaluation states to prevent stale/incorrect views
                                for key in ["itinerary", "itinerary_cbf", "itinerary_cf", "evaluation", "evaluation_cbf", "evaluation_cf", "itinerary_num_days"]:
                                    if key in st.session_state:
                                        del st.session_state[key]
                                st.rerun()

                # Render the generated itineraries
                if "itinerary_num_days" in st.session_state:
                    saved_days = st.session_state["itinerary_num_days"]
                    if curr_algo == "Compare Both Models":
                        tab1, tab2 = st.tabs(["Content-Based Filtering Itinerary", "User-Based Collaborative Filtering Itinerary"])
                        with tab1:
                            if "itinerary_cbf" in st.session_state:
                                render_itinerary_ui(
                                    st.session_state["itinerary_cbf"],
                                    "Content-Based Filtering Itinerary",
                                    saved_days,
                                    len(st.session_state.get("cbf_recs", pd.DataFrame())),
                                    excluded_df=st.session_state.get("excluded_cbf", pd.DataFrame()),
                                )
                        with tab2:
                            if "itinerary_cf" in st.session_state:
                                render_itinerary_ui(
                                    st.session_state["itinerary_cf"],
                                    "User-Based Collaborative Filtering Itinerary",
                                    saved_days,
                                    len(st.session_state.get("cf_recs", pd.DataFrame())),
                                    excluded_df=st.session_state.get("excluded_cf", pd.DataFrame()),
                                )
                    else:
                        if "itinerary" in st.session_state:
                            render_itinerary_ui(
                                st.session_state["itinerary"],
                                f"{curr_algo} Itinerary",
                                saved_days,
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
