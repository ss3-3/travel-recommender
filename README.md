# Travel Recommendation System

## 1. Project Overview

The main research focus of this project is the controlled comparison of **Content-Based Filtering (CBF)** and **User-Based Collaborative Filtering (UBCF)** for personalized attraction recommendation. Both models are evaluated under the same experimental conditions using per-user train/test interaction splits.

Given a tourist/user, the system recommends attractions that the user has not already visited or rated.

- **Content-Based Filtering (CBF):** Recommends attractions whose descriptive features are similar to attractions the user rated highly.
- **User-Based Collaborative Filtering (UBCF):** Recommends attractions based on rating patterns from similar users.

As a downstream practical extension of the recommendation system, the project also includes an **itinerary generation module** that constructs realistic, geographically coherent travel plans from the recommended candidate attractions. The itinerary module is not a globally optimal route optimizer, but rather a feasibility-driven scheduling extension.

## 2. System Features

- Personalized attraction recommendations by tourist ID
- Content-Based Filtering using TF-IDF attraction features and cosine similarity
- User-Based Collaborative Filtering (UBCF) using user similarity and similarity-weighted rating prediction
- Controlled model comparison evaluated under the same experimental conditions
- Downstream one-day itinerary module implementing:
  - Geographic candidate selection using a fixed-radius (50 km) Haversine-distance grouping around the best-supported anchor candidate
  - Haversine distance-based Greedy Nearest-Neighbor route ordering among the selected stops
- Exclusion of already visited/rated attractions from recommendation results
- Interactive Streamlit application for selecting users, models, travel pace, and trip duration
- Offline evaluation using Precision@K, Recall@K, F1@K, and Recommendation Coverage

## 3. Folder Structure

```text
travel-recommender/
├── app.py
├── README.md
├── requirements.txt
├── pyrefly.toml
├── check_dataset.py
├── data/
│   ├── tourism_recommendation_dataset_en.csv
│   └── coordinates.csv
├── docs/
│   ├── architecture.md
│   ├── ai_decisions.md
│   ├── chapter1_notes.md
│   ├── chapter2_notes.md
│   ├── literature_matrix.xlsx
│   ├── project_progress.md
│   └── research_questions.md
├── notebooks/
│   ├── 01_preprocessing_analysis.ipynb
│   ├── 03_content_based.ipynb
│   ├── 04_collaborative.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_itinerary_analysis.ipynb
├── scratch/
│   ├── test_preprocessing.py
│   ├── test_content_based.py
│   ├── test_collaborative.py
│   ├── test_itinerary.py
│   ├── test_evaluation.py
│   ├── check_dataset.py
│   ├── debug_cf.py
│   └── verify_split.py
└── src/
    ├── preprocessing.py
    ├── content_based.py
    ├── collaborative.py
    ├── evaluation.py
    └── itinerary.py
```

Important files and folders:

- `app.py`: Streamlit application for running the recommender interactively.
- `src/preprocessing.py`: data loading, validation, cleaning, attraction preparation, interaction preparation, and train/test splitting.
- `src/content_based.py`: Content-Based Filtering implementation.
- `src/collaborative.py`: User-Based Collaborative Filtering implementation.
- `src/evaluation.py`: offline evaluation metrics and model comparison helpers.
- `src/itinerary.py`: day-by-day itinerary generation logic (clustering and sequencing recommended attractions).
- `data/`: contains the tourism recommendation dataset and geodetic `coordinates.csv` used by the project.
- `notebooks/`: exploratory analysis and model development notebooks.
- `scratch/`: verification scripts used during development.
- `docs/architecture.md`: architecture and implementation design documentation.
- `requirements.txt`: project dependency list.

## 4. Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 5. Running the Project

Launch the Streamlit application from the project root:

```bash
streamlit run app.py
```

In the application, users can:

- Select a tourist ID
- Choose Content-Based Filtering, User-Based Collaborative Filtering (UBCF), or compare both models
- Select the number of recommendations to display
- View ranked attraction recommendations with model-specific scores
- Specify the number of destinations (M) to include and generate a geographically grouped one-day itinerary (using fixed-radius 50 km geographic candidate selection and Haversine Greedy Nearest-Neighbor ordering)
- Download the generated itinerary as a CSV file

## 6. Dataset

The dataset is used to support tourist attraction recommendation based on user-attraction interactions.

Main data entities:

- **Tourists/users:** represented by `tourist_id`
- **Attractions:** represented using attraction name, category, level, city, province, ticket price, and main spots
- **Ratings/interactions:** represented by user ratings for attractions

Implemented preprocessing steps include:

- Loading the CSV dataset
- Validating required columns
- Cleaning text fields
- Filling missing `main_spots` values with empty strings
- Creating a unique `attraction_uid` using attraction name, city, and province
- Preparing attraction metadata for CBF
- Preparing user-item interaction data for CF
- Splitting interactions into train and test sets per user

## 7. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Streamlit
- Jupyter Notebook (for exploratory analysis and model development)

## 8. Recommendation Algorithms

### Content-Based Filtering

The Content-Based Filtering model represents each attraction using a combined text feature built from attraction category, attraction level, province, city, and cleaned main spots. Category and level are repeated in the content string to give them stronger influence.

The attraction content is transformed using TF-IDF. For a target user, the system builds a rating-weighted user profile from attractions the user rated at or above the rating threshold of `4.0`. Cosine similarity is then computed between the user profile and all attraction vectors. Already visited/rated attractions are removed, and the remaining attractions are ranked by similarity score.

### User-Based Collaborative Filtering (UBCF)

The User-Based Collaborative Filtering model builds a user-item rating matrix from training interactions, where rows represent tourists and columns represent attractions.

Cosine similarity is computed between users based on overlapping rated attractions. For a target tourist, the system finds the top similar neighbours and predicts attraction ratings using a similarity-weighted average of neighbour ratings. Already visited/rated attractions are excluded, and recommendations are ranked by predicted rating.

### Downstream Itinerary Generation

The itinerary module is a downstream practical extension of the recommendation system. It translates the Top-N recommended candidates into a single, geographically feasible one-day travel plan capped at a user-specified number of destinations (M). It is not designed to find globally optimal routes.

The itinerary generation pipeline runs as follows:

1. **Attach Coordinates:** Joins the Top-N candidate recommendations with geodetic latitude/longitude from `data/coordinates.csv`.
2. **Geographic Candidate Selection:** For each candidate, computes the Haversine distance to every other candidate and groups those within a fixed 50 km radius. The candidate whose group has the most members is selected as the anchor (ties broken by better/lower recommendation rank); candidates outside the anchor's group are excluded from the itinerary as geographically incompatible but remain visible as valid recommendations in the UI.
3. **Capacity Capping:** The anchor's group is capped at the requested number of destinations (M), keeping the highest-ranked members.
4. **Haversine Route Ordering:** The selected stops are sequenced using a Greedy Nearest-Neighbor heuristic based on the Haversine distance formula, starting from the lowest-rank (highest-priority) stop.

## 9. Evaluation

The offline evaluation module compares CBF and UBCF under the same experimental conditions, using held-out test interactions from a per-user train/test split.

Evaluation is performed offline so that both recommendation approaches can be compared fairly using the same train/test interaction split.

Implemented metrics:

- **Precision@K:** Proportion of recommended attractions that appear in the user's held-out interactions.
- **Recall@K:** Proportion of held-out attractions captured by the recommendation list.
- **F1@K:** Harmonic mean of Precision@K and Recall@K.
- **Recommendation Coverage:** Proportion of test users for whom the model returns a non-empty recommendation list.

Separately, the generated one-day itinerary is evaluated using dedicated itinerary-level metrics (see `src/itinerary_evaluation.py`):

- **Average Consecutive-Stop Distance:** Mean Haversine distance between consecutive stops in the generated route.
- **Total Travel Distance:** Sum of Haversine distances across the full route.
- **Geographic Compactness:** Average pairwise Haversine distance between all selected stops.
- **Candidate Carryover Rate:** Proportion of Top-N recommended candidates retained in the final itinerary.

## 10. Development Workflow

1. Data preprocessing
2. Feature construction
3. Model implementation
4. Offline evaluation
5. Streamlit integration
6. Testing and refinement

## 11. Future Improvements

- Hybrid recommendation combining CBF and CF outputs
- Improved cold-start handling for new users or users with limited ratings
- More advanced ranking models
- Larger or more diverse tourism datasets
- Additional attraction metadata for richer personalization

## 12. Model Comparison

The project implements two recommendation approaches with different recommendation strategies:

| Model | Recommendation Basis |
|---|---|
| Content-Based Filtering | Similarity between attraction features and user preference profile |
| User-Based Collaborative Filtering (UBCF) | Rating patterns from similar users |

CBF focuses on item characteristics, while UBCF leverages similarities between users.