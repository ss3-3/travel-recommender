# Travel Recommendation System

## 1. Project Overview

This project implements a travel destination recommendation system for suggesting tourist attractions based on user preferences and historical rating behaviour.

The recommendation problem addressed is personalized attraction ranking: given a tourist/user, the system recommends attractions that the user has not already visited or rated.

Two recommendation approaches are implemented:

- **Content-Based Filtering (CBF):** recommends attractions whose descriptive features are similar to attractions the user rated highly.
- **User-Based Collaborative Filtering (CF):** recommends attractions based on rating patterns from similar users.

The system prepares attraction and interaction data, builds model-specific representations, generates ranked recommendations, and provides an interactive Streamlit interface for demonstration.

## 2. System Features

- Personalized attraction recommendations by tourist ID
- Content-Based Filtering using TF-IDF attraction features and cosine similarity
- User-Based Collaborative Filtering using user similarity and similarity-weighted rating prediction
- Exclusion of already visited/rated attractions from recommendation results
- Interactive Streamlit application for selecting users, models, and recommendation count
- Offline evaluation using Precision@K, Recall@K, F1-score@K, and Coverage
- Verification scripts for preprocessing, recommendation logic, and evaluation

## 3. Folder Structure

```text
travel-recommender/
├── app.py
├── README.md
├── requirements.txt
├── pyrefly.toml
├── check_dataset.py
├── debug_cf.py
├── data/
│   └── tourism_recommendation_dataset_en.csv
├── docs/
│   ├── architecture.md
│   ├── ai_decisions.md
│   ├── chapter1_notes.md
│   ├── chapter2_notes.md
│   ├── literature_matrix.xlsx
│   ├── meeting_notes.md
│   ├── project_progress.md
│   └── research_questions.md
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_content_based.ipynb
│   ├── 03_collaborative.ipynb
│   └── 04_evaluation.ipynb
├── scratch/
│   ├── test_preprocessing.py
│   ├── test_content_based.py
│   ├── test_collaborative.py
│   ├── test_evaluation.py
│   ├── check_dataset.py
│   └── verify_split.py
└── src/
    ├── preprocessing.py
    ├── content_based.py
    ├── collaborative.py
    └── evaluation.py
```

Important files and folders:

- `app.py`: Streamlit application for running the recommender interactively.
- `src/preprocessing.py`: data loading, validation, cleaning, attraction preparation, interaction preparation, and train/test splitting.
- `src/content_based.py`: Content-Based Filtering implementation.
- `src/collaborative.py`: User-Based Collaborative Filtering implementation.
- `src/evaluation.py`: offline evaluation metrics and model comparison helpers.
- `data/`: contains the tourism recommendation dataset used by the project.
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
- Choose Content-Based Filtering, User-Based Collaborative Filtering, or compare both models
- Select the number of recommendations to display
- View ranked attraction recommendations with model-specific scores

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

### User-Based Collaborative Filtering

The User-Based Collaborative Filtering model builds a user-item rating matrix from training interactions, where rows represent tourists and columns represent attractions.

Cosine similarity is computed between users based on overlapping rated attractions. For a target tourist, the system finds the top similar neighbours and predicts attraction ratings using a similarity-weighted average of neighbour ratings. Already visited/rated attractions are excluded, and recommendations are ranked by predicted rating.

## 9. Evaluation

The evaluation module compares CBF and CF using held-out test interactions from the per-user train/test split.

Implemented metrics:

- **Precision@K:** proportion of recommended attractions that appear in the user's held-out interactions
- **Recall@K:** proportion of held-out attractions captured by the recommendation list
- **F1-score@K:** harmonic mean of Precision@K and Recall@K
- **Coverage:** proportion of test users for whom the model returns a non-empty recommendation list

Evaluation is performed offline so that both recommendation approaches can be compared using the same train/test interaction split.

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
| User-Based Collaborative Filtering | Rating patterns from similar users |

CBF focuses on item characteristics, while CF leverages similarities between users.