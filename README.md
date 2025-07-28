# Movie Recommendation System

## Introduction
This mini-project implements a movie recommendation system using content-based filtering techniques. The goal is to recommend movies to users based on textual similarity between movie features such as genres, keywords, and descriptions.

## Objective
The main objective is to build a recommendation engine that:
- Analyzes movie metadata
- Computes similarity scores between movies
- Suggests top similar movies based on user input

## Dataset
The dataset used is based on the [MovieLens](https://grouplens.org/datasets/movielens/) movie metadata, containing information such as:
- Movie titles
- Genres
- Tags and keywords
- Overview/description

The data was cleaned and preprocessed to remove null values and ensure consistency.

## Technologies Used
- Python 3.x
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Difflib


## Methodology**
1. **Data Preprocessing:** Load and clean movie data.
2. **Feature Extraction:** Use `TfidfVectorizer` to convert text (genres, tags, etc.) into numerical vectors.
3. **Similarity Calculation:** Apply `cosine_similarity` to compute similarity scores between movie vectors.
4. **User Input Matching:** Use `difflib.get_close_matches()` to handle fuzzy matching of user-provided movie titles.
5. **Recommendation:** Return top-N movies with the highest similarity scores.

## How to Run
1. Clone the repository or download the notebook.
2. Install dependencies:
## UI Design

The project includes a simple web interface built using Streamlit. Users can input a movie name, and the system will return the top 5 similar movies.

To launch the app:
```bash
streamlit run app.py

