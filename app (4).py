import streamlit as st
import pandas as pd
import ast
from pathlib import Path

from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Recommender System")
st.write("Select a movie and get the Top 5 similar movies.")


# ---------------------------------------------------------
# File paths
# Put these files in the same folder as app.py
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MOVIE_FILE = BASE_DIR / "tmdb_5000_movies.csv.zip"
CREDITS_FILE = BASE_DIR / "tmdb_5000_credits.csv.zip"


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def load_csv_file(path):
    """Load either a CSV or a zipped CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path.name}"
        )
    return pd.read_csv(path)


def convert(obj):
    result = []
    for item in ast.literal_eval(obj):
        result.append(item["name"])
    return result


def convert3(obj):
    result = []
    for item in ast.literal_eval(obj):
        if len(result) < 3:
            result.append(item["name"])
        else:
            break
    return result


def fetch_director(obj):
    result = []
    for item in ast.literal_eval(obj):
        if item["job"] == "Director":
            result.append(item["name"])
            break
    return result


@st.cache_data
def prepare_data():
    """
    Reproduces the preprocessing used in the uploaded notebook.
    """
    movie = load_csv_file(MOVIE_FILE)
    credits = load_csv_file(CREDITS_FILE)

    # Same merge used in the notebook
    df = movie.merge(credits, on="title")

    df = df[
        ["id", "title", "genres", "keywords",
         "overview", "cast", "crew"]
    ]

    # Same missing-value handling
    df.dropna(inplace=True)

    # Convert JSON-like columns
    df["genres"] = df["genres"].apply(convert)
    df["keywords"] = df["keywords"].apply(convert)
    df["cast"] = df["cast"].apply(convert3)
    df["crew"] = df["crew"].apply(fetch_director)

    # Overview -> list of words
    df["overview"] = df["overview"].apply(
        lambda x: x.split()
    )

    # Remove spaces inside names
    df["genres"] = df["genres"].apply(
        lambda x: [i.replace(" ", "") for i in x]
    )
    df["keywords"] = df["keywords"].apply(
        lambda x: [i.replace(" ", "") for i in x]
    )
    df["cast"] = df["cast"].apply(
        lambda x: [i.replace(" ", "") for i in x]
    )
    df["crew"] = df["crew"].apply(
        lambda x: [i.replace(" ", "") for i in x]
    )

    # Create tag exactly as in the notebook
    df["tag"] = (
        df["overview"]
        + df["genres"]
        + df["keywords"]
        + df["cast"]
        + df["crew"]
    )

    df = df[["id", "title", "tag"]]

    df["tag"] = df["tag"].apply(
        lambda x: " ".join(x)
    )

    # Lowercase
    df["tag"] = df["tag"].apply(
        lambda x: x.lower()
    )

    # Porter stemming
    ps = PorterStemmer()

    def stem(text):
        words = []
        for word in text.split():
            words.append(ps.stem(word))
        return " ".join(words)

    df["tag"] = df["tag"].apply(stem)

    # Same CountVectorizer settings as the notebook
    cv = CountVectorizer(
        max_features=5000,
        stop_words="english"
    )

    vectors = cv.fit_transform(df["tag"])

    return df.reset_index(drop=True), vectors


@st.cache_data
def recommend(movie_title):
    """
    Returns the Top 5 recommendations using cosine similarity.
    Instead of storing the entire 4800x4800 similarity matrix,
    calculate similarity only for the selected movie.
    """
    df, vectors = prepare_data()

    matches = df[
        df["title"].str.lower() == movie_title.lower()
    ]

    if matches.empty:
        return pd.DataFrame()

    movie_index = matches.index[0]

    # Similarity of selected movie against all movies
    distances = cosine_similarity(
        vectors[movie_index],
        vectors
    ).flatten()

    # Sort by similarity, excluding the selected movie
    movie_list = sorted(
        enumerate(distances),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    recommendations = []

    for index, score in movie_list:
        recommendations.append({
            "Movie": df.iloc[index]["title"],
            "Similarity": round(float(score), 4)
        })

    return pd.DataFrame(recommendations)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
try:
    df, vectors = prepare_data()
except Exception as e:
    st.error("Could not load the movie dataset.")
    st.code(str(e))
    st.info(
        "Keep tmdb_5000_movies.csv.zip and "
        "tmdb_5000_credits.csv.zip in the same folder as app.py."
    )
    st.stop()


# ---------------------------------------------------------
# User interface
# ---------------------------------------------------------
movie_titles = df["title"].drop_duplicates().sort_values().tolist()

selected_movie = st.selectbox(
    "Choose a movie:",
    movie_titles
)

if st.button("🎯 Recommend Top 5", use_container_width=True):

    recommendations = recommend(selected_movie)

    if recommendations.empty:
        st.warning("Movie not found.")
    else:
        st.subheader(
            f"Top 5 recommendations for: {selected_movie}"
        )

        for rank, row in recommendations.iterrows():
            st.markdown(
                f"### {rank + 1}. {row['Movie']}"
            )
            st.caption(
                f"Similarity score: {row['Similarity']:.4f}"
            )

        st.success("Recommendations generated successfully!")


# ---------------------------------------------------------
# Project information
# ---------------------------------------------------------
with st.expander("About this project"):
    st.write(
        """
        This movie recommender system is based on the approach
        implemented in the accompanying Jupyter Notebook.

        The system:
        - combines movie overview, genres, keywords, cast and director
        - applies Porter stemming
        - converts text into numerical features using CountVectorizer
        - calculates cosine similarity
        - returns the Top 5 most similar movies
        """
    )
