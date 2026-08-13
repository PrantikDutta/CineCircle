import streamlit as st
import pickle
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CineCircle",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# TMDB API KEY
# ============================================================

API_KEY = st.secrets["TMDB_API_KEY"]


# ============================================================
# LOAD MODEL FILES
# ============================================================

@st.cache_resource
def load_models():

    movies = pickle.load(
        open("movies.pkl", "rb")
    )

    vectors = pickle.load(
        open("vectors.pkl", "rb")
    )

    tfidf = pickle.load(
        open("tfidf.pkl", "rb")
    )

    return movies, vectors, tfidf


try:

    movies, vectors, tfidf = load_models()

except FileNotFoundError:

    st.error(
        "❌ Model files not found. "
        "Make sure movies.pkl, vectors.pkl "
        "and tfidf.pkl are in the same folder as app.py."
    )

    st.stop()

except Exception as e:

    st.error(
        f"❌ Error loading model files: {e}"
    )

    st.stop()


# ============================================================
# TMDB MOVIE INFORMATION
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_movie_data(movie_title):

    url = "https://api.themoviedb.org/3/search/movie"

    params = {
        "api_key": API_KEY,
        "query": movie_title
    }

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    session = requests.Session()

    retry_strategy = Retry(
        total=2,
        backoff_factor=1,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504
        ],
        allowed_methods=["GET"]
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy
    )

    session.mount(
        "https://",
        adapter
    )

    try:

        response = session.get(
            url,
            params=params,
            headers=headers,
            timeout=5
        )

        response.raise_for_status()

        data = response.json()

        if data.get("results"):

            movie = data["results"][0]

            poster_path = movie.get(
                "poster_path"
            )

            if poster_path:

                poster = (
                    "https://image.tmdb.org/t/p/w500"
                    + poster_path
                )

            else:

                poster = None

            rating = movie.get(
                "vote_average"
            )

            overview = movie.get(
                "overview"
            )

            release_date = movie.get(
                "release_date"
            )

            return {
                "poster": poster,
                "rating": rating,
                "overview": overview,
                "release_date": release_date
            }

    except requests.exceptions.Timeout:

        print(
            f"TMDB timeout: {movie_title}"
        )

    except requests.exceptions.ConnectionError:

        print(
            f"TMDB connection error: {movie_title}"
        )

    except requests.exceptions.RequestException as e:

        print(
            f"TMDB request error: {e}"
        )

    except Exception as e:

        print(
            f"Unexpected TMDB error: {e}"
        )

    finally:

        session.close()

    return {
        "poster": None,
        "rating": None,
        "overview": None,
        "release_date": None
    }


# ============================================================
# MOVIE RECOMMENDATION FUNCTION
# ============================================================

def recommend_movies(
    selected_movies,
    top_n=5
):

    selected_indices = []

    for movie in selected_movies:

        matches = movies[
            movies["title"].str.lower()
            == movie.lower()
        ]

        if not matches.empty:

            selected_indices.append(
                matches.index[0]
            )

    if not selected_indices:

        return [], []


    # ========================================================
    # CALCULATE SIMILARITY ONLY WHEN NEEDED
    # ========================================================

    selected_vectors = vectors[
        selected_indices
    ]


    similarity_scores = cosine_similarity(
        selected_vectors,
        vectors
    )


    # Average similarity across selected movies

    scores = similarity_scores.mean(
        axis=0
    )


    # Don't recommend movies already selected

    for index in selected_indices:

        scores[index] = -1


    # ========================================================
    # TOP RECOMMENDATIONS
    # ========================================================

    recommended_indices = np.argsort(
        scores
    )[::-1][:top_n]


    recommendations = []

    recommendation_scores = []


    for index in recommended_indices:

        recommendations.append(
            movies.iloc[index]
        )

        recommendation_scores.append(
            scores[index]
        )


    return (
        recommendations,
        recommendation_scores
    )


# ============================================================
# MATCH LABEL
# ============================================================

def get_match_label(percentage):

    if percentage >= 85:

        return "🔥 Excellent Match"

    elif percentage >= 70:

        return "👍 Great Match"

    elif percentage >= 55:

        return "😊 Good Match"

    elif percentage >= 40:

        return "🙂 Decent Match"

    else:

        return "🎬 Worth Trying"


# ============================================================
# TITLE
# ============================================================

st.title("🎬 CineCircle")

st.subheader(
    "Find a movie everyone in your group will enjoy 🍿"
)

st.write(
    "Select the movies your friends like and CineCircle "
    "will recommend movies based on the group's preferences."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "🎥 Group Preferences"
)

num_friends = st.sidebar.slider(
    "Number of friends",
    min_value=2,
    max_value=10,
    value=3
)


# ============================================================
# MOVIE LIST
# ============================================================

movie_titles = sorted(
    movies["title"]
    .dropna()
    .unique()
    .tolist()
)


# ============================================================
# SELECT MOVIES
# ============================================================

st.markdown(
    "### 🎞️ Select movies your group likes"
)

selected_movies = []


for i in range(num_friends):

    movie = st.selectbox(
        f"Friend {i + 1}'s favourite movie",
        movie_titles,
        key=f"friend_{i}"
    )

    selected_movies.append(
        movie
    )


# ============================================================
# RECOMMENDATION BUTTON
# ============================================================

if st.button(
    "🍿 Find Our Movie"
):

    if len(set(selected_movies)) < 2:

        st.warning(
            "Please select at least two different movies."
        )

    else:

        with st.spinner(
            "Finding the perfect movie for your group... 🎬"
        ):

            recommendations, recommendation_scores = (
                recommend_movies(
                    selected_movies,
                    top_n=5
                )
            )


        if not recommendations:

            st.error(
                "Sorry, no recommendations could be generated."
            )

        else:

            st.success(
                "🎉 Here are the movies your group might enjoy!"
            )


            # =================================================
            # DISPLAY MOVIES ONE AFTER ANOTHER
            # =================================================

            for i, movie in enumerate(
                recommendations
            ):

                title = movie["title"]


                # =============================================
                # TMDB INFORMATION
                # =============================================

                movie_data = fetch_movie_data(
                    title
                )

                poster = movie_data[
                    "poster"
                ]

                rating = movie_data[
                    "rating"
                ]

                overview = movie_data[
                    "overview"
                ]

                release_date = movie_data[
                    "release_date"
                ]


                # =============================================
                # GROUP MATCH SCORE
                # =============================================

                score = recommendation_scores[
                    i
                ]

                percentage = min(
                    max(
                        score * 100,
                        0
                    ),
                    100
                )


                match_label = get_match_label(
                    percentage
                )


                # =============================================
                # MOVIE TITLE
                # =============================================

                st.markdown(
                    f"## 🎬 {i + 1}. {title}"
                )


                # =============================================
                # MOVIE CARD LAYOUT
                # =============================================

                col1, col2 = st.columns(
                    [1, 3]
                )


                # =============================================
                # POSTER
                # =============================================

                with col1:

                    if poster:

                        st.image(
                            poster,
                            width=180
                        )

                    else:

                        st.info(
                            "🎬 Poster unavailable"
                        )


                # =============================================
                # MOVIE DETAILS
                # =============================================

                with col2:

                    st.markdown(
                        f"### {match_label}"
                    )


                    # Group Match

                    st.markdown(
                        f"⭐ **Group Match: "
                        f"{percentage:.1f}%**"
                    )


                    # TMDB Rating

                    if rating is not None:

                        st.markdown(
                            f"🌟 **TMDB Rating: "
                            f"{rating:.1f}/10**"
                        )


                    # Release Year

                    if release_date:

                        year = release_date[
                            :4
                        ]

                        st.markdown(
                            f"📅 **Release Year: "
                            f"{year}**"
                        )


                    # Genres

                    if "genres" in movies.columns:

                        genres = movie[
                            "genres"
                        ]

                        if genres:

                            st.markdown(
                                f"🎭 **Genres:** "
                                f"{genres}"
                            )


                    # Overview

                    if overview:

                        st.write(
                            overview
                        )


                # =============================================
                # SEPARATOR
                # =============================================

                st.markdown(
                    "---"
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "---"
)

st.caption(
    "🎬 CineCircle | Group Movie Recommendation System"
)