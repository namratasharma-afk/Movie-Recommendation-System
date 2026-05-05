import streamlit as st
import pandas as pd
import numpy as np
import requests
import urllib.parse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv('data/movies.csv')
    ratings = pd.read_csv('data/ratings.csv')
    return movies, ratings

movies, ratings = load_data()

@st.cache_resource
def build_models(movies, ratings):
    movies['genres'] = movies['genres'].str.replace('|', ' ')
    movies['tags'] = movies['genres']

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()
    similarity = cosine_similarity(vectors)

    movie_ratings = ratings.merge(movies, on='movieId')
    pt = movie_ratings.pivot_table(index='title', columns='userId', values='rating')
    pt.fillna(0, inplace=True)

    similarity_scores = cosine_similarity(pt)

    return similarity, pt, similarity_scores

similarity, pt, similarity_scores = build_models(movies, ratings)

def collaborative_recommend(movie):
    if movie not in pt.index:
        return []

    index = pt.index.get_loc(movie)
    distances = similarity_scores[index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:10]

    return [pt.index[i[0]] for i in movies_list if pt.index[i[0]] != movie]


# -----------------------------
# Hybrid Model
# -----------------------------
def hybrid_recommend(movie):
    content_movies = collaborative_recommend(movie)
    collab_movies = collaborative_recommend(movie)

    final = []
    
    # pehle content-based add karo
    for m in content_movies:
        if m not in final:
            final.append(m)
    
    # phir collaborative add karo
    for m in collab_movies:
        if m not in final:
            final.append(m)
    
    return final[:10]   # always 10

def fetch_poster(movie_title):
    api_key = "e6946520"
    
    # Step 1: Clean title (remove year)
    clean_title = movie_title.split('(')[0].strip()
    
    # Step 2: URL encode 
    clean_title = urllib.parse.quote(clean_title)
    
    url = f"http://www.omdbapi.com/?t={clean_title}&apikey={api_key}"
    
    data = requests.get(url).json()
    
    # Debug print (optional)
    # print(data)

    if data.get('Response') == "True" and data.get('Poster') != "N/A":
        return data.get('Poster')
    else:
        return None


# -----------------------------
# UI
# -----------------------------
st.markdown(
    "<h1 style='text-align: center;'>🎬 Movie Recommendation System</h1>",
    unsafe_allow_html=True
)

movie_list = movies['title'].values
selected_movie = st.selectbox("Select a movie", movie_list, key="movie_select")

if st.button("🎯 Get Recommendations"):
    try:
        with st.spinner("Finding best movies for you... 🎬"):
            recommendations = hybrid_recommend(selected_movie)

        st.markdown("## 🍿 Top Picks For You")

        if not recommendations:
            st.warning("No recommendations found.")
        else:
            cols = st.columns(5)

            for i, movie in enumerate(recommendations):
                poster = fetch_poster(movie)

                with cols[i % 5]:
                    if poster:
                        st.image(poster, use_container_width=True)
                    else:
                        st.image("https://via.placeholder.com/200x300?text=No+Image", use_container_width=True)

                    st.caption(movie)

    except Exception as e:
        st.error(f"Error: {e}")