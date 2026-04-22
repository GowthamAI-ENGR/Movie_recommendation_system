import pickle
import pandas as pd
import requests
import streamlit as st
import os
from dotenv import load_dotenv

# -------------------------------
# Load .env variables
# -------------------------------
load_dotenv()
api_key = os.getenv("TMDB_API_KEY")

# -------------------------------
# Load Data
# -------------------------------
movies = pd.DataFrame(pickle.load(open('movie_dict.pkl', 'rb')))
similarity = pickle.load(open('similarity.pkl', 'rb'))

movies_list = movies['title'].values

# -------------------------------
# TMDB Poster Fetch Function
# -------------------------------
poster_cache = {}

def fetch_poster(movie_id):
    if movie_id in poster_cache:
        return poster_cache[movie_id]
    
    if not api_key:
        poster_url = "https://via.placeholder.com/500x750?text=API+Key+Missing"
        poster_cache[movie_id] = poster_url
        return poster_url
    
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}"
    
    try:
        data = requests.get(url, timeout=10).json()
        
        if data.get('poster_path'):
            poster_url = "https://image.tmdb.org/t/p/w500/" + data['poster_path']
        else:
            # Get movie title for placeholder
            movie_title = movies.iloc[movies[movies['movie_id'] == movie_id].index[0]].title
            poster_url = f"https://via.placeholder.com/500x750?text={movie_title.replace(' ', '+')}"
    
    except:
        # Get movie title for placeholder
        try:
            movie_title = movies.iloc[movies[movies['movie_id'] == movie_id].index[0]].title
            poster_url = f"https://via.placeholder.com/500x750?text={movie_title.replace(' ', '+')}"
        except:
            poster_url = "https://via.placeholder.com/500x750?text=Movie+Poster"
    
    poster_cache[movie_id] = poster_url
    return poster_url

# -------------------------------
# Recommendation Function
# -------------------------------
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movie_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_posters

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommendation System")

st.info("Note: Movie posters are currently unavailable due to network connectivity issues. Showing movie titles only.")

selected_movie = st.selectbox("Select a movie", movies_list)

if st.button("Recommend"):
    names, posters = recommend(selected_movie)

    st.subheader("Top 5 Recommendations")

    cols = st.columns(5)

    for i in range(5):
        with cols[i]:
            st.markdown(f"**{names[i]}**")
            # Temporarily disabled poster display due to network issues
            # st.image(posters[i])
