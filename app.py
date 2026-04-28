import pickle
import pandas as pd
import numpy as np
import requests
import streamlit as st
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import ast

# -------------------------------
# Load .env variables
# -------------------------------
load_dotenv()
api_key = os.getenv("TMDB_API_KEY")

# -------------------------------
# Data Preparation Functions (for when pickle files don't exist)
# -------------------------------
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

def prepare_data():
    """Generate pickle files from CSV if they don't exist"""
    movies_df = pd.read_csv('tmdb_5000_movies.csv')
    credits_df = pd.read_csv('tmdb_5000_credits.csv')
    
    movies_df = movies_df.merge(credits_df, on='title')
    movies_df = movies_df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies_df.dropna(inplace=True)
    
    movies_df['genres'] = movies_df['genres'].apply(convert)
    movies_df['keywords'] = movies_df['keywords'].apply(convert)
    movies_df['cast'] = movies_df['cast'].apply(convert3)
    movies_df['crew'] = movies_df['crew'].apply(fetch_director)
    
    movies_df['genres'] = movies_df['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies_df['keywords'] = movies_df['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies_df['cast'] = movies_df['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies_df['crew'] = movies_df['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
    
    movies_df['tags'] = movies_df['overview'].fillna('') + ' ' + movies_df['genres'].apply(lambda x: ' '.join(x)) + ' ' + movies_df['keywords'].apply(lambda x: ' '.join(x)) + ' ' + movies_df['cast'].apply(lambda x: ' '.join(x)) + ' ' + movies_df['crew'].apply(lambda x: ' '.join(x))
    movies_df['tags'] = movies_df['tags'].apply(lambda x: x.lower())
    
    new_df = movies_df[['movie_id', 'title', 'tags']]
    
    # Create movie_dict.pkl
    movies_dict = {
        'movie_id': new_df['movie_id'].values,
        'title': new_df['title'].values,
        'tags': new_df['tags'].values,
    }
    with open('movie_dict.pkl', 'wb') as f:
        pickle.dump(movies_dict, f)
    
    # Create similarity.pkl
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    similarity = cosine_similarity(vectors)
    
    with open('similarity.pkl', 'wb') as f:
        pickle.dump(similarity, f)
    
    return movies_dict, similarity

# -------------------------------
# Load Data
# -------------------------------
# Check if pickle files exist, if not generate them
if not os.path.exists('movie_dict.pkl') or not os.path.exists('similarity.pkl'):
    st.info("Generating movie data... Please wait.")
    data, similarity = prepare_data()
else:
    data = pickle.load(open('movie_dict.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))

# Fix pickle data: flatten 2D arrays and filter empty ones
fixed_data = {}
for k, v in data.items():
    arr = np.array(v)
    if arr.size > 0:
        fixed_data[k] = arr.flatten()
movies = pd.DataFrame(fixed_data)

movies_list = movies['title'].values

# -------------------------------
# Performance Optimization: Pre-sorted indices
# -------------------------------
# Pre-compute sorted indices for all movies (faster than sorting on each request)
print("Pre-computing sorted similarity indices...")
sorted_indices = np.argsort(-similarity, axis=1)
# Store top 6 (including self) for each movie
top_similar = sorted_indices[:, 1:6]  # Shape: (4806, 5)
print("Pre-computation complete!")

# -------------------------------
# TMDB Poster Fetch Function (Optimized)
# -------------------------------
poster_cache = {}
# Create a movie_id to title mapping for quick lookups
movie_id_to_title = dict(zip(movies['movie_id'].values, movies['title'].values))

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
            movie_title = movie_id_to_title.get(movie_id, "Movie")
            poster_url = f"https://via.placeholder.com/500x750?text={movie_title.replace(' ', '+')}"
    
    except:
        movie_title = movie_id_to_title.get(movie_id, "Movie")
        poster_url = f"https://via.placeholder.com/500x750?text={movie_title.replace(' ', '+')}"
    
    poster_cache[movie_id] = poster_url
    return poster_url

def fetch_posters_parallel(movie_ids):
    """Fetch multiple posters in parallel"""
    posters = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_poster, mid): mid for mid in movie_ids}
        for future in as_completed(futures):
            posters.append(future.result())
    return posters

# -------------------------------
# Recommendation Function (Optimized)
# -------------------------------
# Create title to index mapping for O(1) lookup
title_to_index = {title: idx for idx, title in enumerate(movies['title'].values)}

def recommend(movie):
    # O(1) lookup instead of O(n) search
    index = title_to_index.get(movie)
    if index is None:
        return [], []
    
    # Use pre-sorted indices (no sorting needed!)
    similar_indices = top_similar[index]
    
    recommended_movies = []
    recommended_posters = []
    
    # Fetch posters in parallel
    movie_ids = [movies.iloc[i].movie_id for i in similar_indices]
    titles = [movies.iloc[i].title for i in similar_indices]
    posters = fetch_posters_parallel(movie_ids)
    
    return titles, posters

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommendation System")

# Use session state to cache recommendations
if 'last_movie' not in st.session_state:
    st.session_state.last_movie = None
if 'cached_recommendations' not in st.session_state:
    st.session_state.cached_recommendations = None

selected_movie = st.selectbox("Select a movie", movies_list)

if st.button("Recommend"):
    # Check cache first
    if st.session_state.last_movie == selected_movie and st.session_state.cached_recommendations:
        names, posters = st.session_state.cached_recommendations
        st.info("📦 Loaded from cache (no API calls)")
    else:
        with st.spinner("Finding similar movies..."):
            names, posters = recommend(selected_movie)
        # Cache the results
        st.session_state.last_movie = selected_movie
        st.session_state.cached_recommendations = (names, posters)

    st.subheader("Top 5 Recommendations")

    cols = st.columns(5)

    for i in range(5):
        with cols[i]:
            st.markdown(f"**{names[i]}**")
            # Enable poster display
            st.image(posters[i])
