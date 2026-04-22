import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Load the datasets
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge the datasets
movies = movies.merge(credits, on='title')

# Select relevant columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Drop rows with missing values
movies.dropna(inplace=True)

# Function to convert string representation of lists to actual lists
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

# Function to convert cast to top 3 actors
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

# Function to fetch director from crew
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

# Apply data cleaning functions
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(fetch_director)

# Remove spaces from list elements
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

# Create tags column
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Create new dataframe with relevant columns
new_df = movies[['movie_id', 'title', 'tags']]

# Convert tags to lowercase and join
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# Create movie dictionary
movies_dict = {
    'movie_id': new_df['movie_id'].values,
    'title': new_df['title'].values,
    'tags': new_df['tags'].values,
    'year': movies['title'].str.extract(r'(\d{4})').fillna('').values,
    'vote_average': movies.get('vote_average', pd.Series()).values
}

# Save movie dictionary
with open('movie_dict.pkl', 'wb') as f:
    pickle.dump(movies_dict, f)

print("Movie dictionary saved successfully!")

# Create vector representation
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Calculate cosine similarity
similarity = cosine_similarity(vectors)

# Save similarity matrix
with open('similarity.pkl', 'wb') as f:
    pickle.dump(similarity, f)

print("Similarity matrix saved successfully!")
print(f"Processed {len(new_df)} movies")