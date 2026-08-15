import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# Title
st.title("🎬 Movie Recommendation System")
st.write("Enter a movie name and get 5 similar movies instantly!")

# Load and process data
@st.cache_data
def load_data():
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    
    credits.columns = ['id', 'title', 'cast', 'crew']
    movies = movies.merge(credits, on='title')

    def extract_names(text, limit=3):
        try:
            items = ast.literal_eval(text)
            return [item['name'].replace(" ", "") for item in items[:limit]]
        except:
            return []

    def get_director(text):
        try:
            crew = ast.literal_eval(text)
            for member in crew:
                if member['job'] == 'Director':
                    return [member['name'].replace(" ", "")]
            return []
        except:
            return []

    movies['genres'] = movies['genres'].apply(extract_names)
    movies['keywords'] = movies['keywords'].apply(extract_names)
    movies['cast'] = movies['cast'].apply(extract_names)
    movies['crew'] = movies['crew'].apply(get_director)
    movies['tags'] = movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    movies['tags'] = movies['tags'].apply(lambda x: ' '.join(x))

    data = movies[['id_x', 'title', 'tags']].copy()
    data.rename(columns={'id_x': 'id'}, inplace=True)

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(data['tags']).toarray()
    similarity = cosine_similarity(vectors)

    return data, similarity

# Load data
with st.spinner('Loading movie database...'):
    data, similarity = load_data()

st.success(f"✅ {len(data)} movies loaded!")

# Movie selector
movie_list = data['title'].values
selected_movie = st.selectbox("🎥 Select a movie:", movie_list)

# Recommend button
if st.button("🔍 Get Recommendations"):
    idx = data[data['title'] == selected_movie].index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
    
    st.subheader(f"Movies similar to '{selected_movie}':")
    
    cols = st.columns(5)
    for i, (index, score) in enumerate(distances[1:6]):
        with cols[i]:
            st.info(f"🎬 {data.iloc[index]['title']}")