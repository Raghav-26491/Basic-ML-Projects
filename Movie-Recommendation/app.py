import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def load_data():
    movies = pd.read_csv("Movies_dataset.csv")

    movies = movies[['id', 'title', 'overview', 'genre']]

    movies['overview'] = movies['overview'].fillna("")
    movies['genre'] = movies['genre'].fillna("")

    movies['tags'] = movies['overview'] + " " + movies['genre']

    new_data = movies[['id', 'title', 'tags']]

    return new_data


@st.cache_resource
def compute_similarity(data):
    cv = CountVectorizer(max_features=10000, stop_words='english')
    vectors = cv.fit_transform(data['tags'].values.astype('U')).toarray()

    similarity = cosine_similarity(vectors)
    return similarity


def recommend(movie, data, similarity):
    if movie not in data['title'].values:
        return ["Movie not found in dataset"]

    index = data[data['title'] == movie].index[0]
    distances = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )

    recommended_movies = []
    for i in distances[1:6]:
        recommended_movies.append(data.iloc[i[0]].title)

    return recommended_movies


st.title("🎬 Movie Recommendation System")

movies = load_data()
similarity = compute_similarity(movies)

selected_movie = st.selectbox("Select a movie:", movies['title'].values)

if st.button("Show Recommendations"):
    recommendations = recommend(selected_movie, movies, similarity)
    st.write("### Recommended Movies:")
    for r in recommendations:
        st.write("👉", r)
