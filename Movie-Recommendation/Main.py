import streamlit as st
import pickle
import numpy as np

movies = pickle.load(open('movies_list.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))


def recommend(movie):
    if movie not in movies['title'].values:
        return ["Movie not found in dataset!"]

    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]

    # Get top 6 indices (including self), then filter
    top_indices = np.argpartition(distances, -6)[-6:]
    top_indices = top_indices[np.argsort(-distances[top_indices])]

    recommended_movies = []
    for i in top_indices:
        if i != index:   # skip the selected movie itself
            recommended_movies.append(movies.iloc[i].title)

    return recommended_movies[:5]

st.title("🎬 Movie Recommendation System")

movie_list = movies['title'].values
selected_movie = st.selectbox("Select a movie:", movie_list)

if st.button("Show Recommendations"):
    recommendations = recommend(selected_movie)
    st.subheader("Recommended Movies:")
    for rec in recommendations:
        st.write("👉 " + rec)
