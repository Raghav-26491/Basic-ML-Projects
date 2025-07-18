import streamlit as st
import pickle

movies = pickle.load(open('movies_list.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))
movies_list = movies['title'].values
st.title("Movie Recommendation System")

selected_movie = st.selectbox(
    "Select a Movie:",
    movies_list
)

import streamlit.components.v1 as components

def recommend(movie):
    index = movies[movies['title']==movie].index[0]
    distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector: vector[1])
    recommend_movie = []
    for i in distance[1:6]:
        movies_id = movies.iloc[i[0]].id
        recommend_movie.append(movies.iloc[i[0]].title)
    return recommend_movie

if st.button("Show Recommendations"):
    movie_names = recommend(selected_movie)
    for movie in movie_names:
        st.write(movie)
