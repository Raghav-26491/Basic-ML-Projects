🎬 Movie Recommendation System

This Movie Recommendation System uses content-based filtering to recommend movies based on their similarity in tags (combining overview and genre).

📌 Overview

This project processes a dataset of movies (Movies_dataset.csv) and builds a recommendation engine using CountVectorizer and cosine similarity.

Unlike earlier versions, this implementation does not require precomputed pickle files. Instead, the app dynamically processes the dataset and caches the similarity matrix for efficiency.

✨ Features
🔹 Data Pre-processing

Cleans dataset and creates a tags column by combining overview and genre.

Handles missing values gracefully.

🔹 Recommendation Algorithm

Uses CountVectorizer + cosine similarity to compute similarity between movies.

Caches similarity matrix with Streamlit for faster performance.

🔹 Streamlit App

Interactive interface to select a movie and get top 5 similar recommendations.

Runs smoothly with no need for extra preprocessing scripts.

📂 Files

app.py – Streamlit app containing the recommendation system.

Movies_dataset.csv – Dataset containing movie information.

🛠 Setup Instructions

Acknowledgments

Dataset sourced from Kaggle.

Built using Pandas, Scikit-Learn, Numpy, and Streamlit.


Project Output Photo 

<img width="1440" height="900" alt="Movie" src="https://github.com/user-attachments/assets/56a6e789-1443-45c0-993e-4066d1ae174c" />


