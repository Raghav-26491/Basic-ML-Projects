🎬 Movie Recommendation System

This Movie Recommendation System uses content-based filtering to recommend movies based on their similarity in tags (combining overview, genre, and language).

📌 Overview

This project processes a dataset of movies (Movies_dataset.csv) and builds a recommendation engine using TF-IDF Vectorization and cosine similarity.

To keep the app fast, heavy preprocessing (vectorization + similarity matrix) is done once and saved as pickles. The Streamlit app then loads these precomputed files for real-time recommendations.

✨ Features

Data Pre-processing:

Cleans dataset and creates a tags column by combining overview, genre, and original_language.

Recommendation Algorithm:

Uses TF-IDF + cosine similarity to compute similarity between movies.

Optimized with numpy.argpartition for faster top-N recommendations.

Streamlit App:

Interactive interface to select a movie and instantly get similar recommendations.

Runs smoothly even for large datasets.

📂 Files

1. preprocess.py – Precomputes TF-IDF vectors & cosine similarity, saves results into pickle files.

2. main.py – Streamlit app that loads preprocessed files and provides recommendations.

3. Movies_dataset.csv – Dataset containing movie information.

4. movies_list.pkl – Pickled processed movie metadata.

5. similarity.pkl – Pickled similarity matrix (not included in repo due to large size; generate it with preprocess.py).

🛠 Setup Instructions

Clone repository.

1. Run python preprocess.py once to generate pickle files.

2. Launch app with streamlit run main.py.

3. Select a movie → Click Show Recommendations → See top 5 similar movies.

🙏 Acknowledgments

Dataset sourced from Kaggle.

Built using Pandas, Scikit-Learn, Numpy, and Streamlit.


Project Output Photo 

<img width="1440" height="900" alt="Movie" src="https://github.com/user-attachments/assets/56a6e789-1443-45c0-993e-4066d1ae174c" />


