Movie Recommendation System

This Movie Recommendation System uses content-based filtering to recommend movies based on their similarity in tags (combining overview and genre).

Overview
This project utilizes a dataset of movies (Movies_dataset.csv) containing information about movie titles, overviews, and genres. Initially, the dataset is processed to extract relevant information and create a recommendation system based on movie similarities.

Features

•	Data Pre-processing: The dataset is cleaned and transformed to create a tags column by combining overview and genre.

•	Recommendation Algorithm: Uses cosine similarity to suggest movies similar to the selected one.

•	Streamlit App: Utilizes Streamlit for creating an interactive user interface to select movies and view recommendations.

How to Use

•	Setup: Clone this repository to your local environment.

•	Dependencies: Install necessary dependencies by running pip install -r requirements.txt.

•	Run the App: Execute streamlit run app.py to launch the app.

•	Select a Movie: Choose a movie from the dropdown menu.

•	Get Recommendations: Click the "Show Recommendations" button to view recommended movies.

Files

1.	main.py: Contains the main logic for generating movie recommendations.

2.	app.py: Streamlit-based interface to interact with the recommendation system.

3.	Movies_dataset.csv: Dataset used for movie information.

4.	movies_list.pkl: Pickled file containing processed movie data.

5.	similarity.pkl: (Not included) The similarity.pkl file could not be added to the repository due to its large size.

Setup Instructions

•	Clone the repository:
•	git clone https://github.com/Raghav-26491/Movie-Recommendation-System.git

•	Run the Streamlit app:
•	streamlit run app.py

Acknowledgments

The dataset is sourced from Kaggle.

This project is a part of my Bharat Intern Internship Task.

Project Output Photo 

<img width="1440" height="900" alt="Movie" src="https://github.com/user-attachments/assets/56a6e789-1443-45c0-993e-4066d1ae174c" />


