import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

file_path = '/Users/raghav/Desktop/Coding/PycharmProjects/Python/Movie Recommendation System/Movies_dataset.csv'
movies = pd.read_csv(file_path)

# Keep only relevant columns
movies = movies[['id', 'title', 'genre', 'original_language', 'overview']]

# Fill NaN with empty strings
for col in ['genre', 'original_language', 'overview']:
    movies[col] = movies[col].fillna('')

movies['tags'] = (
    movies['overview'] + " " +
    movies['genre'] + " " +
    movies['original_language']
)

tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
vector = tfidf.fit_transform(movies['tags'].values.astype('U')).toarray()

similarity = cosine_similarity(vector)


pickle.dump(movies, open('movies_list.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("✅ Preprocessing done. Files saved: movies_list.pkl & similarity.pkl")
