import pandas as pd

# File path
file_path = '/Users/raghav/Desktop/Movies_dataset.csv'

# Read the CSV file into a Pandas DataFrame
movies = pd.read_csv(file_path)

# Set Pandas options to display all columns and rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)     # Show all rows (if needed)

# Display the first 10 rows of the DataFrame
movies.head(10)
movies.describe()
movies.info()
movies.isnull().sum()
# print(movies.columns)

movies = movies[['id','title','overview','genre']]
# print(movies.head(10))

movies['tags'] = movies['overview'] + movies['genre']
# print(movies.head(10))

new_data = movies.drop(columns=['overview','genre'])
print(new_data.head(10))
print()

from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(max_features=10000,stop_words='english')
print(cv)

vector=cv.fit_transform(new_data['tags'].values.astype('U')).toarray()
print(vector.shape)
print()

from sklearn.metrics.pairwise import cosine_similarity

similarity=cosine_similarity(vector)
print(similarity)
print()

print(new_data[new_data['title']=="The Godfather"].index[0])
print()

distance = sorted(list(enumerate(similarity[2])), reverse=True, key=lambda vector:vector[1])
for i in distance[0:5]:
    print(new_data.iloc[i[0]].title)
print()

def recommend(movies):
    index=new_data[new_data['title']==movies].index[0]
    distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector: vector[1])
    for i in distance[0:5]:
        print(new_data.iloc[i[0]].title)

recommend("Iron Man")
print()
recommend("Batman Begins")
print()

import pickle

pickle.dump(new_data, open('movies_list.pkl','wb'))
pickle.dump(similarity, open('similarity.pkl','wb'))
print(pickle.load(open('movies_list.pkl','rb')).head(10))



