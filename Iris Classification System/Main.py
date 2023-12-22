import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load the csv data
df = pd.read_csv("Iris.csv")
# print(df.head())

print()

# deleting ID column
df = df.drop(columns=['Id'])
print(df.head())

print(df.describe())
print()
print(df.info())
print()

print(df['Species'].value_counts())
print()

# Preprocessing Dataset
print(df.isnull().sum())

# Data Analysis
# Plot histograms
plt.figure(figsize=(8, 6))

plt.subplot(221)
df['SepalLengthCm'].hist()
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.title('Histogram of Sepal Length')

plt.subplot(222)
df['SepalWidthCm'].hist()
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.title('Histogram of Sepal Width')

plt.subplot(223)
df['PetalLengthCm'].hist()
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.title('Histogram of Petal Length')

plt.subplot(224)
df['PetalWidthCm'].hist()
plt.xlabel('Petal Width (cm)')
plt.ylabel('Frequency')
plt.title('Histogram of Petal Width')

plt.tight_layout()
plt.figure()

# Create list of colors and class labels
colors = ['red', 'orange', 'blue']
species = ['Iris-virginica', 'Iris-versicolor', 'Iris-setosa']

for i in range(3):
    # filter data on each class
    x = df[df['Species'] == species[i]]
    # plot the scatter plot
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c=colors[i], label=species[i])

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
plt.figure()

for i in range(3):
    # filter data on each class
    x = df[df['Species'] == species[i]]
    # plot the scatter plot
    plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c=colors[i], label=species[i])

plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()
plt.figure()

for i in range(3):
    # filter data on each class
    x = df[df['Species'] == species[i]]
    # plot the scatter plot
    plt.scatter(x['PetalLengthCm'], x['SepalLengthCm'], c=colors[i], label=species[i])

plt.xlabel("Petal Length")
plt.ylabel("Sepal Length")
plt.legend()
plt.figure()

for i in range(3):
    # filter data on each class
    x = df[df['Species'] == species[i]]
    # plot the scatter plot
    plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c=colors[i], label=species[i])

plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.legend()
plt.figure()

# Display the Correlation Matrix
from sklearn.preprocessing import LabelEncoder

# Create a label encoder instance
label_encoder = LabelEncoder()

# Apply label encoding to the 'Species' column
df['Species'] = label_encoder.fit_transform(df['Species'])

# Compute correlations
correlation_matrix = df.corr()
print(correlation_matrix)

corr = df.corr()
# plot the heat map
fig, ax= plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True, ax=ax, cmap="coolwarm")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle

# input data
X = df.drop(columns=['Species'])
# output data
Y = df['Species']
# Split the data for train and test
x_train, x_test, y_train, y_test = train_test_split(X, Y,test_size=0.30)

# Logistic Regression
model = LogisticRegression()
model.fit(x_train,y_train)
print("Logistic Regression Accuracy:", model.score(x_test,y_test)*100)

# Model Training
print(model.fit(x_train.values,y_train.values))

# Print Metric to get performance
print("Accuracy:", model.score(x_test,y_test)*100)

# K-nearest neighbors
model = KNeighborsClassifier()
model.fit(x_train,y_train)
print("K-Nearest Neighbors Accuracy:", model.score(x_test,y_test)*100)

# Model Training
print(model.fit(x_train.values,y_train.values))

# Print Metric to get performance
print("Accuracy:", model.score(x_test,y_test)*100)

# Decision Tree
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
print("Decision Tree Accuracy:", model.score(x_test,y_test)*100)

# Model Training
print(model.fit(x_train.values,y_train.values))

# Print Metric to get performance
print("Accuracy:", model.score(x_test,y_test)*100)

filename = 'saved_model.sav'
try:
    with open(filename,'wb') as file:
        pickle.dump(model,file)
    print("Model Saved Successfully")
except Exception:
    print(f"Error Saving the model: {Exception}")

load_model = pickle.load(open(filename, 'rb'))

# Prediction
prediction_index = load_model.predict([[6.0, 2.2, 4.0, 1.0]])

# Mapping class index to label
class_index_to_label = {
    0: 'Setosa',
    1: 'Versicolor',
    2: 'Virginica'
}

predicted_label = class_index_to_label[prediction_index[0]]
print(predicted_label)

prediction_index=load_model.predict([[4,3,1,5]])
predicted_label = class_index_to_label[prediction_index[0]]
print(predicted_label)


print(x_test.head())