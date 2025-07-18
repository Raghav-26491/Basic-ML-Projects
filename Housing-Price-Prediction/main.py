import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams["figure.figsize"]=(20,10)

df=pd.read_csv("Housing Data.csv")
print(df.head(5))

print(df.shape)
print(df.columns)

print(df['area_type'].unique())
print(df['area_type'].value_counts())

df=df.drop(['area_type','society','balcony','availability'],axis='columns')
print(df.shape)

print(df.isnull().sum())

print(df.columns)
df=df.dropna()
print(df.isnull().sum())

print(df.shape)

df.loc[:, 'bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))
print(df.bhk.unique())

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

print(df[~df['total_sqft'].apply(is_float)].head(10))


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

df.total_sqft=df.total_sqft.apply(convert_sqft_to_num)
df=df[df.total_sqft.notnull()]
print(df.head(2))

print(df.loc[30])


df['price_per_sqft']=df['price']*100000/df['total_sqft']
print(df.head())

df_stats = df['price_per_sqft'].describe()
print(df_stats)

df.to_csv("hp.csv",index=False)


df.location = df.location.apply(lambda x: x.strip())
location_stats = df['location'].value_counts(ascending=False)
print(location_stats)

print(location_stats.values.sum())

print(len(location_stats[location_stats>10]))

print(len(location_stats))

print(len(location_stats[location_stats<=10]))

location_stats_less_than_10 = location_stats[location_stats<=10]
print(location_stats_less_than_10)

print(len(df.location.unique()))

df.location=df.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
print(len(df.location.unique()))

print(df.head(10))

print(df[df.total_sqft/df.bhk<300].head())

print(df.shape)

df = df[~(df.total_sqft/df.bhk<300)]
print(df.shape)

print(df.price_per_sqft.describe())

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df=remove_pps_outliers(df)
print(df.shape)

def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (8,8)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price in Lakhs (Indian Rupees)")
    plt.title(location)
    plt.legend()

plot_scatter_chart(df,"Rajaji Nagar")
plt.figure()
plot_scatter_chart(df,"Hebbal")
plt.figure()

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df=remove_bhk_outliers(df)

plot_scatter_chart(df,"Rajaji Nagar")
plt.figure()

plot_scatter_chart(df,"Hebbal")
plt.figure()

matplotlib.rcParams["figure.figsize"] = (8,8)
plt.hist(df.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")
plt.figure()

print(df.bath.unique())

matplotlib.rcParams["figure.figsize"] = (8,8)
plt.hist(df.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")
# plt.show()

print(df[df.bath>10])
print()

print(df[df.bath>df.bhk+2])
print()

print(df.head(2))
print()

df=df.drop(['size','price_per_sqft'],axis='columns')
print(df.head(3))
print()

dummies=pd.get_dummies(df.location)
print(dummies.head(3))

df=pd.concat([df,dummies.drop('other',axis='columns')],axis='columns')
print(df.head(3))

df=df.drop('location',axis='columns')
print(df.head(2))

Y=df.price
print(Y.head(3))
print()

X=df.drop('price',axis='columns')
print(X.head(3))
print()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit, cross_val_score, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

# Training a Linear Regression model
lr_clf = LinearRegression()
lr_clf.fit(X_train, y_train)

# Evaluating the Linear Regression model
lr_score = lr_clf.score(X_test, y_test)
print("Linear Regression Model Score:", lr_score)

# Setting up cross-validation
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

# Performing cross-validation on Linear Regression
lr_cv_scores = cross_val_score(LinearRegression(), X, Y, cv=cv)
print("Linear Regression Cross-Validation Scores:", lr_cv_scores)

def find_best_model_using_gridsearchcv(X, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'copy_X': [True, False],
                'fit_intercept': [True, False],
                'n_jobs': [-1, 1],
                'positive': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['friedman_mse', 'squared_error', 'absolute_error', 'poisson'],
                'splitter': ['best', 'random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

best_model_info = find_best_model_using_gridsearchcv(X, Y)
print("Best Model Information:")
print(best_model_info)

def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns == location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]

predicted_price = predict_price('1st Phase JP Nagar', 1000, 2, 2)
print("Predicted Price:", predicted_price)
print()
A = predict_price('1st Phase JP Nagar',1000, 3, 3)
print("Predicted Price:", A)
print()
B = predict_price('Indira Nagar',1000, 2, 2)
print("Predicted Price:", B)
print()
C = predict_price('Indira Nagar',1000, 3, 3)
print("Predicted Price:", C)
print()
print()

plt.show()