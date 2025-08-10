import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from xgboost import XGBClassifier
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data=pd.read_csv("D:\\Fast\\python\\Intern\\task5\\movies.csv")
print(data.head())
print(data.shape)

selected_features=['genres','director','keywords','tagline','cast']

for features in selected_features:
    data[features]=data[features].fillna('')


combined_features=data['genres']+' '+data['director']+' '+data['keywords']+' '+data['tagline']+' '+data['cast']

vectorizer=TfidfVectorizer()
features=vectorizer.fit_transform(combined_features)

similarity=cosine_similarity(features)
print(similarity)

movie_name=input("Enter your favourite movie name: ")

list_of_all_titles = data['title'].tolist()
print(list_of_all_titles)

find_close_match=difflib.get_close_matches(movie_name,list_of_all_titles)
print(find_close_match)

close_match=find_close_match[0]

index_of_the_movie = data[data.title == close_match]['index'].values[0]

similarity_score = list(enumerate(similarity[index_of_the_movie]))
print(len(similarity_score))
sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = data[data.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1
