import pandas as pd
import requests
import numpy as np

# Define the URL for movie data
myurl = "https://liangfgithub.github.io/MovieData/movies.dat?raw=true"

# Fetch the data from the URL
response = requests.get(myurl)
best_genre = pd.read_csv('./data/best_by_genre.csv')
similarity = pd.read_csv('./data/top_similarity_matrix.csv',index_col=0)
top_movie_subset = pd.read_csv('./data/top_viewed_movies.csv')
# Split the data into lines and then split each line using "::"
movie_lines = response.text.split('\n')
movie_data = [line.split("::") for line in movie_lines if line]

# Create a DataFrame from the movie data
movies = pd.DataFrame(movie_data, columns=['movie_id', 'title', 'genres'])
movies['movie_id'] = movies['movie_id'].astype(int)
movies['movie_id_str'] = movies['movie_id'].apply(lambda x: 'm'+str(x))

genres = list(
    sorted(set([genre for genres in movies.genres.unique() for genre in genres.split("|")]))
)



def make_user_rating_series(rating_dict,similarity_matrix):
    n = len(similarity_matrix.index)
    user_rating = pd.Series([np.nan]* n, index=similarity_matrix.index)
    movie_ids = ['m'+ str(key) for key in rating_dict.keys()]
    user_rating.loc[movie_ids] = list(rating_dict.values())
    return user_rating

def myIBCF(user_ratings, similarity_matrix,turn_into_series=True):
    predictions = {}

    if turn_into_series:
        user_ratings = make_user_rating_series(user_ratings,similarity_matrix)

    for movie in user_ratings.index:
        if pd.isna(user_ratings[movie]):

            weighted_sum = np.nansum(similarity_matrix.loc[movie] * user_ratings)
            sum_of_similarities = np.nansum(similarity_matrix.loc[movie].where(~user_ratings.isna()))

            if sum_of_similarities == 0:
                prediction = np.nan
            else:
                prediction = weighted_sum / sum_of_similarities

            predictions[movie] = prediction

    predictions_series = pd.Series(predictions, name='pred_rating')
    sorted_pred_series = pd.DataFrame(predictions_series.sort_values(ascending=False,kind='stable').head(10)).reset_index()
    sorted_pred_series = sorted_pred_series[sorted_pred_series['pred_rating']>0]

    num_recs = len(sorted_pred_series)
    if num_recs < 10:
        sorted_pred_series = pd.concat([sorted_pred_series,top_movie_subset.head(10-num_recs)],axis=0)

    return sorted_pred_series

def get_displayed_movies():
    return movies.head(100)

def get_recommended_movies(new_user_ratings):
    top_predictions = myIBCF(new_user_ratings,similarity)
    merged_movies = movies.merge(top_predictions,left_on='movie_id_str',right_on='index')
    recommended_movies = merged_movies[['movie_id','title','genres']]
    return recommended_movies

def get_popular_movies(genre: str):
    if genre in genres:
        return best_genre[best_genre['genres'] == genre]
    else:
        return movies[10:20]
