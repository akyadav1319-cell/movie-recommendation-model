import pandas as pd
movies=pd.read_csv('ml-latest-small/movies.csv')
ratings=pd.read_csv('ml-latest-small/ratings.csv')
print(movies.head())
print(ratings.head())
print(movies.shape)
print(ratings.shape)
movie_matrix=ratings.pivot_table(index='userId',columns='movieId',values='rating')
print(movie_matrix.shape)
print(movie_matrix.head())
movie_matrix_filled=movie_matrix.fillna(0)
from sklearn.metrics.pairwise import cosine_similarity
similarity= cosine_similarity(movie_matrix_filled.T)
print(similarity.shape)
similarity_table=pd.DataFrame(similarity)
print(similarity_table)
def recommend(movie_name):
    # Step 1 - find the movieId for the movie name we type
    movie_id = movies[movies['title'] == movie_name]['movieId'].values[0]
    
    # Step 2 - find where this movie sits in our matrix
    movie_idx = list(movie_matrix_filled.columns).index(movie_id)
    
    # Step 3 - grab that movie's row from similarity matrix
    # this row has scores against every other movie
    scores = list(enumerate(similarity[movie_idx]))
    
    # Step 4 - sort highest to lowest
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    # Step 5 - skip first result (it's the movie itself, always 1.0)
    scores = scores[1:6]
    
    # Step 6 - convert indices back to movie names
    movie_indices = [i[0] for i in scores]
    recommended_ids = movie_matrix_filled.columns[movie_indices]
    recommended_titles = movies[movies['movieId'].isin(recommended_ids)]['title']
    
    return recommended_titles

# TEST IT 🎉
print(recommend('Toy Story (1995)'))