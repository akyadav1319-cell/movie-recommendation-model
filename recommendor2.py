import pandas as pd
df=pd.read_csv('ml-latest-small/ratings.csv')
matrix=df.pivot_table(index='userId',columns='movieId',values='rating')
from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(matrix.T)
table=pd.DataFrame(similarity)
print(table.head())