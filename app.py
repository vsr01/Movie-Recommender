# %%
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
import numpy as np
import os

# %%
data_file = 'imdb_top_1000.csv'

# %%
try:
    df = pd.read_csv(data_file)
    print(f'Movie count: {len(df)}')

    df.rename(columns={'Series_Title': 'title'}, inplace=True)

    df['Overview'] = df['Overview'].fillna('')
    df['Genre'] = df['Genre'].fillna('')
    df['Director'] = df['Director'].fillna('')
    df['Star1'] = df['Star1'].fillna('')
    df['Star2'] = df['Star2'].fillna('')
    df['Star3'] = df['Star3'].fillna('')
    df['Star4'] = df['Star4'].fillna('')

    df['tags'] = df['Overview'] + ' ' + \
                 df['Genre'].str.replace(',', ' ') + ' ' + \
                 df['Director'] + ' ' + \
                 df['Star1'] + ' ' + df['Star2'] + ' ' + \
                 df['Star3'] + ' ' + df['Star4']
    
    df['tags'] = df['tags'].apply(lambda x: x.lower().strip())

    df.drop_duplicates(subset=['title'], inplace=True)
    print(f'Movie count after dropping duplicate titles: {len(df)}')

except FileNotFoundError:
    print(f'Error {data_file} not found')
    exit()


# %%
print('Loading the embedding model')
model = SentenceTransformer('all-MiniLM-L6-V2')
print('model loaded')
print('Generating movie embeddings... This might take a moment')
movie_embeddings = model.encode(df['tags'].tolist(), show_progress_bar=True)
print('Embeddings generated')

# %%
def recommend_movies(movie_title, top_n=5):
    movie_title_lower = movie_title.lower().strip()

    movie_idx_candidates = df[df['title'].str.lower()==movie_title_lower].index

    if len(movie_idx_candidates) == 0:
        movie_idx_candidates = df[df['title'].str.lower().str.contains(movie_title_lower, na=False)].index
        if len(movie_idx_candidates) == 0:
            return f'Sorry, movie {movie_title} not found in our database. Please try another title or a partial match'
    
    movie_idx = movie_idx_candidates[0]

    target_embedding = movie_embeddings[movie_idx].reshape(1, -1)
    similarities = cosine_similarity(target_embedding,movie_embeddings).flatten()

    sorted_indices = similarities.argsort()[::-1]

    recommended_movies_list = []
    count = 0
    for idx in sorted_indices:
        if idx == movie_idx:
            continue
        recommended_movies_list.append(df.iloc[idx]['title'])
        count +=1
        if count >= top_n:
            break

    if not recommended_movies_list:
        return 'No recommendations found (perhaps the dataset is too small or the movie is unique)'
    
    return 'Recommended Movies:\n' + '\n'.join([f'- {m}' for m in recommended_movies_list])


# %%
print("Setting up Gradio interface...")
iface = gr.Interface(
    fn=recommend_movies,
    inputs=gr.Textbox(label="Enter a movie title (e.g., The Dark Knight, Inception)"),
    outputs=gr.Textbox(label="Recommendations"),
    title="IMDB Top 1000 Movie Recommender (Text Embeddings)",
    description="Enter a movie title from the IMDB Top 1000 to get recommendations based on content similarity (overview, genre, director, stars).",
    examples=[
        "The Shawshank Redemption",
        "The Dark Knight",
        "Inception",
        "Pulp Fiction",
        "Spirited Away",
        "Interstellar",
        "Forrest Gump"
    ]
)

if __name__ == "__main__":
    print("Gradio interface starting. Open your browser to the URL provided.")
    iface.launch(share=True)

# %%



