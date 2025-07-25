import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
import numpy as np
import os # For checking file existence

# --- 1. Use the provided Movie Dataset ---
DATA_FILE = 'imdb_top_1000.csv'
EMBEDDINGS_FILE = 'imdb_top_1000_movie_embeddings.npy' # Define the file name for embeddings
DF_PROCESSED_FILE = 'imdb_top_1000_processed_movies_df.csv' # Define file name for processed DataFrame

# --- 2. Load and Prepare Data for Embeddings ---
df = None # Initialize df outside try-except
try:
    if os.path.exists(DF_PROCESSED_FILE):
        print(f"Loading processed DataFrame from '{DF_PROCESSED_FILE}'...")
        df = pd.read_csv(DF_PROCESSED_FILE)
    else:
        print(f"'{DF_PROCESSED_FILE}' not found. Processing raw data from '{DATA_FILE}'.")
        df = pd.read_csv(DATA_FILE)
        print(f"Loaded {len(df)} movies from '{DATA_FILE}'.")

        # Rename 'Series_Title' to 'title' for consistency
        df.rename(columns={'Series_Title': 'title'}, inplace=True)

        # Fill any potential NaN values in text columns with empty strings
        df['Overview'] = df['Overview'].fillna('')
        df['Genre'] = df['Genre'].fillna('')
        df['Director'] = df['Director'].fillna('')
        df['Star1'] = df['Star1'].fillna('')
        df['Star2'] = df['Star2'].fillna('')
        df['Star3'] = df['Star3'].fillna('')
        df['Star4'] = df['Star4'].fillna('')

        # Create a 'tags' column by combining relevant text features
        df['tags'] = df['Overview'] + ' ' + \
                     df['Genre'].str.replace(',', ' ') + ' ' + \
                     df['Director'] + ' ' + \
                     df['Star1'] + ' ' + df['Star2'] + ' ' + \
                     df['Star3'] + ' ' + df['Star4']

        df['tags'] = df['tags'].apply(lambda x: x.lower().strip())

        # Drop duplicate movie titles
        df.drop_duplicates(subset=['title'], inplace=True, keep='first') # Keep first occurrence
        print(f"After dropping duplicates, {len(df)} unique movies remain.")

        # Save the processed DataFrame to avoid reprocessing text in future runs
        df.to_csv(DF_PROCESSED_FILE, index=False)
        print(f"Processed DataFrame saved to '{DF_PROCESSED_FILE}'.")

except FileNotFoundError:
    print(f"Error: Required data file(s) not found. Ensure '{DATA_FILE}' is in the same directory.")
    exit()

# --- 3. Choose and Load a Free Text Embedding Model ---
# This model will always be loaded because it's needed for the `encode` step if embeddings are not found.
# And if the user inputs a new movie title to embed (which our current system doesn't do, but a future one might).
print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

# --- 4. Generate Movie Embeddings (or Load Them) ---
movie_embeddings = None # Initialize movie_embeddings outside try-except
if os.path.exists(EMBEDDINGS_FILE):
    print(f"Loading embeddings from '{EMBEDDINGS_FILE}'...")
    movie_embeddings = np.load(EMBEDDINGS_FILE)
    print("Embeddings loaded.")
else:
    print("Embeddings file not found. Generating movie embeddings...")
    movie_embeddings = model.encode(df['tags'].tolist(), show_progress_bar=True)
    np.save(EMBEDDINGS_FILE, movie_embeddings) # Save the generated embeddings
    print(f"Embeddings generated and saved to '{EMBEDDINGS_FILE}'.")

# Consistency check: Ensure the number of embeddings matches the number of movies
if len(movie_embeddings) != len(df):
    print("Mismatch: Number of embeddings does not match number of movies in DataFrame.")
    print("Re-generating embeddings to ensure consistency.")
    movie_embeddings = model.encode(df['tags'].tolist(), show_progress_bar=True)
    np.save(EMBEDDINGS_FILE, movie_embeddings) # Overwrite with correct embeddings
    print(f"Embeddings re-generated and saved to '{EMBEDDINGS_FILE}'.")


# --- 5. Implement Similarity Calculation (within the recommendation function) ---
# --- 6. Develop the Recommendation Logic ---
def recommend_movies(movie_title, top_n=5):
    movie_title_lower = movie_title.lower().strip() # Normalize input

    # Find the index of the input movie.
    movie_idx_candidates = df[df['title'].str.lower() == movie_title_lower].index

    if len(movie_idx_candidates) == 0:
        # Try a less strict match (e.g., contains) if exact match fails
        movie_idx_candidates = df[df['title'].str.lower().str.contains(movie_title_lower, na=False)].index
        if len(movie_idx_candidates) == 0:
            return f"Sorry, movie '{movie_title}' not found in our database. Please try another title or a partial match."

    movie_idx = movie_idx_candidates[0] # Get the first match if multiple exist

    # Calculate cosine similarity between the input movie and all other movies
    target_embedding = movie_embeddings[movie_idx].reshape(1, -1)
    similarities = cosine_similarity(target_embedding, movie_embeddings).flatten()

    # Get the indices of movies sorted by similarity in descending order
    sorted_indices = similarities.argsort()[::-1]

    recommended_movies_list = []
    count = 0
    for idx in sorted_indices:
        if idx == movie_idx: # Skip the movie itself
            continue
        recommended_movies_list.append(df.iloc[idx]['title'])
        count += 1
        if count >= top_n:
            break

    if not recommended_movies_list:
        return "No recommendations found (perhaps the dataset is too small or the movie is unique)."

    return "Recommended Movies:\n" + "\n".join([f"- {m}" for m in recommended_movies_list])

# --- 7. Build a Gradio Frontend ---
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

# To run locally:
if __name__ == "__main__":
    print("Gradio interface starting. Open your browser to the URL provided.")
    iface.launch(share=True)