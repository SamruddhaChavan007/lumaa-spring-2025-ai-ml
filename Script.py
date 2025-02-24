# Name: Samruddha Chavan
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_dataset(file_path):
    """
    Load the dataset from a CSV file.
    Ensure that missing descriptions are replaced with empty strings.
    """
    try:
        df = pd.read_csv(file_path)
        # Fill missing descriptions with empty strings
        df['description'] = df['description'].fillna('')
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

def build_tfidf_matrix(descriptions):
    """
    Build a TF-IDF matrix from the text descriptions.
    Uses English stop words for better vector quality.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    return tfidf_matrix, vectorizer

def get_recommendations(query, tfidf_matrix, df, vectorizer, top_n=5):
    """
    Compute the cosine similarity between the query and each movie description,
    then return the top_n movies with the highest similarity scores.
    """
    # Vectorize the input query
    query_vec = vectorizer.transform([query])
    # Compute cosine similarity between the query and all movie descriptions
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    # Get indices of the top_n most similar movies
    top_indices = cosine_sim.argsort()[-top_n:][::-1]
    # Extract corresponding movie details and their similarity scores
    recommendations = df.iloc[top_indices]
    scores = cosine_sim[top_indices]
    return recommendations, scores

def main():
    # Check if a query was provided via the command line
    if len(sys.argv) < 2:
        print("Usage: python recommend.py \"<your movie preference query>\"")
        sys.exit(1)
    
    query = sys.argv[1]
    # Path to the dataset (adjust if necessary)
    dataset_path = "netflix_titles.csv"
    
    print("Loading dataset...")
    df = load_dataset(dataset_path)
    
    print("Building TF-IDF matrix...")
    tfidf_matrix, vectorizer = build_tfidf_matrix(df['description'])
    
    print("Computing recommendations...")
    recommendations, scores = get_recommendations(query, tfidf_matrix, df, vectorizer)
    
    print("\nTop recommendations for your query:")
    for idx, (i, row) in enumerate(recommendations.iterrows()):
        print(f"{idx+1}. Title: {row['title']}  (Similarity Score: {scores[idx]:.4f})")
    
if __name__ == "__main__":
    main()
