import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticSearch:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.data = None
        self.embeddings = None

    def load_and_embed_data(self, data_path):
        self.data = pd.read_csv(data_path)
        self.embeddings = self.model.encode(
            self.data['full_text'].tolist(), show_progress_bar=True)

    def search(self, query, top_k=10):
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                'id': self.data.iloc[idx]['id'],
                'title': self.data.iloc[idx]['title'],
                'abstract': self.data.iloc[idx]['abstract'],
                'similarity': similarities[idx]
            })

        return results


def main():
    semantic_search = SemanticSearch()

    # Load and embed data
    print("Loading and embedding data...")
    semantic_search.load_and_embed_data('data/train_data.csv')

    # Example search
    query = "quantum computing applications"
    print(f"\nPerforming semantic search for: '{query}'")
    results = semantic_search.search(query)

    print("\nTop 5 semantically similar articles:")
    for i, result in enumerate(results[:5], 1):
        print(
            f"{i}. {result['title']} (Similarity: {result['similarity']:.4f})")


if __name__ == "__main__":
    main()
