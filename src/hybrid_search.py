import pandas as pd
from full_text_search import FullTextSearch
from semantic_search import SemanticSearch


class HybridSearch:
    def __init__(self, index_name='arxiv_articles', model_name='all-MiniLM-L6-v2', es_host='http://localhost:9200', es_username=None, es_password=None):
        self.full_text_search = FullTextSearch(
            host=es_host, index_name=index_name, username=es_username, password=es_password)
        self.semantic_search = SemanticSearch(model_name)
        self.data = None

    def load_data(self, data_path):
        self.data = pd.read_csv(data_path)
        self.full_text_search.create_index()
        self.full_text_search.index_data(data_path)
        self.semantic_search.load_and_embed_data(data_path)

    def search(self, query, top_k=10, alpha=0.5):
        # Perform full-text search
        full_text_results = self.full_text_search.search(query, size=top_k*2)
        full_text_ids = [result['id'] for result in full_text_results]

        # Perform semantic search
        semantic_results = self.semantic_search.search(query, top_k=top_k*2)
        semantic_ids = [result['id'] for result in semantic_results]

        # Combine results
        combined_results = {}
        for i, doc_id in enumerate(full_text_ids):
            combined_results[doc_id] = alpha * (1 - i / len(full_text_ids))

        for i, doc_id in enumerate(semantic_ids):
            if doc_id in combined_results:
                combined_results[doc_id] += (1 - alpha) * \
                    (1 - i / len(semantic_ids))
            else:
                combined_results[doc_id] = (
                    1 - alpha) * (1 - i / len(semantic_ids))

        # Sort and return top results
        sorted_results = sorted(combined_results.items(
        ), key=lambda x: x[1], reverse=True)[:top_k]

        final_results = []
        for doc_id, score in sorted_results:
            article = self.data[self.data['id'] == doc_id].iloc[0]
            final_results.append({
                'id': doc_id,
                'title': article['title'],
                'abstract': article['abstract'],
                'score': score
            })

        return final_results


def main():
    hybrid_search = HybridSearch(
        es_host='http://localhost:9200',
        es_username='elastic',  # Replace with your actual username
        es_password='yPks5m1u'  # Replace with your actual password
    )

    print("Loading and indexing data...")
    hybrid_search.load_data('data/train_data.csv')

    query = "quantum computing applications in machine learning"
    print(f"\nPerforming hybrid search for: '{query}'")
    results = hybrid_search.search(query)

    print("\nTop 5 hybrid search results:")
    for i, result in enumerate(results[:5], 1):
        print(f"{i}. {result['title']} (Score: {result['score']:.4f})")


if __name__ == "__main__":
    main()
