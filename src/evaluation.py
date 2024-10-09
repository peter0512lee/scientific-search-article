import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from hybrid_search import HybridSearch
from semantic_search import SemanticSearch


def mean_average_precision(relevant_docs, retrieved_docs):
    if not relevant_docs:
        return 0.0

    score = 0.0
    num_hits = 0.0
    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / len(relevant_docs)


def ndcg(relevant_docs, retrieved_docs, k=10):
    dcg = 0.0
    idcg = sum((1.0 / np.log2(i + 2)
               for i in range(min(len(relevant_docs), k))))

    for i, doc in enumerate(retrieved_docs[:k]):
        if doc in relevant_docs:
            dcg += 1.0 / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_search(search_system, test_data, queries):
    map_scores = []
    ndcg_scores = []

    for query, relevant_docs in queries:
        results = search_system.search(query)
        retrieved_docs = [result['id'] for result in results]

        map_scores.append(mean_average_precision(
            relevant_docs, retrieved_docs))
        ndcg_scores.append(ndcg(relevant_docs, retrieved_docs))

    return np.mean(map_scores), np.mean(ndcg_scores)


def optimize_alpha(search_system, test_data, queries, alpha_range=np.arange(0.1, 1.0, 0.1)):
    best_alpha = 0
    best_map = 0
    best_ndcg = 0

    for alpha in alpha_range:
        search_system.alpha = alpha
        map_score, ndcg_score = evaluate_search(
            search_system, test_data, queries)

        print(
            f"Alpha: {alpha:.1f}, MAP: {map_score:.4f}, NDCG: {ndcg_score:.4f}")

        if map_score > best_map:
            best_alpha = alpha
            best_map = map_score
            best_ndcg = ndcg_score

    return best_alpha, best_map, best_ndcg


def main():
    # Load test data
    test_data = pd.read_csv('data/test_data.csv')

    # Create some example queries and relevant documents (you would typically have a separate test set for this)
    queries = [
        ("quantum computing applications", ["1234", "5678", "9101"]),
        ("machine learning in physics", ["2468", "1357", "8024"]),
        ("climate change models", ["1111", "2222", "3333"]),
    ]

    # Initialize search system
    search_system = HybridSearch()
    search_system.load_data('data/train_data.csv')

    print("Evaluating initial performance...")
    initial_map, initial_ndcg = evaluate_search(
        search_system, test_data, queries)
    print(f"Initial MAP: {initial_map:.4f}, NDCG: {initial_ndcg:.4f}")

    print("\nOptimizing alpha parameter...")
    best_alpha, best_map, best_ndcg = optimize_alpha(
        search_system, test_data, queries)

    print(f"\nBest performance:")
    print(
        f"Alpha: {best_alpha:.1f}, MAP: {best_map:.4f}, NDCG: {best_ndcg:.4f}")


if __name__ == "__main__":
    main()
