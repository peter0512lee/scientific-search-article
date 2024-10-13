import pandas as pd
from elasticsearch import Elasticsearch, helpers


class FullTextSearch:
    def __init__(self, host='http://localhost:9200', index_name='arxiv_articles', username=None, password=None):
        self.es = Elasticsearch(
            [host],
            basic_auth=(username, password) if username and password else None
        )
        self.index_name = index_name

    def create_index(self):
        index_body = {
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "title": {"type": "text"},
                    "abstract": {"type": "text"},
                    "categories": {"type": "keyword"},
                    "full_text": {"type": "text"}
                }
            }
        }
        self.es.indices.create(index=self.index_name,
                               body=index_body, ignore=400)

    def index_data(self, data_path):
        df = pd.read_csv(data_path)
        actions = [
            {
                "_index": self.index_name,
                "_id": row['id'],
                "_source": row.to_dict()
            }
            for _, row in df.iterrows()
        ]
        helpers.bulk(self.es, actions)

    def search(self, query, size=10):
        body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^2", "abstract", "full_text"]
                }
            }
        }
        result = self.es.search(index=self.index_name, body=body, size=size)
        return [hit['_source'] for hit in result['hits']['hits']]


def main():
    # Replace with your actual credentials
    fts = FullTextSearch(
        host='http://localhost:9200',
        username='',
        password=''
    )

    # Create index and index data
    fts.create_index()
    fts.index_data('data/train_data.csv')

    # Example search
    query = "quantum computing"
    results = fts.search(query)

    print(f"Search results for '{query}':")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']}")


if __name__ == "__main__":
    main()
