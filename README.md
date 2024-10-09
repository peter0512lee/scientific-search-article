# Scientific Article Hybrid Search System

## Project Overview

This project implements a hybrid search system for scientific articles, combining traditional full-text search with modern embedding-based semantic search. The system is designed to provide more relevant and diverse search results by leveraging the strengths of both search methods.

## Features

- Full-text search using Elasticsearch
- Semantic search using sentence embeddings (SentenceTransformer)
- Hybrid search combining both methods
- Evaluation metrics (MAP and NDCG) for search quality assessment
- Parameter optimization for improved performance
- Simple web interface for user interaction

## Technologies Used

- Python 3.8+
- Elasticsearch
- Flask
- SentenceTransformer
- pandas
- scikit-learn
- NumPy

## Project Structure

```bash
scientific_article_search/
│
├── data/
│   ├── train_data.csv
│   └── test_data.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── full_text_search.py
│   ├── semantic_search.py
│   ├── hybrid_search.py
│   ├── evaluation.py
│   └── app.py
│
├── templates/
│   └── index.html
│
├── requirements.txt
└── README.md
```

## Installation and Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/scientific_article_search.git
   cd scientific_article_search
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Install and run Elasticsearch (follow instructions from the official Elasticsearch documentation)

5. Prepare the data:

   ```bash
   python src/data_preprocessing.py
   ```

## Usage

1. Run the Flask application:

   ```bash
   python src/app.py
   ```

2. Open a web browser and navigate to `http://localhost:5000`

3. Enter a search query in the provided input field and click "Search" to see the results

## Evaluation and Optimization

To evaluate and optimize the search system:

```bash
python src/evaluation.py
```

This script will output the initial performance metrics and the optimized parameters for the hybrid search system.

## Future Improvements

- Implement more advanced NLP techniques (e.g., named entity recognition, topic modeling)
- Expand the dataset to cover a broader range of scientific domains
- Develop a more sophisticated user interface with advanced search options
- Implement user feedback mechanisms to continuously improve search results
- Explore cloud deployment options for scalability

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License.
