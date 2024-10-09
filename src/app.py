from flask import Flask, render_template, request, jsonify
from hybrid_search import HybridSearch

app = Flask(__name__, template_folder='../templates')

# Initialize the search system
search_system = HybridSearch(es_username='elastic', es_password='yPks5m1u')
search_system.load_data('data/train_data.csv')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    results = search_system.search(query, top_k=10)
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)
