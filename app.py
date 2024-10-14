from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

# Load dataset and initialize TF-IDF vectorizer
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data
stop_words = stopwords.words('english')

# Create TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=10000)
X_tfidf = vectorizer.fit_transform(documents)

# Perform LSA using TruncatedSVD
svd = TruncatedSVD(n_components=100)  # Reduce to 100 dimensions
X_lsa = svd.fit_transform(X_tfidf)

def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    # Transform the query using the same TF-IDF vectorizer and LSA
    query_tfidf = vectorizer.transform([query])
    query_lsa = svd.transform(query_tfidf)

    # Compute cosine similarity between query and all documents
    similarities = cosine_similarity(query_lsa, X_lsa)[0]
    
    # Get indices of top 5 most similar documents
    top_indices = np.argsort(similarities)[-5:][::-1]
    
    # Retrieve the top 5 documents and their similarity scores
    top_documents = [documents[i] for i in top_indices]
    top_similarities = similarities[top_indices]

    return top_documents, top_similarities.tolist(), top_indices.tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices})

if __name__ == '__main__':
    app.run(debug=True)
