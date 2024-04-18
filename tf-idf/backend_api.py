from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load data
df = pd.read_csv('data/arxiv100.csv')
df = df.head(1000)

# Combine the title and abstract columns to create the documents
df['documents'] = df['title'] + ' ' + df['abstract']

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text

# Apply preprocessing
df['preprocessed_documents'] = df['documents'].apply(preprocess_text)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['preprocessed_documents'])

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    
    # Preprocess query
    preprocessed_query = preprocess_text(query)
    
    # TF-IDF vectorization for query
    query_vector = tfidf_vectorizer.transform([preprocessed_query])
    
    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Sort documents by similarity scores
    result_indices = cosine_similarities.argsort()[::-1]
    
    # Get sorted titles based on similarity scores
    sorted_titles = [df.iloc[index]['title'] for index in result_indices]
    
    return jsonify(sorted_titles)

if __name__ == '__main__':
    app.run(debug=True)
