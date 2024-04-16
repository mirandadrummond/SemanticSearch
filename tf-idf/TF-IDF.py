import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv('arxiv100.csv')
df = df.head(1000)
# Query
query = "Roche-lobe overflow channel and composed of a main-sequence dwarf "

# Combine the title and abstract columns to create the documents
df['documents'] = df['title'] + ' ' + df['abstract']

# Preprocessing function
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    # Join tokens back into string
    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text

# Apply preprocessing
df['preprocessed_documents'] = df['documents'].apply(preprocess_text)

# Append the query to the preprocessed documents
preprocessed_documents = df['preprocessed_documents'].tolist()
preprocessed_documents.append(preprocess_text(query))

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_documents)

# Calculate cosine similarity between the query and documents
cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

# Sort documents by similarity scores
result_indices = cosine_similarities.argsort()[::-1]

# Get sorted titles based on similarity scores
sorted_titles = [df.iloc[index]['title'] for index in result_indices]

print("Sorted Titles Based on Cosine Similarity:")
print(sorted_titles)