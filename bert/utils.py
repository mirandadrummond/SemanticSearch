from transformers import BertTokenizer, BertModel
import torch
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Load pre-trained model tokenizer (vocabulary) and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

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

def generate_embeddings(text):
    """Generate embeddings for a given piece of text using BERT."""
    preprocessed_text = preprocess_text(text)
    encoded_input = tokenizer(preprocessed_text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**encoded_input)
    return output.last_hidden_state[:,0,:].numpy()  # Use the CLS token's embedding

