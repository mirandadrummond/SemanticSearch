import streamlit as st
import requests
import os
from st_milvus_connection import MilvusConnection
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from bert.create_db import search_similar_texts

os.environ["milvus_uri"] = os.getenv("MILVUS_URI")
os.environ["milvus_token"] = os.getenv("MILVUS_TOKEN")

conn = st.connection("milvus", type=MilvusConnection)

# Streamlit UI
st.title('Semantic Search Engine (BERT)')

# Search input
query = st.text_input('Enter an article to search:', '')

# Search button
if st.button('Search'):
    TOPK = 5
    querytext = [query]
    results = search_similar_texts(querytext, TOPK)
    
    
    st.header('Top 5 Matching Articles:')
        
    if len(results) > 0:
        for idx, title in enumerate(results[:10], 1):
            st.write(f"{idx}. {title}")
    else:
        st.write('No matching articles found.')
else:
    st.write('Error fetching results.')
