import streamlit as st
import requests
import os
from st_milvus_connection import MilvusConnection
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from bert.create_db import search_similar_texts

# Streamlit UI
st.title('Semantic Search Engine (BERT)')

# Search input
query = st.text_input('Enter an article to search:', '')

TOPK = st.slider('Select the number of results to display:', 1, 5, 1)

# Search button
if st.button('Search'):
    results = search_similar_texts(query, TOPK)

    st.header('Search Results:')
    
    if results:
        for hits in results:  # Iterate through each set of hits (likely just one if TOPK is per the entire search)
            for idx, hit in enumerate(hits, 1):
                title = hit.entity.get("title")
                abstract = hit.entity.get("abstract")
                score = round(hit.distance, 3)
                st.subheader(f"{idx}. {title} (Score: {score})")
                st.write(abstract)
    else:
        st.write('No matching articles found.')
else:
    st.write('Please enter a query and press the search button.')