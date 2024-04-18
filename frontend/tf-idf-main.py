import streamlit as st
import requests

# Streamlit UI
st.title('Semantic Search Engine (TF-IDF)')

# Search input
query = st.text_input('Enter an article to search:', '')

# Search button
if st.button('Search'):
    response = requests.get(f'http://127.0.0.1:5000/search?query={query}')
    
    if response.status_code == 200:
        results = response.json()
        
        st.header('Top 10 Matching Articles:')
        
        if len(results) > 0:
            for idx, title in enumerate(results[:10], 1):
                st.write(f"{idx}. {title}")
        else:
            st.write('No matching articles found.')
    else:
        st.write('Error fetching results.')
