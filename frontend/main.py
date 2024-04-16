import streamlit as st
import pandas as pd

# Sample data for research papers
data = {
    'title': [
        'Machine Learning Techniques in Natural Language Processing',
        'Advancements in Quantum Computing Algorithms',
        'CRISPR Technology and Genome Editing',
        'Climate Change Impact on Biodiversity',
        'Artificial Intelligence in Healthcare Diagnostics'
    ],
    'description': [
        'Exploring the latest machine learning models for NLP tasks.',
        'Recent developments and breakthroughs in quantum computing algorithms.',
        'Understanding CRISPR technology and its applications in genome editing.',
        'Assessing the effects of climate change on global biodiversity.',
        'Utilizing AI for improving healthcare diagnostics and patient care.'
    ]
}


df = pd.DataFrame(data)

# Search function
def search(query, data):
    return data[data['title'].str.contains(query, case=False)]

# Streamlit UI
st.title('Semantic Search Engine')

# Search input
query = st.text_input('Enter an article to search:', '')

# Search button
if st.button('Search'):
    results = search(query, df)
    st.write(results)
