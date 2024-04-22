import pandas as pd
import numpy as np
import time
from tqdm import tqdm

import nltk
nltk.download('stopwords')

from milvus import default_server, debug_server
from pymilvus import connections, utility, DataType, FieldSchema, CollectionSchema, Collection

from bert.utils import generate_embeddings

# Variable declaration
COLLECTION_NAME = "arxiv_10000"
DIMENSION = 768
BATCH_SIZE = 128
TOPK = 5
COUNT = 10000

print('loading data')

arxiv_e = pd.read_csv('/Users/mirandadrummond/VSCode/SemanticSearch/bert/arxiv100_embedded_1_prep.csv')
def convert(data_string):
    # Remove square brackets and split the string into individual numbers
    clean_string = data_string.replace('[', '').replace(']', '').strip()
    data_list = [float(item) for item in clean_string.split()]

    # Create a NumPy array from the list of floats
    data_array = np.array(data_list)

    return data_array

arxiv_e['embeddings'] = arxiv_e['embeddings'].apply(convert)

##############################################
# Setting up milvus and creating collection  #
##############################################
# debug_server.stop()
debug_server.cleanup()
debug_server.start()
connections.connect(host = "127.0.0.1", port = default_server.listen_port)

print('server started')

utility.get_server_version()

# Check if the collection is already available, if yes, then drop it and create a new one
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

# object should be inserted in the format of (title, date, location, speech embedding)
fields = [
    FieldSchema(name = "id", dtype = DataType.INT64, is_primary = True, auto_id = True),
    FieldSchema(name = "title", dtype = DataType.VARCHAR, max_length = 800),
    FieldSchema(name = "abstract", dtype = DataType.VARCHAR, max_length = 9000),
    FieldSchema(name = "label", dtype = DataType.VARCHAR, max_length = 20),
    FieldSchema(name = "embedding", dtype = DataType.FLOAT_VECTOR, dim = DIMENSION)
]
schema = CollectionSchema(fields = fields)
collection = Collection(name = COLLECTION_NAME, schema = schema)

# Create the index
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 100},
}
collection.create_index(field_name = "embedding", index_params = index_params)
collection.load()

for batch in tqdm(np.array_split(arxiv_e, 10), desc="Processing batches"):
    data = [
        batch['title'].tolist(),
        batch['abstract'].tolist(),
        batch['label'].tolist(),
        batch['embeddings'].tolist()
    ]

    collection.insert(data)

collection.flush()

def search_similar_texts(query_text= "tell me about astrophysics", TOPK=5):
    search_data = generate_embeddings(query_text)

    start = time.time()
    res = collection.search(
        data = search_data,  # Embed search value
        anns_field = "embedding",  # Search across embeddings
        param = {"metric_type": "L2",
                "params": {"nprobe": 20}},
        limit = TOPK,  # Limit to top_k results per search
        output_fields = ["title","abstract"]  # Include title field in result 
    )

    end = time.time()

    for hits_i, hits in enumerate(res):
        print("Query:", query_text[hits_i])
        #print("Abstract:", search_terms[hits_i])
        print("Search Time:", end-start)
        print("Results:\n")
        for hit in hits:
            print( hit.entity.get("title"), "----", round(hit.distance, 3))
            print()
            print( hit.entity.get("abstract"), "----", round(hit.distance, 3))
            print()
        print()

    return res
        

    # default_server.stop()