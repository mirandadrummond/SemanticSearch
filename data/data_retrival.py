# Store the dataset in Milvus db
# Declare the required variables like collection name etc.
COLLECTION_NAME = "arxiv_10000"
DIMENSION = 1024
BATCH_SIZE = 128
TOPK = 5
COUNT = 10000

# Connect to the milvus server
from milvus import default_server
from pymilvus import connections, utility, DataType, FieldSchema, CollectionSchema, Collection

default_server.start()
connections.connect(host = "127.0.0.1", port = default_server.listen_port)

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

cohere_client = cohere.Client("enter-your-API-key")#prod key
# Extract embeddings from questions using Cohere
def embed(texts):
    res = cohere_client.embed(texts, model = "embed-english-v3.0", input_type = "search_document")
    return res.embeddings

for batch in tqdm(np.array_split(arxiv, (COUNT/BATCH_SIZE) + 1)):
    #titles = 
    abstracts = batch['abstract'].tolist()
    data = [
        batch['title'].tolist(),
        abstracts,
        batch['label'].tolist(),
        embed(abstracts)
    ]

    collection.insert(data)

# Flush at the end to make sure all rows are sent for indexing
collection.flush()