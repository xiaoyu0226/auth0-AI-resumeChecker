from opensearchpy import OpenSearch
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import BaseVectorStore
from langchain.schema import Document

"""
Initialize the OpenSearch vector store
client: OpenSearch client instance
index_name: The name of the OpenSearch index (resumes in this project)
embedding_model: The embedding model used for vectorization (use OpenAIEmbeddings)
"""
class OpenSearchVectorStore(BaseVectorStore):
    def __init__(self, client, index_name, embedding_model):
        self.client = client
        self.index_name = index_name
        self.embedding_model = embedding_model

    """
    Unlike FAISS, you need to self define a function to:
    1. embed the document
    2. match orignal document page_content with embedding
    3. index the document + embedding
    """
    def add_documents(self, documents):
        for doc in documents:
            # Embed the document text to generate the embedding
            embedding = self.embedding_model.embed_documents([doc.page_content])[0]
            
            # Prepare the document body to store in OpenSearch
            doc_body = {
                "text": doc.page_content,  # Document text
                "embedding": embedding.tolist(),  # Embedding as list (convert numpy array to list)
                "metadata": doc.metadata  # Metadata (id, access level), we NEED ID FOR FGA AUTHENTICATION
            }
            
            # Index the document into OpenSearch
            self.client.index(index=self.index_name, body=doc_body)
    
    """
    Unlike FAISS, you need to self define a function for document retrieval
    Retrieve documents based on a query by performing a k-NN search.
    query: The query string to search for similar documents
    """
    def _get_relevant_documents(self, query):
        # Get the embedding of the query
        query_embedding = self.embedding_model.embed_query(query)

        # Perform k-NN search in OpenSearch
        response = self.client.search(
            index=self.index_name,
            body={
                "size": 4,    # Match FAISS default k value
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": query_embedding.tolist(),
                            "k": 4    # Match FAISS default value 
                        }
                    }
                }
            }
        )

        # Map OpenSearch response to langchain Document objects with metadata
        opensearch_hits = response['hits']['hits']
        documents = []
        for hit in opensearch_hits:
            # Extract the document content (text) and metadata from OpenSearch
            text = hit["_source"].get("text", "")
            metadata = hit["_source"].get("metadata", {})
            
            # Ensure metadata contains necessary fields (id, access level, etc.)
            document = Document(page_content=text, metadata=metadata)
            documents.append(document)
        return documents


class OpenSearchStore:
    def __init__(self, store):
        self.store = store

    """
    Initialize the OpenSearch vector store and add documents to OpenSearch
    documents: A list of Document objects to add to the vector store
    Mimicing FAISS style
    """
    @classmethod
    def from_documents(cls, documents):
        # OpenSearch client setup
        client = OpenSearch(
            hosts=[{'host': 'your-opensearch-endpoint', 'port': 9200}],
            http_auth=('your-username', 'your-password'),  # Put in your own authentication info
            use_ssl=True,
            verify_certs=True
        )

        # Define the OpenSearch index name
        index_name = "resumes"

        # Check if the index exists, create it if not
        if not client.indices.exists(index=index_name):
            client.indices.create(index=index_name, body={
                "settings": {
                    "index": {
                        "knn": True,  # Enable k-NN search
                        "number_of_shards": 1,
                        "number_of_replicas": 1
                    }
                },
                "mappings": {
                    "properties": {
                        "text": {"type": "text"},
                        "embedding": {
                            "type": "dense_vector",  # Embedding field as dense vector
                            "dims": 1536  # Dimension of embeddings matching OpenAI's embeddings
                        }
                    }
                }
            })

        embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
        vector_store = OpenSearchVectorStore(client, index_name, embedding_model)    # note openSearch handles storage for document directly in an index
        
        vector_store.add_documents(documents)

        return cls(vector_store)

    def as_retriever(self):
        return self.store