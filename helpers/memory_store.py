import faiss

from langchain_openai import OpenAIEmbeddings
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS


class MemoryStore:
    def __init__(self, store):
        self.store = store

    @classmethod
    def from_documents(cls, documents):
        embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
        index = faiss.IndexFlatL2(1536)
        docstore = InMemoryDocstore({})
        index_to_docstore_id = {}
        vector_store = FAISS(embedding_model, index, docstore, index_to_docstore_id)

        vector_store.add_documents(documents)

        return cls(vector_store)

    def as_retriever(self):
        return self.store.as_retriever()