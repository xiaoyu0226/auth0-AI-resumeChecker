from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from helpers.memory_store import MemoryStore
from openfga_sdk.client.models import ClientBatchCheckItem
from helpers.read_documents import read_documents
from helpers.fga_retriever import FGARetriever

load_dotenv()


class RAG:
    def __init__(self):
        documents = read_documents()
        self.vector_store = MemoryStore.from_documents(documents)
        self.prompt = ChatPromptTemplate.from_template(
            """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\\nQuestion: {question}\\nContext: {context}\\nAnswer:"""
        )

        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def query(self, user_id: str, question: str):
        chain = (
            {
                "context": FGARetriever(
                    retriever=self.vector_store.as_retriever(),
                    build_query=lambda doc: ClientBatchCheckItem(
                        user=f"user:{user_id}",
                        object=f"resume:{doc.metadata.get('id')}",
                        relation="viewer",
                    ),
                ),
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        return chain.invoke(question)


if __name__ == "__main__":
    rag = RAG()

    question = "What are Xiao's past experiences?"

    # Xiao will have access to Xiao's resume
    response = rag.query("Xiao", question)
    print("Response to Xiao:", response)

    # All has access to public resume template
    # response = rag.query("Emily", question)
    # print("Response to Emily:", response)

    # Emily will not have access to Xiao's resume
    response = rag.query("Emily", question)
    print("Response to Emily:", response)