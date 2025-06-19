import pandas as pd
from tqdm import tqdm

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


books = None
isbns = None
descriptions = None

documents = None

ollama_embedding = None
db_books = None


def read_dataset():
    global books, isbns, descriptions
    books = pd.read_csv("books_cleaned.csv")

    # isbns = books["isbn13"].tolist()[:10]
    # descriptions = books["description"].tolist()[:10]

    isbns = books["isbn13"].tolist()
    descriptions = books["description"].tolist()


def create_documents():
    global documents
    documents = [
        Document(page_content=description, metadata={"isbn13": isbn})
        for isbn, description in zip(isbns, descriptions)
    ]


def create_vector_db(batch_size=8):
    global ollama_embedding
    ollama_embedding = OllamaEmbeddings(model="bge-m3", num_thread=8, num_gpu=-1)

    global db_books
    db_books = Chroma(
        collection_name="books",
        embedding_function=ollama_embedding,
        persist_directory="./chroma_db",
    )

    for i in tqdm(range(0, len(documents), batch_size)):
        batch = documents[i : i + batch_size]
        db_books.add_documents(batch)

    # db_books.persist()


def test_vector_db(query: str = None):
    global db_books
    db_books = Chroma(
        collection_name="books",
        embedding_function=ollama_embedding,
        persist_directory="./chroma_db",
    )

    results = db_books.similarity_search(query=query, k=10)
    books_list = []

    for result in results:
        books_list.append(result.metadata["isbn13"])

    print(books[books["isbn13"].isin(books_list)])


if __name__ == "__main__":
    read_dataset()

    create_documents()

    create_vector_db(batch_size=128)

    query = "A book about World War One"
    test_vector_db(query=query)
