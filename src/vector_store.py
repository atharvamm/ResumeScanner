# Reference : https://python.langchain.com/v0.1/docs/integrations/vectorstores/pgvector/
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from src.load_env import get_root_path,load_env_file
import psycopg


def get_connection(new_conn = False):
    # root_path = get_root_path()
    user_name,password = os.getenv("USER"),os.getenv("PASSWORD")
    host,port = os.getenv("HOST"), os.getenv("PORT")
    db_name = os.getenv("DB_NAME")
    connection_str = "postgresql+psycopg://{}:{}@{}:{}/{}".format(user_name,password, host, port, db_name)

    if new_conn:
        conn = psycopg.connect(
        dbname=db_name,
        user=user_name,
        password=password,
        host=host,
        port=port)

        cur = conn.cursor()

        cur.execute("TRUNCATE TABLE {}.public.{} CASCADE;".format(db_name,"langchain_pg_embedding"))
        cur.execute("TRUNCATE TABLE {}.public.{} CASCADE;".format(db_name,"langchain_pg_collection"))



        conn.commit()
        cur.close()
        conn.close()

    return connection_str

def get_embedding(doc,embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")):
    if not(isinstance(doc,list) or isinstance(doc,tuple)):
        doc = list(doc)
    return embedding.embed_documents(doc)

def get_vectorstore(collection_name = "resume_coll",embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"), new_conn = False):
    connection = get_connection(new_conn)

    vectorstore = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
    )

    return vectorstore


# # # To Drop Table
# # vectorstore.drop_tables()

# # text = "This is a test document."
# # query_result = embeddings.embed_query(text)
# # print(len(query_result))
# # doc_result = embeddings.embed_documents([text, "This is not a test document."])
# # print(doc_result)



# docs = [
#     Document(
#         page_content="there are cats in the pond",
#         metadata={"id": 1, "location": "pond", "topic": "animals"},
#     ),
#     Document(
#         page_content="ducks are also found in the pond",
#         metadata={"id": 2, "location": "pond", "topic": "animals"},
#     ),
#     Document(
#         page_content="fresh apples are available at the market",
#         metadata={"id": 3, "location": "market", "topic": "food"},
#     ),
#     Document(
#         page_content="the market also sells fresh oranges",
#         metadata={"id": 4, "location": "market", "topic": "food"},
#     ),
#     Document(
#         page_content="the new art exhibit is fascinating",
#         metadata={"id": 5, "location": "museum", "topic": "art"},
#     ),
#     Document(
#         page_content="a sculpture exhibit is also at the museum",
#         metadata={"id": 6, "location": "museum", "topic": "art"},
#     ),
#     Document(
#         page_content="a new coffee shop opened on Main Street",
#         metadata={"id": 7, "location": "Main Street", "topic": "food"},
#     ),
#     Document(
#         page_content="the book club meets at the library",
#         metadata={"id": 8, "location": "library", "topic": "reading"},
#     ),
#     Document(
#         page_content="the library hosts a weekly story time for kids",
#         metadata={"id": 9, "location": "library", "topic": "reading"},
#     ),
#     Document(
#         page_content="a cooking class for beginners is offered at the community center",
#         metadata={"id": 10, "location": "community center", "topic": "classes"},
#     ),
# ]

# # vectorstore.add_documents(docs, ids=[doc.metadata["id"] for doc in docs])



# docs = [
#     Document(
#         page_content="there are cats in the pond",
#         metadata={"id": 1, "location": "pond", "topic": "animals"},
#     ),
#     Document(
#         page_content="ducks are also found in the pond",
#         metadata={"id": 2, "location": "pond", "topic": "animals"},
#     ),
#     Document(
#         page_content="fresh apples are available at the market",
#         metadata={"id": 3, "location": "market", "topic": "food"},
#     ),
#     Document(
#         page_content="the market also sells fresh oranges",
#         metadata={"id": 4, "location": "market", "topic": "food"},
#     ),
#     Document(
#         page_content="the new art exhibit is fascinating",
#         metadata={"id": 5, "location": "museum", "topic": "art"},
#     ),
#     Document(
#         page_content="a sculpture exhibit is also at the museum",
#         metadata={"id": 6, "location": "museum", "topic": "art"},
#     ),
#     Document(
#         page_content="a new coffee shop opened on Main Street",
#         metadata={"id": 7, "location": "Main Street", "topic": "food"},
#     ),
#     Document(
#         page_content="the book club meets at the library",
#         metadata={"id": 8, "location": "library", "topic": "reading"},
#     ),
#     Document(
#         page_content="the library hosts a weekly story time for kids",
#         metadata={"id": 9, "location": "library", "topic": "reading"},
#     ),
#     Document(
#         page_content="a cooking class for beginners is offered at the community center",
#         metadata={"id": 20, "location": "community center", "topic": "classes"},
#     ),
# ]

# # vectorstore.add_documents(docs, ids=[doc.metadata["id"] for doc in docs])

# # print(vectorstore.similarity_search("kitty", k=2))
