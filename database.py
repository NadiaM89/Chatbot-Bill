import os
import cohere
import chromadb
from dotenv import load_dotenv


# Constantes
COLLECTION_NAME = 'azureAI'
PATH_CONEXION="C:\\Users\\nmask\\OneDrive\\Desktop\\GET_TALENT\\ChromaDB"

def connect_cohere():
    load_dotenv() 
    api_key = os.getenv("COHERE_API_KEY")
    co = cohere.ClientV2()
    return co


    
# Inicialización de ChromaDB
def connect_database(name_coleccion= COLLECTION_NAME, path_coleccion = PATH_CONEXION):
    client = chromadb.PersistentClient(path=path_coleccion)
    collection = client.get_collection(name=name_coleccion)
    return collection


def get_query_embeddings(entrada):
    co = connect_cohere() 
    response = co.embed(
        texts=entrada,
        model="embed-multilingual-v3.0",
        input_type="search_query",
        embedding_types=["float"],
    )
    return response.embeddings.float_




