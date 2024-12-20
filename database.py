import os
import cohere
import chromadb
from dotenv import load_dotenv


# Constantes
COLLECTION_NAME = 'azureAI'
PATH_CONEXION="C:\\Users\\nmask\\OneDrive\\Desktop\\GET_TALENT\\ChromaDB"

def connect_cohere():
    """
    Conecta con el servicio de Cohere utilizando la clave API almacenada en el archivo .env.

    Returns:
    cohere.ClientV2: Una instancia del cliente Cohere.
    """
    # Cargar las variables de entorno desde el archivo .env
    load_dotenv() 
    
    # Obtener la clave API de Cohere desde las variables de entorno
    api_key = os.getenv("COHERE_API_KEY")
    
    # Crear una instancia del cliente Cohere
    co = cohere.ClientV2(api_key)
    
    return co

# Inicialización de ChromaDB
def connect_database(name_coleccion=COLLECTION_NAME, path_coleccion=PATH_CONEXION):
    """
    Conecta a la base de datos ChromaDB y obtiene la colección especificada.

    Args:
    name_coleccion (str): El nombre de la colección a obtener.
    path_coleccion (str): La ruta de la base de datos.

    Returns:
    Collection: La colección obtenida de la base de datos.
    """
    # Crear una instancia del cliente persistente de ChromaDB
    client = chromadb.PersistentClient(path=path_coleccion)
    
    # Obtener la colección especificada
    collection = client.get_collection(name=name_coleccion)
    
    return collection

def get_query_embeddings(entrada):
    """
    Genera embeddings para una lista de textos de consulta utilizando el modelo embed-multilingual-v3.0 de Cohere.

    Args:
    entrada (list): Una lista de textos de consulta para los cuales se generarán los embeddings.

    Returns:
    list: Una lista de embeddings generados para los textos de consulta proporcionados.
    """
    # Conectar con el servicio de Cohere
    co = connect_cohere() 
    
    # Generar embeddings para los textos de consulta proporcionados
    response = co.embed(
        texts=entrada,
        model="embed-multilingual-v3.0",
        input_type="search_query",
        embedding_types=["float"],
    )
    
    return response.embeddings.float_




