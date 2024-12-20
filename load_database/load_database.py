import os
import chromadb
import cohere
from dotenv import load_dotenv
from cohere import ClassifyExample
from chromadb import Documents, EmbeddingFunction, Embeddings
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

def get_embeddings(textos):
    """
    Genera embeddings para una lista de textos utilizando el modelo embed-multilingual-v3.0 de Cohere.

    Args:
    textos (list): Una lista de textos para los cuales se generarán los embeddings.

    Returns:
    list: Una lista de embeddings generados para los textos proporcionados.
    """
    # Conectar con el servicio de Cohere
    co = connect_cohere() 
    
    # Generar embeddings para los textos proporcionados
    response = co.embed(
        texts=textos,
        model="embed-multilingual-v3.0",
        input_type="search_document",
        embedding_types=["float"],
    )
    
    return response.embeddings.float_

class MyEmbeddingFunction(EmbeddingFunction):
    """
    Clase personalizada de EmbeddingFunction que utiliza la función get_embeddings para generar embeddings.
    """
    def __call__(self, input: Documents) -> Embeddings:
        """
        Genera embeddings para los documentos proporcionados.

        Args:
        input (Documents): Una lista de documentos para los cuales se generarán los embeddings.

        Returns:
        Embeddings: Una lista de embeddings generados para los documentos proporcionados.
        """
        return get_embeddings(input)

    
# client = chromadb.PersistentClient(path=PATH_CONEXION)
# collection = client.create_collection(name=COLLECTION_NAME,
#                                       embedding_function=MyEmbeddingFunction(),
#                                       metadata={"hnsw:space": "cosine"})

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

def read_pdf(ruta_pdf):
    """
    Lee un archivo PDF y extrae su texto.

    Args:
    ruta_pdf (str): La ruta del archivo PDF a leer.

    Returns:
    str: El texto extraído del archivo PDF.
    """
    # Abrir el archivo PDF en modo de lectura binaria
    with open(ruta_pdf, 'rb') as pdf_file:
        # Crear un lector de PDF
        pdf_reader = PdfReader(pdf_file)
        
        # Inicializar una cadena para almacenar el texto extraído
        texto = ""
        
        # Iterar sobre cada página del PDF
        for page in pdf_reader.pages:
            # Extraer el texto de la página y agregarlo a la cadena de texto
            texto += page.extract_text()
    
    # Devolver el texto extraído
    return texto

def preparar_fragmentos_metadatos(texto_modulo, titulo_modulo):
    """
    Prepara los fragmentos de texto con los metadatos correspondientes.

    Args:
    texto_modulo (str): El texto del módulo a fragmentar.
    titulo_modulo (str): El título del módulo.

    Returns:
    list: Una lista de diccionarios con los fragmentos de texto y sus metadatos.
    """
    # Dividir el texto en fragmentos utilizando RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=259,
        length_function=len,
    )
    chunks = text_splitter.split_text(texto_modulo)
    
    # Preparar los fragmentos con los metadatos
    chunks_with_metadata = []
    for i, chunk in enumerate(chunks):
        metadata = {
            "title": titulo_modulo,
        }
        chunks_with_metadata.append({"text": chunk, "metadata": metadata})
    
    return chunks_with_metadata

# Diccionario que asocia los títulos de los módulos con las rutas de los archivos PDF correspondientes
lista_pdfs = {
    "Introducción a Servicios de Azure AI": "C:\\Users\\nmask\\OneDrive\\Desktop\\GET_TALENT\\Modulo1 .pdf",
    "Creación de soluciones de visión artificial con Visión de Azure AI": "C:\\Users\\nmask\\OneDrive\\Desktop\\GET_TALENT\\Modulo 2.pdf",
    "Desarrollo de soluciones de procesamiento del lenguaje natural con Servicios de Azure AI": "C:\\Users\\nmask\\OneDrive\\Desktop\\GET_TALENT\\Modulo 3.pdf",
    "Implementación de la minería de conocimiento con Búsqueda de Azure AI": "C:\\Users\\nmask\\OneDrive\\Desktop\\GET_TALENT\\Modulo 4.pdf",
    "Desarrollo de soluciones con Documento de inteligencia de Azure AI": "C:\\Users\\nmask\\OneDrive\\Desktop\\GET_TALENT\\Modulo 5.pdf",
    "Desarrollo de soluciones de inteligencia artificial generativa con Azure OpenAI Service": "C:\\Users\\nmask\\OneDrive\\Desktop\\GET_TALENT\\Modulo 6.pdf"
}

# Lista para almacenar todos los fragmentos de texto de los documentos
all_chunks = []

# Iterar sobre los elementos de lista_pdfs para procesar cada documento
for key in lista_pdfs:
    # Leer el texto del módulo desde el archivo PDF
    texto_modulo = read_pdf(lista_pdfs[key])
    
    # Obtener el título del módulo
    titulo_modulo = lista_pdfs[key]
    
    # Preparar los fragmentos de texto con metadatos
    chunks_modulo = preparar_fragmentos_metadatos(texto_modulo, titulo_modulo)
    
    # Agregar los fragmentos a la lista all_chunks
    all_chunks = all_chunks + chunks_modulo

# Calcular el total de caracteres en todos los fragmentos
total_caracteres = sum(len(chunk["text"]) for chunk in all_chunks)

# Imprimir el total de caracteres en all_chunks
print(f"Total de caracteres en all_chunks: {total_caracteres}")

# Conectar a la base de datos y obtener la colección
collection = connect_database()

try:
    # Agregar los fragmentos de texto y sus metadatos a la colección
    collection.add(
        documents=[chunk["text"] for chunk in all_chunks],
        ids=[str(i+1) for i in range(len(all_chunks))],
        metadatas=[chunk["metadata"] for chunk in all_chunks]
    )
    print("Datos añadidos correctamente.")
except Exception as e:
    # Manejar cualquier error que ocurra al agregar los fragmentos
    print(f"Error al agregar los fragmentos: {e}")

# Consultar el total de IDs en la colección
total_ids = collection.count()

# Imprimir el total de IDs en la colección
print(f"Total de IDs en la colección: {total_ids}")

