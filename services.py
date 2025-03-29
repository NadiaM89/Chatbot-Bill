import models
from database import connect_database, connect_cohere, get_query_embeddings

# Variables globales
# Historial global (en un entorno de producción, esto debería manejarse de manera más robusta)
historial_global = []


def get_documents(entrada):
    """
    Obtiene los documentos más relevantes de la base de datos ChromaDB 
    basados en la similitud de la consulta del usuario.

    Args:
    entrada (str): La consulta del usuario en formato de texto.

    Returns:
    list: Una lista de documentos relevantes.
    """
    # Obtener los embeddings de la entrada utilizando la función get_query_embeddings
    embedding_entrada = get_query_embeddings(entrada)
    
    # Conectar a la base de datos (ChromaDB)
    collection = connect_database()
    
    # Realizar la consulta en la colección utilizando los embeddings generados
    results = collection.query(
        query_embeddings=embedding_entrada,  # Los embeddings de la consulta
        n_results=10,  # Número de documentos más relevantes a devolver
        include=["documents", "metadatas", "distances"]  # Incluir el documento, metadatos y distancia
    )
    
    # Acceder directamente a los documentos más relevantes
    documentos = results['documents'][0]
    
    return documentos



def realizar_reranking(entrada_str):
    """
    Realiza un reordenamiento (reranking) de los documentos recuperados 
    en función de su relevancia para la consulta, sin clasificar el sentimiento.

    Args:
    entrada_str (str): La consulta del usuario en formato de texto.

    Returns:
    list: Una lista de documentos reordenados según su relevancia para la consulta.
    """
    # Convertir la entrada en una lista para el procesamiento
    entrada = [entrada_str]
    
    # Conectar con el servicio de Cohere
    co = connect_cohere()
    
    # Obtener los documentos relevantes utilizando la función get_documents
    documentos = get_documents(entrada)
    
    # Realizar el reranking de los documentos utilizando el modelo de Cohere
    results = co.rerank(
        query=entrada_str, 
        documents=documentos,  # Documentos obtenidos directamente
        top_n=5,  # Número de resultados reordenados que deseas obtener
        model='rerank-multilingual-v3.0'  # Modelo de Cohere para reranking
    )
    
    # Obtener los documentos rerankeados basados en los resultados del reranking
    documentos_rerankeados = [documentos[result.index] for result in results.results]
    
    return documentos_rerankeados


def rag_answer(entrada_str):
    """
    Genera una respuesta a la consulta del usuario utilizando la arquitectura RAG (Retrieval-Augmented Generation).

    Args:
    entrada_str (str): La consulta del usuario en formato de texto.

    Returns:
    str: La respuesta generada basada en la consulta y los documentos relevantes.
    """
    # Conectar con el servicio de Cohere
    co = connect_cohere()
    
    # Utilizar el historial global de la conversación
    global historial_global
    historial_limitado = historial_global[-10:]  # Limitar el historial a las últimas 10 interacciones
    
    # Realizar el reranking de documentos basados únicamente en la consulta
    documentos_rerankeados = realizar_reranking(entrada_str)
    
    # Crear un mensaje del sistema
    system_message = """
    ## Task and Context
    Eres un asistente especializado en la certificación de Microsoft: Azure AI Engineer Associate. 
    Tu nombre es Bill, inspirado en Bill Gates, el fundador de Microsoft. Tu objetivo es ayudar a los estudiantes a prepararse para el examen AI-102: Designing and Implementing an Azure AI Solution. 
    Esta certificación está dirigida a ingenieros de IA que crean, administran e implementan soluciones de IA utilizando servicios de Azure AI, Azure AI Search y Azure OpenAI.
    ## Style Guide
    Responde de manera clara, precisa y animada.
    """
    
    # Crear el mensaje del usuario
    user_message = f"""
    Responde a esta pregunta o pedido: {entrada_str}, en español, utilizando esta información: {documentos_rerankeados}. 
    Proporciona la respuesta en un máximo de 12 oraciones.
    Si la pregunta no está relacionada con la certificación, responde: "Lo siento, solo puedo responder preguntas relacionadas con la certificación Azure AI Engineer Associate de Microsoft."
    """
    
    # Generar la respuesta final usando los documentos rerankeados
    messages = [
        {'role': 'system', 'content': system_message}
    ]
    messages.extend(historial_limitado)
    messages.append({'role': 'user', 'content': user_message})
    
    response = co.chat(
        model="command-r-plus-08-2024",
        messages=messages,
        temperature=0,
        seed=42
    )
    
    # Obtener el texto de la respuesta generada
    respuesta = response.message.content[0].text
    
    # Actualizar el historial global con la nueva interacción
    historial_global.append({'role': 'user', 'content': user_message})
    historial_global.append({'role': 'assistant', 'content': respuesta})

    return respuesta




def chatbot(entrada_str):
    """
    Procesa la entrada del usuario y devuelve una respuesta adecuada. Si la entrada es un saludo, 
    devuelve una respuesta predefinida. De lo contrario, utiliza la función rag_answer para generar una respuesta.

    Args:
    entrada_str (str): La consulta o saludo del usuario en formato de texto.

    Returns:
    str: La respuesta generada basada en la entrada del usuario.
    """
    # Diccionario de respuestas básicas para saludos comunes
    respuestas_basicas = {
        "hola": "¡Hola! ¿En qué puedo ayudarte hoy?",
        "hola!": "¡Hola! ¿En qué puedo ayudarte hoy?",
        "buen día": "¡Buen día! ¿Cómo puedo asistirte?",
        "buenos días": "¡Buenos días! ¿Cómo puedo asistirte?",
        "buenas tardes": "¡Buenas tardes! ¿Qué necesitas saber?",
        "buenas noches": "¡Buenas noches! ¿En qué puedo ayudarte?",
        "adiós": "¡Adiós! Que tengas un buen día.",
        "hasta luego": "¡Hasta luego! Cuídate.",
        "nos vemos": "¡Nos vemos! Que tengas un buen día.",
        "chau": "¡Chau! Espero verte pronto."
    }
    
    # Convertir la entrada del usuario a minúsculas para una comparación más sencilla
    entrada_lower = entrada_str.lower()
    
    # Verificar si la entrada del usuario es un saludo y devolver la respuesta correspondiente
    if entrada_lower in respuestas_basicas:
        return respuestas_basicas[entrada_lower]
    
    # Si la entrada no es un saludo, generar una respuesta utilizando la función rag_answer
    return rag_answer(entrada_str)











