import models
from database import connect_database, connect_cohere, get_query_embeddings
from cohere import ClassifyExample
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



def classify_text(entrada):
    """
    Clasifica el sentimiento de la entrada del usuario utilizando el modelo de Cohere.

    Args:
    entrada (str): La consulta del usuario en formato de texto.

    Returns:
    list: Una lista con la clasificación del sentimiento de la entrada.
    """
    # Conectar con el servicio de Cohere
    co = connect_cohere()
    
    # Definir ejemplos de clasificación para entrenar el modelo
    examples = [
        ClassifyExample(text="Estoy muy contento con mi progreso en el estudio", label="positive"),
        ClassifyExample(text="¡He entendido completamente este tema!", label="positive"),
        ClassifyExample(text="Me siento preparado para el examen", label="positive"),
        ClassifyExample(text="Las explicaciones del chatbot son muy claras", label="positive"),
        ClassifyExample(text="Estoy motivado para seguir estudiando", label="positive"),
        ClassifyExample(text="No entiendo este concepto, es muy difícil", label="negative"),
        ClassifyExample(text="Me siento frustrado con mi progreso", label="negative"),
        ClassifyExample(text="Estoy preocupado por no pasar el examen", label="negative"),
        ClassifyExample(text="Las respuestas del chatbot no me ayudan", label="negative"),
        ClassifyExample(text="Estoy agotado de tanto estudiar", label="negative"),
        ClassifyExample(text="Necesito más información sobre este tema", label="neutral"),
        ClassifyExample(text="Voy a revisar este capítulo nuevamente", label="neutral"),
        ClassifyExample(text="¿Cuál es el siguiente tema que debo estudiar?", label="neutral"),
        ClassifyExample(text="He terminado de leer el material de estudio", label="neutral"),
        ClassifyExample(text="Voy a tomar un descanso antes de continuar", label="neutral")
    ]
    
    # Clasificar la entrada del usuario utilizando el modelo de Cohere
    response = co.classify(
        model='embed-multilingual-v3.0',
        inputs=entrada,
        examples=examples
    )
    
    # Obtener la clasificación del sentimiento de la respuesta
    sentimiento = response.classifications
    
    return sentimiento




def realizar_reranking(entrada_str, sentimiento):
    """
    Realiza un reordenamiento (reranking) de los documentos recuperados en función del sentimiento de la consulta.

    Args:
    entrada_str (str): La consulta del usuario en formato de texto.
    sentimiento (str): El sentimiento clasificado de la consulta del usuario (puede ser 'negative', 'positive' o 'neutral').

    Returns:
    list: Una lista de documentos reordenados según su relevancia para la consulta.
    """
    # Convertir la entrada en una lista para el procesamiento
    entrada = [entrada_str]
    
    # Conectar con el servicio de Cohere
    co = connect_cohere()
    
    # Obtener los documentos relevantes utilizando la función get_documents
    documentos = get_documents(entrada)

    # Priorizar documentos que contengan guías paso a paso y soluciones a problemas comunes si el sentimiento es negativo
    if sentimiento == 'negative':
        documentos_prioritarios = [doc for doc in documentos if 'guía' in doc or 'solución' in doc]
    else:
        documentos_prioritarios = documentos
    
    # Realizar el reranking de los documentos utilizando el modelo de Cohere
    results = co.rerank(
        query=entrada_str, 
        documents=documentos_prioritarios, 
        top_n=5, 
        model='rerank-multilingual-v3.0'
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
    entrada = [entrada_str]
    
    # Clasificar el sentimiento de la consulta del usuario
    sentimiento = classify_text(entrada)[0].prediction
    
    # Realizar el reranking de documentos basados en la consulta y el sentimiento
    documentos_rerankeados = realizar_reranking(entrada_str, sentimiento)
    
    # Crear un mensaje del sistema personalizado basado en el sentimiento
    if sentimiento == 'negative':
        system_message = """
        ## Task and Context
        Tu nombre es Bill y eres un asistente especializado en la certificación de Microsoft: Azure AI Engineer Associate. 
        Tu objetivo es ayudar a los estudiantes a prepararse para el examen AI-102. 
        ## Style Guide
        Responde de manera clara y comprensiva, especialmente si el usuario parece frustrado o confundido.
        """
    else:
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
    El sentimiento del usuario es {sentimiento}. 
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











