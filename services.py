import models
from database import connect_database, connect_cohere, get_query_embeddings
from cohere import ClassifyExample
# Variables globales
# Historial global (en un entorno de producción, esto debería manejarse de manera más robusta)
historial_global = []


def get_documents(entrada):
    # Obtener los embeddings de la entrada
    embedding_entrada = get_query_embeddings(entrada)
    collection = connect_database()
    
    # Realizar la consulta en la colección
    results = collection.query(
        query_embeddings=embedding_entrada, 
        n_results=10, 
        include=["documents", "metadatas", "distances"]  
    )
    
    # Acceder directamente al documento
    documentos = results['documents'][0]
    return documentos


def classify_text(entrada):
    co = connect_cohere()
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
    response = co.classify(
        model='embed-multilingual-v3.0',
        inputs=entrada,
        examples=examples
    )
    sentimiento = response.classifications
    return sentimiento



def realizar_reranking(entrada_str, documentos, sentimiento):
    co = connect_cohere()
    if sentimiento == 'negative':
        # Priorizar documentos que contengan guías paso a paso y soluciones a problemas comunes
        documentos_prioritarios = [doc for doc in documentos if 'guía' in doc or 'solución' in doc]
    else:
        documentos_prioritarios = documentos
    
    results = co.rerank(query=entrada_str, 
                        documents=documentos_prioritarios, 
                        top_n=5, model='rerank-multilingual-v3.0')
    
    documentos_rerankeados = [documentos[result.index] for result in results.results]
    return documentos_rerankeados

def rag_answer(entrada_str):
    co = connect_cohere()
    global historial_global
    historial_limitado = historial_global[-10:]
    entrada = [entrada_str]
    # Obtener documentos relevantes
    documentos = get_documents(entrada)
    sentimiento = classify_text(entrada)[0].prediction
    # Realizar el reranking de documentos
    documentos_rerankeados = realizar_reranking(entrada_str, documentos,sentimiento)
    
    # Crear un mensaje del sistema personalizado basado en el sentimiento
    if sentimiento == 'negative':
        system_message = """
        ## Task and Context
        Tu nombre es Bill y eres un asistente especializado en la certificación de Microsoft: Azure AI Engineer Associate. 
        Tu objetivo es ayudar a los estudiantes a prepararse para el examen AI-102. 
        ##Style Guide
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
    
    respuesta = response.message.content[0].text
    historial_global.append({'role': 'user', 'content': user_message})
    historial_global.append({'role': 'assistant', 'content': respuesta})

    return respuesta


def chatbot(entrada_str):
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
    
    entrada_lower = entrada_str.lower()
    
    if entrada_lower in respuestas_basicas:
        return respuestas_basicas[entrada_lower]
    
    return rag_answer(entrada_str)












