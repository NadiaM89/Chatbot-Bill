from fastapi import APIRouter, HTTPException, status
from models import AskInput, AskOutput
from services import chatbot

# Crear una instancia del router de FastAPI
router = APIRouter()

@router.post("/ask/", tags=['Consulta'], status_code=status.HTTP_200_OK, response_model=AskOutput)
async def ask_question(ask: AskInput):
    """
    Endpoint para realizar una pregunta.

    Args:
    ask (AskInput): Objeto que contiene la pregunta del usuario.

    Returns:
    AskOutput: Objeto que contiene la respuesta generada por el chatbot.
    """
    try:
        # Generar la respuesta utilizando la función chatbot
        respuesta = chatbot(ask.question)
        
        # Devolver la respuesta en el formato especificado por AskOutput
        return AskOutput(answer=respuesta)
    except Exception as e:
        # Manejar cualquier excepción que ocurra y devolver un error HTTP 500
        raise HTTPException(status_code=500, detail=str(e))


