from fastapi import APIRouter, HTTPException, status
from models import AskInput, AskOutput
from services import chatbot

router = APIRouter()

@router.post("/ask/", tags=['Consulta'], status_code=status.HTTP_200_OK, response_model=AskOutput)
async def ask_question(ask: AskInput):
    """Endpoint para realizar una pregunta."""
    try:
        respuesta = chatbot(ask.question)
        return AskOutput(answer=respuesta)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


