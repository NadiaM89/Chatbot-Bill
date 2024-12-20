from pydantic import BaseModel


# Modelos de entrada


class AskInput(BaseModel):
    """Modelo de entrada para realizar una pregunta."""
    question: str

# Modelos de salida

class AskOutput(BaseModel):
    """Modelo de salida para la respuesta a una pregunta."""
    answer: str
