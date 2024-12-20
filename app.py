import streamlit as st
import requests

st.title("BILL")

# Entrada del usuario
user_input = st.text_input("Haz una pregunta:")

if st.button("Enviar"):
    # Realizar la solicitud a la API de FastAPI
    response = requests.post("http://127.0.0.1:8000/ask/", json={"question": user_input})
    
    if response.status_code == 200:
        answer = response.json()["answer"]
        st.write(f"Respuesta: {answer}")
    else:
        st.write("Error al obtener la respuesta. Inténtalo de nuevo.")


