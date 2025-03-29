import streamlit as st
import requests

# Título con emoji de lentes al lado del nombre "BILL"
st.title("BILL 🤓")

# Introducción y datos personales con tamaño más pequeño
st.markdown('<h6>Creado por Nadia Maskarinec 🧠</h6>', unsafe_allow_html=True)


# Entrada del usuario con tamaño más grande
st.markdown('<h1>¡Hazme una pregunta!</h1>', unsafe_allow_html=True)
user_input = st.text_input("", key="user_input")

if st.button("Enviar"):
    # Realizar la solicitud a la API de FastAPI
    response = requests.post("http://127.0.0.1:8000/ask/", json={"question": user_input})
    
    if response.status_code == 200:
        answer = response.json()["answer"]
        st.write(f"{answer}")
    else:
        st.write("Error al obtener la respuesta. Inténtalo de nuevo.")

# Pie de página con tamaño más pequeño
st.markdown('<p><a href="https://www.linkedin.com/in/nadia-maskarinec/">🔗 Mi LinkedIn </a></p>', unsafe_allow_html=True)
st.markdown('<h6>📍 Creado por Nadia Maskarinec - Córdoba, Argentina.</h6>', unsafe_allow_html=True)

